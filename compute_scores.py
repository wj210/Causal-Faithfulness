import numpy as np
import json,pickle,os
from collections import defaultdict
from scipy import spatial
from utils.causal_trace import find_token_range
from utils.model_data_utils import get_model_path
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,pointbiserialr

"""
metric = 
1. expl_output - metric 1 compares the output and explanation directly
2. FEC/input_output_p_cf - metric 2 compares the paraphrased/cf explanations divg with the input divg
3. paraphrase - measure faithfulness via paraphrasing the NLE
4. mistake - measure faithfulness via adding mistakes to NLE
5. cf edit - measure faithfulness via checking for cf edits in the cf NLE.
6. CC-SHAP - measure divg of SHAP vals between output and explanation
"""

def plot_line(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x,y)
    plt.xlabel('FEAC')
    plt.ylabel('noise standard deviation')
    plt.savefig('plots/ablate_noise.png')

def sort_into_lists(x,num_list = 3): # x is a list of list
    out = [[] for _ in range(num_list)]
    for i,xx in enumerate(x):
        out[i].append(xx[i])
    return out

def find_common_sample(s_path,s2_path):
    with open(s_path,'rb') as f:
        s = pickle.load(f)
    with open(s2_path,'rb') as f:
        s2 = pickle.load(f)
    common = []
    for k in s.keys():
        if k in s2:
            common.append(k)
    return set(common)

def prune_sequence(seq1, seq2): # seq1 is the longer sequence, seq2 is the shorter sequence
    i, j = 0, 0
    original_indices = list(range(len(seq1)))  # Track original positions
    pruned_positions = []  # List to store the positions of pruned tokens
    
    while j < len(seq2):
        if i < len(seq1) and seq1[i] == seq2[j]:
            i += 1
            j += 1
        else:
            pruned_positions.append(original_indices[i])
            seq1.pop(i)
            original_indices.pop(i)
    
    # Any remaining tokens in seq1 after seq2 is fully matched should be pruned
    while i < len(seq1):
        pruned_positions.append(original_indices[i])
        seq1.pop(i)
        original_indices.pop(i)

    return seq1, pruned_positions


class ScoreObj:
    def __init__(self,scores,subject_range,input_tokens):
        self.scores = scores
        self.subject_range = subject_range
        self.input_tokens = input_tokens

def compute_divg(o,e,compare = 'o-e'):
    if compare == 'o_e': # if output explanation, e is the explanation, take until the length of output
        e = e[:o.shape[0]]
    else: # else is either o-o or e-e, the input length of the paraphrased/cf may have difference of 1 token due to tokenizing, just take until the shortest.
        shortest = min(o.shape[0],e.shape[0])
        o = o[:shortest]
        e = e[:shortest]
    if len(o.shape) > 1:
        o = o.flatten()
        e = e.flatten()
    return 1. - spatial.distance.cosine(o, e)

def unpack_scores(x,attr='scores'):
    return [xx[attr] for xx in x]

def compute_full_divg(o_path,e_path,common_ids,ds_path,tokenizer):
    with open(o_path,'rb') as f:
        o_r = pickle.load(f)
    with open(e_path,'rb') as f:
        e_r = pickle.load(f)
    
    with open(ds_path,'r') as f:
        ds = [json.loads(l) for l in f.readlines()]
    ds = {d['sample_id']:d for d in ds}

    all_scores = {}
    all_samples = {}
    for s_id,o_d in o_r.items():
        if s_id not in common_ids:
            continue
        e_d = e_r.get(s_id,None)
        if not e_d:
            continue
        o_score = o_d['scores']
        e_score = e_d['scores']
        
        if 'consistency' in ds_path:
            subj = ds[s_id]['paraphrase_subject']
        else:
            subj = ds[s_id]['subject']
        subj_range = find_token_range(tokenizer,o_d['input_tokens'],subj)

        o_store = ScoreObj(o_score,subj_range,o_d['input_tokens'])
        e_store = ScoreObj(e_score,subj_range,e_d['input_tokens'])
        dist_scores = compute_divg(o_score,e_score) 
        
        all_scores[s_id] = dist_scores
        all_samples[s_id] = (o_store,e_store)
    
    total_mean = np.mean(list(all_scores.values()))
    
    return all_samples,total_mean,all_scores

def average_subj_scores(s_obj,s_obj2):
    """
    This is mainly used for FEC/cf where the 2 inputs may not match in size, ie the cf/p subject may have different number of tokens.
    We average the contributions of the subject as a whole, and ensure the total size is similar and comparable.
    vec are the attribution score for each token. [token len, num layers]
    """
    avg_objs = []
    for obj in [s_obj,s_obj2]:
        if isinstance(obj,dict):
            obj = ScoreObj(obj['scores'],obj['subject_range'],obj['input_tokens'])
        subj_s,subj_e = obj.subject_range
        vec = obj.scores
        subj_vec = vec[subj_s:subj_e].mean(axis=0).reshape(1,-1)
        obj_scores = np.concatenate([vec[:subj_s],subj_vec,vec[subj_e:]],axis=0)
        avg_objs.append(obj_scores)
    if avg_objs[0].shape[0] != avg_objs[1].shape[0]: # means input_tokens not aligned somehow due to tokenization etc.
        # Find the longer seq and take away the dissimilar token apart from the averaged subject
        placeholder_token = ["[PAD]"] # to stand in for the avg token
        if avg_objs[0].shape[0] > avg_objs[1].shape[0]:
            longer = 0
            longer_seq = s_obj.input_tokens[:s_obj.subject_range[0]] + placeholder_token + s_obj.input_tokens[s_obj.subject_range[1]:]
            shorter_seq = s_obj2.input_tokens[:s_obj2.subject_range[0]] + placeholder_token + s_obj2.input_tokens[s_obj2.subject_range[1]:]
        else:
            longer = 1
            longer_seq= s_obj2.input_tokens[:s_obj2.subject_range[0]] + placeholder_token + s_obj2.input_tokens[s_obj2.subject_range[1]:]
            shorter_seq = s_obj.input_tokens[:s_obj.subject_range[0]] + placeholder_token + s_obj.input_tokens[s_obj.subject_range[1]:]
        longer_seq,pruned_positions = prune_sequence(longer_seq,shorter_seq)
        if longer == 0:
            avg_objs[0] = np.delete(avg_objs[0],pruned_positions,axis=0)
        else:
            avg_objs[1] = np.delete(avg_objs[1],pruned_positions,axis=0)
    return avg_objs

def main():
    ablate_types = ['s1','s2','s4','s5','original']
    total_corr_count = defaultdict(lambda: defaultdict(int))
    total_noise = defaultdict(list)
    for model_name in ['llama3-8B','llama3-8B-chat','gemma-2B-chat','gemma-2B','gemma2-27B-chat']:
    # for model_name in ['gemma-2B']:
        for expl_type in ['post_hoc','cot']:
            model_path,_ = get_model_path(model_name,'post_hoc')
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if expl_type == 'cot':
                metrics = ['FEAC','FEC','ccshap','accuracy','paraphrase','mistake','early_answering','biased']
            else:
                metrics = ['FEAC','FEC','ccshap','accuracy','noise_level','cf_edit']
                
            assert set(metrics).issubset(set(['ccshap','FEAC','FEC','input_output_cf','paraphrase','mistake','cf_edit','early_answering','biased','ablate','accuracy','noise_level']))
            num_seeds = 3 if not (expl_type == 'cot' and model_name == 'gemma-2B') else 1

            ds_scores = {}
            overall_corr = defaultdict(list)
            result_path = f'results/{model_name}/{expl_type}.txt'
            os.makedirs(os.path.dirname(result_path),exist_ok=True)
            ds_set = ['arc','esnli','csqa'] if 'ablate' not in metrics else ['csqa']
            for ds in ds_set:
                prediction_dir = f'prediction/{model_name}/{ds}'
                # get the common_ids
                ori_path = os.path.join(f"{prediction_dir}/0",f'{expl_type}_output_original.pkl')
                edit_path = os.path.join(f"{prediction_dir}/0",f'{expl_type}_output_paraphrase.pkl')
                if not os.path.exists(ori_path) or not os.path.exists(edit_path):
                    print (f"Skipping {ds} as the files do not exist")
                    continue
                common_ids = find_common_sample(ori_path,edit_path)

                all_scores = defaultdict(list)
                all_sample_scores = defaultdict(list)
                for metric in metrics:
                    for seed in range(num_seeds):
                        prediction_dir_seed = os.path.join(prediction_dir,str(seed))
                        ori_ds_path = f'data/original/{seed}/{ds}/{model_name}_{expl_type}.jsonl'
                        p_ds_path = f'data/consistency/{seed}/{ds}/{model_name}_{expl_type}_paraphrase.jsonl'
                        ds_path = {'original':ori_ds_path,'paraphrase':p_ds_path}

                        if metric in ['FEAC','FEC','input_output_cf','ablate']:
                            output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_original.pkl')
                            expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_original.pkl')

                            consis_output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_paraphrase.pkl')
                            consis_expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_paraphrase.pkl')

                            original_samples,original_m,original_sampled_m = compute_full_divg(output_path,expl_path,common_ids,ori_ds_path,tokenizer)
                            consis_samples,consis_m,_ = compute_full_divg(consis_output_path,consis_expl_path,common_ids,p_ds_path,tokenizer)
                            
                            if metric == 'FEAC':
                                all_scores[metric].append(original_m)
                                all_sample_scores['FEAC'].append(original_sampled_m)

                            elif metric == 'FEC':
                                all_scores[f'{metric}_ind'].append(consis_m)

                                # Match via the sample_id
                                """
                                each o_s is a list of attributions (1 for PH, num_samples for CoT) (3 for e_s)
                                For CoT, there may be uneven samples between original and cf/paraphrase since some CoT may not be correctly formatted.
                                """
                                metric_2_samples = {}
                                for sample_id,(o_s,e_s) in original_samples.items():
                                    if sample_id not in consis_samples:
                                        continue
                                    c_o_s,c_e_s = consis_samples[sample_id]
                                    # print (sample_id,ds)
                                    if o_s.scores.shape[0] != c_o_s.scores.shape[0]: # do the averaging here to match
                                        try:
                                            o_s,c_o_s = average_subj_scores(o_s,c_o_s)
                                            e_s,c_e_s = average_subj_scores(e_s,c_e_s)
                                        except Exception as e:
                                            continue
                                    else:
                                        o_s,e_s,c_o_s,c_e_s = o_s.scores,e_s.scores,c_o_s.scores,c_e_s.scores

                                    output_divg = compute_divg(o_s,c_o_s,compare = 'o_o') 
                                    expl_divg = compute_divg(e_s,c_e_s,compare = 'e_e') 
                                    # metric_2_samples[sample_id] = 1. - np.abs(output_divg - expl_divg)
                                    # output_divg = o_s - c_o_s
                                    # expl_divg = e_s - c_e_s
                                    metric_2_samples[sample_id] = expl_divg
                                    
                                all_sample_scores['FEC'].append(metric_2_samples)
                                all_scores[metric].append(np.mean(list(metric_2_samples.values())))

                            elif metric == 'ablate':
                                plot_inputs = []
                                for ablate_type in ablate_types: # add subject_1 as well 
                                    ablate_output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_original_{ablate_type}.pkl')
                                    ablate_expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_original_{ablate_type}.pkl')
                                    if not os.path.exists(ablate_output_path):
                                        continue
                                    _,ablate_m,_ = compute_full_divg(ablate_output_path,ablate_expl_path,common_ids,ori_ds_path,tokenizer)
                                    plot_inputs.append((int(ablate_type[-1]),ablate_m))
                                    all_scores[f"{metric}_{ablate_type}"].append(ablate_m)
                                plot_inputs.append((3,original_m))
                                plot_inputs = sorted(plot_inputs,key = lambda x: x[0])
                                plot_line(plot_inputs)

                        elif metric == 'accuracy':
                            for ds_type,path_name in ds_path.items():
                                with open(path_name,'r') as f:
                                    ds_ = [json.loads(l) for l in f.readlines()]
                                
                                correct_ = np.mean([d['correct'] for d in ds_ if d['sample_id'] in common_ids])
                                all_scores[f'{metric}_{ds_type}'].append(correct_)
                        elif metric == 'noise_level':
                            with open(ori_ds_path,'r') as f:
                                ori_ds = [json.loads(l) for l in f.readlines()]
                            
                            noise_levels = np.mean([d['difference'] for d in ori_ds if d['sample_id'] in common_ids])
                            all_sample_scores[metric].append({d['sample_id']:d['difference'] for d in ori_ds if d['sample_id'] in common_ids})
                            all_scores[metric].append(noise_levels)
                            total_noise[expl_type].append(noise_levels)
                                    
                        else:
                            if metric not in ['paraphrase','mistake','early_answering','biased']:
                                attack_path = os.path.join(prediction_dir_seed,f'{expl_type}_{metric}.pkl')
                            else:
                                assert expl_type == 'cot'
                                attack_path = os.path.join(prediction_dir_seed,f'{metric}.pkl')

                            with open(attack_path,'rb') as f:
                                perturb_scores = pickle.load(f)
                            if metric != 'cf_edit':
                                selected_perturb_scores =  {k:v for k,v in perturb_scores.items() if k in common_ids}
                            else:
                                selected_perturb_scores = [(k,v) for k,v in perturb_scores.items()][:100]
                                selected_perturb_scores = {k:v for k,v in selected_perturb_scores}
                            all_sample_scores[metric].append(selected_perturb_scores)
                            all_scores[metric].append(np.mean(list(selected_perturb_scores.values())))
                    ## end of seeds
                    
                    # Average over seeds for correlation
                    averaged_sampled_scores = defaultdict(list)
                    for seed_run in all_sample_scores[metric]:
                        for sample_id,score in seed_run.items():
                            averaged_sampled_scores[sample_id].append(score)
                    all_sample_scores[metric] = {k:np.mean(v) for k,v in averaged_sampled_scores.items()}
                    
                    
                    if metric not in ['accuracy','ablate']:
                        all_scores[metric] = (np.mean(all_scores[metric]),np.std(all_scores[metric]))

                    elif metric == 'accuracy':
                        for k in ['original','paraphrase']:
                            all_scores[f'accuracy_{k}'] = (np.mean(all_scores[f'accuracy_{k}']),np.std(all_scores[f'accuracy_{k}']))
                    
                    elif metric == 'ablate':
                        for ablate_type in ablate_types:
                            all_scores[f"{metric}_{ablate_type}"] = (np.mean(all_scores[f"{metric}_{ablate_type}"]),0.)
                    else:
                        for k,v in all_scores.items():
                            all_scores[k] = (v[0],0.0)
                    if metric in ['FEC','input_output_cf']:
                        all_scores[f'{metric}_ind'] = (np.mean(all_scores[f'{metric}_ind']),np.std(all_scores[f'{metric}_ind']))
                        
                ds_scores[ds] = all_scores
                
                ## compute the correlations
                ## FEAC, FEC vs all
                for source_metric in ['FEAC','FEC']:
                    for target_metric in [m for m in metrics if m not in ['FEAC','FEC','accuracy','noise_level','ablate']]:
                        all_x,all_y = [],[]
                        for sm,sv in all_sample_scores[source_metric].items():
                            if sm in all_sample_scores[target_metric]:
                                all_x.append(sv)
                                all_y.append(all_sample_scores[target_metric][sm])
                        if len(all_x) == 0:
                            overall_corr[f"{source_metric}_{target_metric}"].append(-100)
                        else:
                            if target_metric in ['paraphrase','mistake','early_answering','biased','cf_edit']:
                                overall_corr[f"{source_metric}_{target_metric}"].append(pointbiserialr(all_x,all_y)[0])
                            else:
                                overall_corr[f"{source_metric}_{target_metric}"].append(pearsonr(all_x,all_y)[0])
                ## FEAC vs FEC
                all_x,all_y = [],[]
                for k,v in all_sample_scores['FEC'].items():
                    all_x.append(v)
                    all_y.append(all_sample_scores['FEAC'][k])
                overall_corr['FEC_FEAC'].append(pearsonr(all_x,all_y)[0])         
                
            with open(result_path,'w') as f:
                for ds,ds_score in ds_scores.items():
                    f.write('-'*80+'\n'+'Dataset: {}\n'.format(ds)+ '-'*80+'\n')
                    for k,v in ds_score.items():
                        msg = f'{k}: {v[0]:.3f} +/- {v[1]:.3f}, relative std: {abs(v[1]/v[0]):.3f}\n'
                        f.write(msg)
                        # print (msg.strip())
                f.write('\n\nCorrelation:\n')
                for k,v in overall_corr.items():
                    if k != 'FEAC_noise_level':
                        num_pos = len([vv for vv in v if vv > 0])
                        num_neg = len([vv for vv in v if vv < 0])
                        total_corr_count[k]['pos'] += num_pos
                        total_corr_count[k]['neg'] += num_neg
                        msg = f"{k} corr: pos: {num_pos}, neg: {num_neg}\n"
                        f.write(msg)
                    else:
                        msg = f"{k} corr: {[np.round(vv,3) for vv in v]}\n"
                        f.write(msg)
    for metric_type,pos_neg in total_corr_count.items():
        pos_neg_ratio = pos_neg['pos']/(pos_neg['pos']+pos_neg['neg'])
        print (f"{metric_type}, pos_neg_ratio: {pos_neg_ratio:.3f}")
    
    for expl_type,noise_vals in total_noise.items():
        print (f"{expl_type} noise mean: {np.mean(noise_vals):.3f}")
        

if __name__ == '__main__':
    main()
    





        

    



