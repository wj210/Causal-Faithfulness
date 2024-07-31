import numpy as np
import json,pickle,os
from collections import defaultdict
from scipy import spatial
from utils.causal_trace import find_token_range

"""
metric = 
1. expl_output - metric 1 compares the output and explanation directly
2. input_output_p/input_output_p_cf - metric 2 compares the paraphrased/cf explanations divg with the input divg
3. paraphrase - measure faithfulness via paraphrasing the NLE
4. mistake - measure faithfulness via adding mistakes to NLE
5. cf edit - measure faithfulness via checking for cf edits in the cf NLE.
6. CC-SHAP - measure divg of SHAP vals between output and explanation
"""

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
    return common


class ScoreObj:
    def __init__(self,scores,subject_range):
        self.scores = scores
        self.subject_range = subject_range

def compute_divg(o,e,compare = 'o-e'):
    if len(o.shape) > 1:
        o = o.flatten()
        e = e.flatten()
    if compare == 'o_e': # if output explanation, e is the explanation, take until the length of output
        e = e[:o.shape[0]]
    else: # else is either o-o or e-e, the input length of the paraphrased/cf may have difference of 1 token due to tokenizing, just take until the shortest.
        shortest = min(o.shape[0],e.shape[0])
        o = o[:shortest]
        e = e[:shortest]
    return 1. - spatial.distance.cosine(o, e)

def unpack_scores(x,attr='scores'):
    return [xx[attr] for xx in x]

def compute_full_divg(o_path,e_path,common_ids):
    with open(o_path,'rb') as f:
        o_r = pickle.load(f)
    with open(e_path,'rb') as f:
        e_r = pickle.load(f)


    all_scores = []
    all_samples = {}
    for s_id,o_d in o_r.items():
        if s_id not in common_ids:
            continue
        e_d = e_r.get(s_id,None)
        if not e_d:
            continue
        o_score = o_d['scores']
        e_score = e_d['scores']

        o_store = ScoreObj(o_score,o_d['subject_range'])
        e_store = ScoreObj(e_score,e_d['subject_range'])
        dist_scores = compute_divg(o_score,e_score) 
        all_scores.append(dist_scores)
        all_samples[s_id] = (o_store,e_store)
    
    total_mean = np.mean(all_scores)
    
    return all_samples,total_mean

def average_subj_scores(s_obj):
    """
    This is mainly used for input_output_p/cf where the 2 inputs may not match in size, ie the cf/p subject may have different number of tokens.
    We average the contributions of the subject as a whole, and ensure the total size is similar and comparable.
    vec are the attribution score for each token. [token len, num layers]
    """
    subj_s,subj_e = s_obj.subject_range
    vec = s_obj.scores
    subj_vec = vec[subj_s:subj_e].mean(axis=0).reshape(1,-1)
    s_obj.scores = np.concatenate([subj_vec,vec[subj_e:]],axis=0)
    return s_obj

def main():
    ablate_types = ['noise_1','noise_2','noise_4','noise_5','subject_1','subject_-1' ,'subject_back_1' ,'subject_2']
    # metrics = ['ablate']
    
    expl_type = 'post_hoc'
    # expl_type = 'cot'
    model_name = 'llama3-8B-chat'
    # model_name = 'gemma-2B-chat'
    # model_name = 'phi3-chat'
    # model_name = 'llama2-7B-chat'
    # ds = 'esnli'
    ds = 'csqa'
    
    random = False

    if expl_type == 'cot':
        metrics = ['output_expl','input_output_p','ccshap','accuracy','noise_level','paraphrase','mistake']
    else:
        metrics = ['output_expl','input_output_p', 'ccshap','cf_edit','accuracy','noise_level']
    assert set(metrics).issubset(set(['ccshap','output_expl','input_output_p','input_output_cf','paraphrase','mistake','cf_edit','ablate','accuracy','noise_level']))
    num_seeds = 3 if 'ablate' not in metrics else 1

    if random:
        add_flag = '_r'
    else:
        add_flag = ''

    prediction_dir = f'prediction/{model_name}/{ds}'
    result_path = f'results/{model_name}/{ds}/{expl_type}.txt'
    os.makedirs(os.path.dirname(result_path),exist_ok=True)

    out_msg = []

    if 'input_output_p' in metrics:
        ds_type = 'paraphrase'
    else:
        ds_type = 'cf'

    if len(set(['paraphrase','mistake']).intersection(set(metrics))) > 0:
        assert expl_type == 'cot', 'Only cot can be used for paraphrase and mistake'


    all_scores = defaultdict(list)
    for metric in metrics:
        for seed in range(num_seeds):
            prediction_dir_seed = os.path.join(prediction_dir,str(seed))
            ori_ds_path = f'data/original/{seed}/{ds}/{model_name}_{expl_type}.jsonl'
            p_ds_path = f'data/consistency/{seed}/{ds}/{model_name}_{expl_type}_paraphrase.jsonl'
            # cf_ds_path = f'data/consistency/{seed}/{ds}/{model_name}_{expl_type}_cf.jsonl'
            ds_path = {'original':ori_ds_path,'paraphrase':p_ds_path}

            if metric in ['output_expl','input_output_p','input_output_cf','ablate']:
                output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_original{add_flag}.pkl')
                expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_original{add_flag}.pkl')

                consis_output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_{ds_type}{add_flag}.pkl')
                consis_expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_{ds_type}{add_flag}.pkl')

                common_ids = find_common_sample(output_path,consis_output_path) # for paraphrase and cf, will have different size of matching ids with the original, find the matching ids 
                # for cf, we measure against the cf_edit, since we already match the ids to cf_edit in main.py

                original_samples,original_m = compute_full_divg(output_path,expl_path,common_ids)
                consis_samples,consis_m = compute_full_divg(consis_output_path,consis_expl_path,common_ids)

                if metric == 'output_expl':
                    all_scores[metric].append(original_m)

                if metric in ['input_output_p','input_output_cf']:
                    all_scores[f'{metric}_ind'].append(consis_m)

                    # Match via the sample_id
                    """
                    each o_s is a list of attributions (1 for PH, num_samples for CoT) (3 for e_s)
                    For CoT, there may be uneven samples between original and cf/paraphrase since some CoT may not be correctly formatted.
                    """
                    metric_2_samples = []
                    for sample_id,(o_s,e_s) in original_samples.items():
                        if sample_id not in consis_samples:
                            continue
                        c_o_s,c_e_s = consis_samples[sample_id]

                        if o_s.scores.shape[0] != c_o_s.scores.shape[0]: # do the averaging here to match
                            o_s,e_s,c_o_s,c_e_s = average_subj_scores(o_s),average_subj_scores(e_s),average_subj_scores(c_o_s),average_subj_scores(c_e_s)
                        o_s,e_s,c_o_s,c_e_s = o_s.scores.flatten(),e_s.scores.flatten(),c_o_s.scores.flatten(),c_e_s.scores.flatten()

                        output_divg = compute_divg(o_s,c_o_s,compare = 'o_o') 
                        expl_divg = compute_divg(e_s[:o_s.shape[0]],c_e_s[:c_o_s.shape[0]],compare = 'e_e') 
                        metric_2_samples.append(1. - np.abs(output_divg - expl_divg))
                        
                    all_scores[metric].append(np.mean(metric_2_samples))
                    
                
                if metric == 'ablate':
                    for ablate_type in ablate_types: # add subject_1 as well 
                        ablate_output_path = os.path.join(prediction_dir_seed,f'{expl_type}_output_{ablate_type}.pkl')
                        ablate_expl_path = os.path.join(prediction_dir_seed,f'{expl_type}_expl_{ablate_type}.pkl')
                        if not os.path.exists(ablate_output_path):
                            continue
                        _,ablate_m = compute_full_divg(ablate_output_path,ablate_expl_path)
                        all_scores[f"{metric}_{ablate_type}"].append(np.abs(original_m-ablate_m)/original_m)

            elif metric == 'accuracy':
                for ds_type,path_name in ds_path.items():
                    with open(path_name,'r') as f:
                        ds_ = [json.loads(l) for l in f.readlines()]
                    correct_ = np.mean([d['correct'] for d in ds_])
                    all_scores[f'{metric}_{ds_type}'].append(correct_)
            elif metric == 'noise_level':
                with open(ori_ds_path,'r') as f:
                    ori_ds = [json.loads(l) for l in f.readlines()]
                
                noise_levels = [d['difference'] for d in ori_ds if d['correct']]
                noise_levels = sorted(noise_levels,reverse=True)
                avg_noise_levels = np.mean(noise_levels[:100])
                all_scores[metric].append(avg_noise_levels)
                        
            else:
                if metric not in ['paraphrase','mistake']:
                    attack_path = os.path.join(prediction_dir_seed,f'{expl_type}_{metric}.pkl')
                else:
                    assert expl_type == 'cot'
                    attack_path = os.path.join(prediction_dir_seed,f'{metric}.pkl')

                with open(attack_path,'rb') as f:
                    perturb_scores = pickle.load(f)
                all_scores[metric].append(np.mean(list(perturb_scores.values())))
            
        if metric not in ['ablate','accuracy']:
            all_scores[metric] = (np.mean(all_scores[metric]),np.std(all_scores[metric]))

        elif metric == 'accuracy':
            for k in ['original','paraphrase']:
                all_scores[f'accuracy_{k}'] = (np.mean(all_scores[f'accuracy_{k}']),np.std(all_scores[f'accuracy_{k}']))
        else:
            for k,v in all_scores.items():
                all_scores[k] = (v[0],0.0)
        if metric in ['input_output_p','input_output_cf']:
            all_scores[f'{metric}_ind'] = (np.mean(all_scores[f'{metric}_ind']),np.std(all_scores[f'{metric}_ind']))

        # means = np.array(all_scores[metric]).mean(axis= 0)
        # stds = np.array(all_scores[metric]).std(axis = 0)
        # all_scores[metric] = (means.mean(),stds.mean())

    with open(result_path,'w') as f:
        for k,v in all_scores.items():
            msg = f'{k}: {v[0]:.3f} +/- {v[1]:.3f}\n'
            f.write(msg)
            print (msg)

if __name__ == '__main__':
    main()
    





        

    



