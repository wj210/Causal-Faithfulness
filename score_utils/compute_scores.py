import numpy as np
import json,pickle,os,sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to PYTHONPATH
sys.path.append(parent_dir)
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from scipy import spatial, stats
from utils.causal_trace import find_token_range
from transformers import AutoTokenizer
from utils.model_data_utils import get_model_path
from utils.extra_utils import alpha_to_int,join_choices,format_input
from utils.prediction import format_mcq
from utils.plot_utils import plot_category_barplot

"""
Get scores for causal, CFF and CC-SHAP
"""

def compute_divg(o,e,compute_type = 'single'): # single compute over token and layer wise, else all means compute over 3 kinds: token, layer and token-layer
    if compute_type == 'all':
        to_compute = ['single','token','layer']
    else:
        to_compute = [compute_type]
    out = {}
    for com in to_compute:
        if com == 'single':
            if len(o.shape) > 1:
                temp_o = o.flatten()
                temp_e = e.flatten()
        elif com == 'token':
            temp_o = o.sum(axis = 1)
            temp_e = e.sum(axis = 1)
        else:
            temp_o = o.sum(axis = 0)
            temp_e = e.sum(axis = 0)
        out[com] = 1. - spatial.distance.cosine(temp_o, temp_e)
    return out


def renormalize_shap_values(shap_values,tokenizer,data_dict,type_ = 'original'):
    choices = join_choices(data_dict['choices'])
    if type_ == 'original':
        question = data_dict['question']
        subject = data_dict['subject']
    else:
        question = data_dict['cf_question']
        subject = data_dict['cf_subject']
    input_tokens = format_mcq(question,choices,expl_=False,is_chat=True) # only do for chat model so far.
    input_tokens = format_input(input_tokens,tokenizer)
    truncated_input = input_tokens.split("\n\nAnswer:")[0]
    truncated_len = len(tokenizer([truncated_input], return_tensors="pt", padding=False, add_special_tokens=False).input_ids[0])
    assert truncated_len == len(shap_values), f'{truncated_len} != {len(shap_values)}, should be able to recreate here.'

    encoded_input= tokenizer.encode(truncated_input,add_special_tokens=False)
    ## Split the rest of comve such that we only left with the original question.
    token_start,token_end = find_token_range(tokenizer,encoded_input,question,include_chat_template = True)
    subject_range =  find_token_range(tokenizer,encoded_input,subject,include_chat_template = True)
    if token_start is None or subject_range[0] is None:
        return None,None
    # if subject_range[1] - subject_range[0] > (token_end - token_start)/2:
    #     return None,None # we skip instances where the subject is more than half of the input. (makes it easier for the subject tokens to be important)
    to_deduct = np.abs(shap_values[:token_start]).sum() + np.abs(shap_values[token_end:]).sum()
    if (np.abs(shap_values).sum() - to_deduct) < 1e-8:
        return 0,0
    shap_values = shap_values[token_start:token_end]/ (np.abs(shap_values).sum() - to_deduct)
    return shap_values,list(range(*subject_range))


# Function to compute average percentile rank of a subset within its vector
def compute_average_percentile(vector, subset_indices,tokenizer=None,data_dict=None,renormalize =False,type_='original'):
    if renormalize:
        vector,subset_indices = renormalize_shap_values(vector,tokenizer,data_dict,type_ = type_)
        if vector is None:
            return None,None
        elif isinstance(subset_indices,int) and subset_indices == 0:
            return 0,False
    else:
        if subset_indices[0] is None:
            return None,None
        subset_indices = range(subset_indices[0],subset_indices[1])
    total_value = vector[subset_indices].mean()
    percentile = stats.percentileofscore(vector, total_value, kind='rank')
    non_subset_indices = [i for i in range(len(vector)) if i not in subset_indices]
    non_subset_values = vector[non_subset_indices].mean()
    is_important = total_value > non_subset_values
    return percentile,is_important

def compute_high_percentile_and_importance(percentiles,importances):
    high_cf = []
    high_cf_and_o = []
    robust_cf = []
    robust_cf_and_o = []
    for cf,o,cf_i,o_i in zip(percentiles['cf'],percentiles['original'],importances['cf'],importances['original']):
        if cf > 75:
            high_cf.append(1)
            if o > 75:
                high_cf_and_o.append(1)
        else:
            high_cf.append(0)
            if o > 75:
                high_cf_and_o.append(0)
        
        if cf_i:
            robust_cf.append(1)
            if o_i:
                robust_cf_and_o.append(1)
        else:
            robust_cf.append(0)
            if o_i:
                robust_cf_and_o.append(0)
    return np.mean(high_cf),np.mean(high_cf_and_o),np.mean(robust_cf),np.mean(robust_cf_and_o),len(robust_cf_and_o)/len(percentiles['cf']),len(high_cf_and_o)/len(percentiles['cf'])


def compute_relative_importance(score_dict,data_dict,tokenizer,metric = 'diff_prob'):
    """
    Given score_dict, retrieve the metric key and both the answer and corrupted token positions from the full input token.
    input token: input_tokens[0]
    corrupted token range: subject_range[0] or [1] , both is the same
    answer token range requires the 'choice' and 'pred' values from data_dict to retrieve the full choice string and then use find_token_range to get the positions
    Then compute for both, the relative importance over the summed importance over the full input.
    """
    scores = np.maximum(score_dict[metric],0).sum(axis=1) # sum over all layers, considering the positive values only
    
    corrupt_pos = range(0,score_dict['subject_range'][0][1]-score_dict['subject_range'][0][0])
    pred = data_dict['pred']
    choice = data_dict['choices'][alpha_to_int(pred)]
    answer_str = f"({pred}) {choice}"
    answer_pos = find_token_range(tokenizer,score_dict['input_tokens'][0],answer_str)
    all_answer_pos = find_token_range(tokenizer,score_dict['input_tokens'][0],'Choices:')
    all_answer_pos = range(all_answer_pos[0],scores.shape[0])
    answer_pos = range(answer_pos[0],answer_pos[1])

    total_score = scores.sum()
    corrupt_ratio = scores[corrupt_pos].sum()/total_score
    answer_ratio = scores[answer_pos].sum()/total_score
    all_answer_ratio = scores[all_answer_pos].sum()/total_score

    return corrupt_ratio,answer_ratio,all_answer_ratio


def load_base_scores(pred_ds):
    out = {}
    out['ACC'] = sum([d['correct'] for d in pred_ds])/len(pred_ds)
    out['Corrupt_ACC'] = 1 - (sum([d['incorrect'] for d in pred_ds])/len(pred_ds))
    out['ans_logit_diff'] = np.mean([d['high_logit'] - d['low_logit'] for d in pred_ds])
    out['ans_prob_diff'] = np.mean([d['high_prob'] - d['low_prob'] for d in pred_ds]) / np.mean([d['high_prob'] for d in pred_ds])
    out['expl_logit_diff'] = np.mean([np.mean(np.array(d['high_expl_logit']) - np.array(d['low_expl_logit'])) for d in pred_ds])
    out['expl_prob_diff'] = np.mean([np.mean(np.array(d['high_expl_prob'])- np.array(d['low_expl_prob'])) for d in pred_ds])/ np.mean([np.mean(d['high_expl_prob']) for d in pred_ds])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,choices = ['gemma2-2B-chat','gemma2-2B','gemma2-9B-chat','gemma2-9B','gemma2-27B-chat','gemma2-27B'],default = 'gemma2-2B-chat')
    parser.add_argument('--metric',type=str,choices = ['causal','cf_edit','ccshap','ood','ood_GN'],default = ['causal'],nargs='+')
    parser.add_argument('--causal_type',type=str,default = ['STR'],nargs='+')
    parser.add_argument('--dataset_name',type=str,choices = ['csqa','esnli','comve'],default= ['csqa'],nargs='+')
    parser.add_argument('--num_seeds',type=int,default = 1)
    args = parser.parse_args()

    if 'ood' not in args.metric or 'ood_GN' not in args.metric:
        result_file = f'results/{args.model_name}.txt'
        args.metric.append('base')
    else:
        result_file = f'results/{args.model_name}_ood.txt'
    os.makedirs(os.path.dirname(result_file),exist_ok=True)

    with open(result_file,'w') as f:
        f.write(f'Model: {args.model_name}\n\n')

    tokenizer = AutoTokenizer.from_pretrained(get_model_path(args.model_name))

    for dataset in args.dataset_name:
        zero_causal_counts = defaultdict(int)
        invalid_shap_counts = 0
        invalid_cf_edit_counts = 0
        with open(result_file,'a') as f:
            f.write(f'Dataset: {dataset}\n' + '--'*50+'\n')
        for metric in args.metric:
            seeds = list(range(args.num_seeds))
            seed_scores = []
            for seed in seeds:
                ds_path = f'data/{seed}/{dataset}/{args.model_name}.jsonl' if metric != 'ood' else f'data/{seed}/{dataset}/{args.model_name}_ood_STR.jsonl'
                with open(ds_path,'r') as f:
                    pred_ds = [json.loads(l) for l in f]

                pred_ds_dict = {d['sample_id']:d for d in pred_ds}
                score_path = f'prediction/{args.model_name}/{dataset}/{seed}/{metric}.pkl'
                if metric == 'causal':
                    score_dict = defaultdict(list)
                    for causal_type in args.causal_type:
                        causal_score_path = score_path.replace('.pkl',f'_{causal_type}.pkl')
                        if not os.path.exists(causal_score_path):
                            print (f'{causal_score_path} does not exist')
                            continue
                        with open(causal_score_path,'rb') as f:
                            scores = pickle.load(f)
                        for sample_id,(ans_score,expl_score) in scores.items():
                            for k in ['diff_prob']:
                                if np.linalg.norm(ans_score[k]) < 1e-8 or np.linalg.norm(expl_score[k]) < 1e-8:
                                    # print (f'Zero vector for {k} for sample_id {sample_id}')
                                    zero_causal_counts[k] += 1
                                    continue
                                causal_computed_scores = compute_divg(ans_score[k],expl_score[k],compute_type = 'all')
                                for compute_type,c_score in causal_computed_scores.items():
                                    score_dict[f"{causal_type}_{k}_{compute_type}"].append(c_score)
                            
                            ## We look at the relative importance of the answer tokens and the corrupted tokens (average across all layers)
                            ans_importance = compute_relative_importance(ans_score,pred_ds_dict[sample_id],tokenizer) # contains the corup ratio/answer ratio
                            expl_importance = compute_relative_importance(expl_score,pred_ds_dict[sample_id],tokenizer) 
                            if ans_importance is None or expl_importance is None:
                                continue
                            score_dict[f"{causal_type}_ans_corrupt_importance"].append(ans_importance[0])
                            score_dict[f"{causal_type}_ans_ans_importance"].append(ans_importance[1])
                            score_dict[f"{causal_type}_ans_all_ans_importance"].append(ans_importance[2])
                            score_dict[f"{causal_type}_expl_corrupt_importance"].append(expl_importance[0])
                            score_dict[f"{causal_type}_expl_ans_importance"].append(expl_importance[1])
                            score_dict[f"{causal_type}_expl_all_ans_importance"].append(expl_importance[2])


                    for k in score_dict.keys():
                        score_dict[k] = np.mean(score_dict[k])
                    seed_scores.append(score_dict)
                
                elif metric in ['cf_edit','ccshap']:
                    with open(score_path,'rb') as f:
                        scores = pickle.load(f)
                    score_dict = []
                    for sample_id,(score,valid) in scores.items():
                        if not valid:
                            if metric == 'ccshap': # distance = 0
                                invalid_shap_counts += 1
                            else:
                                invalid_cf_edit_counts += 1 # did not managed to find any edits that changed the answer.
                        score_dict.append(score)
                    seed_scores.append(np.mean(score_dict))
                
                elif metric == 'base':
                    other_metrics = load_base_scores(pred_ds)
                    seed_scores.append(other_metrics)

                ### OOD Analysis ###
                elif metric == 'ood':
                    ccshap_path = f'prediction/{args.model_name}/{dataset}/{seed}/ccshap_ood.pkl'

                    with open(ccshap_path,'rb') as f:
                        ccshap_ood_scores = pickle.load(f)
                    
                    shap_percentiles = defaultdict(list)
                    shap_importance = defaultdict(list)
                    ood_shap_error = 0
                    for sample_id,ccshap_samples in ccshap_ood_scores.items():
                        data_dict = pred_ds_dict[sample_id]
                        ori_ccshap = ccshap_samples['original']
                        cf_ccshap = ccshap_samples['cf']
                        if not np.isnan(ori_ccshap[0]).any() and not np.isnan(cf_ccshap[0]).any():
                            ori_subj_percentile = compute_average_percentile(ori_ccshap[0],ori_ccshap[1],tokenizer,data_dict,renormalize=False,type_='original')
                            cf_subj_percentile = compute_average_percentile(cf_ccshap[0],cf_ccshap[1],tokenizer,data_dict,renormalize=False,type_='cf')
                            if ori_subj_percentile[0] is None or cf_subj_percentile[0] is None:
                                ood_shap_error += 1
                                continue
                            shap_percentiles['original'].append(ori_subj_percentile[0])
                            shap_percentiles['cf'].append(cf_subj_percentile[0])
                            shap_importance['original'].append(ori_subj_percentile[1])
                            shap_importance['cf'].append(cf_subj_percentile[1])
                        else: # means all the shap values are zero (the marginalized out have most of the values)
                            shap_percentiles['original'].append(0)
                            shap_percentiles['cf'].append(0)
                            shap_importance['original'].append(False)
                            shap_importance['cf'].append(False)
                    
                    overall_mean = {}
                    overall_mean['shap_cf_percent'] = np.mean(shap_percentiles['cf'])
                    overall_mean['shap_ori_percent'] = np.mean(shap_percentiles['original'])
                    shap_updates = compute_high_percentile_and_importance(shap_percentiles,shap_importance)
                    overall_mean['shap_high_cf_percent'] = shap_updates[0]
                    overall_mean['shap_high_cf_o_percent'] = shap_updates[1]            
                    overall_mean['shap_high_cf_imp'] = shap_updates[2]
                    overall_mean['shap_high_cf_o_imp'] = shap_updates[3]  
                    overall_mean['shap_percent_counts'] = shap_updates[4] # relative len of shap_high_cf_o_percent
                    overall_mean['shap_imp_counts'] = shap_updates[-1] 
                    print (f'{ood_shap_error} error out of {len(ccshap_ood_scores)} samples')

                    ## Robustness for causal ###             
                    causal_path = f'prediction/{args.model_name}/{dataset}/{seed}/causal_ood.pkl'

                    with open(causal_path,'rb') as f:
                        causal_ood_scores = pickle.load(f)
                    
                    causal_percentile = defaultdict(list)
                    causal_importance = defaultdict(list)
                    for sample_id,(ori_causal,cf_causal) in causal_ood_scores.items():
                        ori_causal_scores = ori_causal['diff_prob'].sum(axis=1)
                        cf_causal_scores = cf_causal['diff_prob'].sum(axis=1)
                        ori_causal_range = [0,ori_causal['subject_range'][0][1] - ori_causal['subject_range'][0][0]]
                        cf_causal_range = [0,cf_causal['subject_range'][0][1] - cf_causal['subject_range'][0][0]]
                        ori_p,ori_i = compute_average_percentile(ori_causal_scores,ori_causal_range)
                        cf_p,cf_i = compute_average_percentile(cf_causal_scores,cf_causal_range)
                        causal_percentile['original'].append(ori_p)
                        causal_percentile['cf'].append(cf_p)
                        causal_importance['original'].append(ori_i)
                        causal_importance['cf'].append(cf_i)
                    
                    overall_mean['causal_cf_percent'] = np.mean(causal_percentile['cf'])
                    overall_mean['causal_ori_percent'] = np.mean(causal_percentile['original'])
                    causal_updates = compute_high_percentile_and_importance(causal_percentile,causal_importance)
                    overall_mean['causal_high_cf_percent'] = causal_updates[0]
                    overall_mean['causal_high_cf_o_percent'] = causal_updates[1]            
                    overall_mean['causal_high_cf_imp'] = causal_updates[2]
                    overall_mean['causal_high_cf_o_imp'] = causal_updates[3]  
                    overall_mean['causal_percent_counts'] = causal_updates[4] # relative len of shap_high_cf_o_percent
                    overall_mean['causal_imp_counts'] = causal_updates[-1]

                    ## Plot out the results ##
                    plot_path = f'plots/ood/{args.model_name}_SHAP.png'
                    cats = ['CF','Both']
                    values_dict = {}
                    values_dict['SHAP'] = [overall_mean['shap_high_cf_imp']*100,overall_mean['shap_high_cf_o_imp']*100]
                    values_dict['AP (STR)'] = [overall_mean['causal_high_cf_percent']*100,overall_mean['causal_high_cf_o_percent']*100]
                    plot_category_barplot(cats,values_dict,plot_path,colors = ['purple','blue'])

                elif metric == 'ood_GN': 
                    """
                    1) difference between the original prob and corrupted prob (of the counterfactual answer)
                    2) Measure the total softmax probabilities of the two possible answers (original and counterfactual) in the CF
                    """
                    ood_gn_path = f'data/{seed}/comve/{args.model_name}_ood_GN.jsonl'
                    with open(ood_gn_path,'r') as f:
                        ood_gn_scores = [json.loads(l) for l in f]
                    
                    original_high_prob = np.mean([d['high_prob'] for d in ood_gn_scores])
                    cf_low_prob = np.mean([d['low_cf_prob'] for d in ood_gn_scores])

                    STR_cf_prob = np.mean([d['high_cf_prob'] for d in ood_gn_scores])
                    STR_original_low_prob = np.mean([d['low_prob'] for d in ood_gn_scores])

                    GN_cf_prob = np.mean([d['high_gn_cf_prob'] for d in ood_gn_scores])
                    GN_original_low_prob = np.mean([d['low_gn_probs'] for d in ood_gn_scores])

                    ## plot it out
                    plot_path = f'plots/ood/{args.model_name}_GN.png'
                    os.makedirs(os.path.dirname(plot_path),exist_ok=True)
                    cats = ['Original','CF']
                    values_dict = {}
                    values_dict['Clean'] = [original_high_prob,cf_low_prob]
                    values_dict['STR'] = [STR_original_low_prob,STR_cf_prob]
                    values_dict['GN'] = [GN_original_low_prob,GN_cf_prob]
                    plot_category_barplot(cats,values_dict,plot_path,colors = ['green','blue','red'])
                    exit(f'Done plotting GN ood for {args.model_name}')

            ## Average over seeds ##
            if metric in ['causal','base']:
                overall_mean = {k: np.mean([s[k] for s in seed_scores]) for k in seed_scores[0].keys()}
                # overall_std = {k: np.std([s[k] for s in seed_scores]) for k in seed_scores[0].keys()}
            elif metric in ['cf_edit','ccshap']:
                overall_mean = {f'{metric.upper()}':np.mean(seed_scores)}
                # overall_std = {f'{metric.upper()}':np.std(seed_scores)}

            with open(result_file,'a') as f:
                for score_name,mean_score in overall_mean.items():
                    f.write(f'{score_name} : {mean_score:.3f} \n')
                    print(f'Dataset: {dataset} Metric: {score_name} : {mean_score:.3f}\n')
    
        with open(result_file,'a') as f:
            for k,v in zero_causal_counts.items():
                f.write(f'Zero Causal counts for {k}:{v//args.num_seeds}\n')
            f.write(f'Invalid SHAP counts: {invalid_shap_counts//args.num_seeds}\n')
            f.write(f'Invalid CF EDIT counts: {invalid_cf_edit_counts//args.num_seeds}\n')



if __name__ == '__main__':
    main()
