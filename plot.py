from utils.plot_utils import plot_trace_heatmap,plot_trace_barplot,plot_trace_barplot_joined,plot_overall_layer
from utils.extra_utils import top_n_indices,alpha_to_int,untuple_dict
from utils.model_data_utils import get_model_path
from utils.causal_trace import find_token_range
import os
import pickle
import numpy as np
import json
from transformers import AutoTokenizer
from collections import defaultdict
from compute_scores import compute_divg,average_subj_scores
import random

def avg_scores(dicts):
    s = []
    s.extend(d['scores'] for d in dicts)
    out = dicts[0]
    out['scores'] = np.mean(s,axis = 0)
    return out

def main():
    # dataset_name = 'esnli'
    dataset_name = 'csqa'
    # model_name = 'llama3-8B'
    model_name = 'llama3-8B-chat'
    # model_name = 'gemma2-27B-chat'
    # model_name = 'gemma-2B-chat'
    # model_name = 'gemma-2B'
    # expl_type='cot'
    expl_type = 'post_hoc'
    ds_type = 'original'
    # ds_type = 'paraphrase'
    seed = 0 # any random seed
    rank = 'FEAC'
    # rank = 'FEC'
    
    model_path,_ = get_model_path(model_name,expl_type='post_hoc')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prediction_dir = f'prediction/{model_name}/{dataset_name}/{seed}'
    
    common_keys = [set() for _ in range(2)]
    # for i,m in enumerate(['llama3-8B-chat','gemma-2B-chat','gemma2-27B-chat']):
    for i,m in enumerate(['llama3-8B-chat','llama3-8B']):
        o_path = f'prediction/{m}/{dataset_name}/{seed}/{expl_type}_output_original.pkl'
        with open(o_path,'rb') as f:
            o = pickle.load(f)
        common_keys[i] = set(o.keys())
    common_keys = set.intersection(*common_keys)
    
    random.seed(1)
    random_keys = random.sample(list(common_keys),min(10,len(list(common_keys)))) 
        
    
    output_path = os.path.join(prediction_dir,f'{expl_type}_output_original.pkl')
    expl_path = os.path.join(prediction_dir,f'{expl_type}_expl_original.pkl')
    
    edit_output_path = os.path.join(prediction_dir,f'{expl_type}_output_paraphrase.pkl')
    edit_expl_path = os.path.join(prediction_dir,f'{expl_type}_expl_paraphrase.pkl')
    
    
    if ds_type == 'original':
        data_path = f"data/original/{seed}/{dataset_name}/{model_name}_{expl_type}.jsonl"
    else:
        data_path = f"data/consistency/{seed}/{dataset_name}/{model_name}_{expl_type}_{ds_type}.jsonl"
           
    
    with open(data_path,'r') as f:
        known_ds = [json.loads(l) for l in f]
    known_ds = {d['sample_id']:d for d in known_ds}
    
    with open(output_path,'rb') as f:
        output_results = pickle.load(f)
    with open(expl_path,'rb') as f:
        expl_results = pickle.load(f)
    with open(edit_output_path,'rb') as f:
        edit_output_results = pickle.load(f)
    with open(edit_expl_path,'rb') as f:
        edit_expl_results = pickle.load(f)
 
    
    ## rank by highest or lowest diverg
    score_d = {}
    for sample_id,output_d in output_results.items():
        if rank == 'FEAC':
            score_d[sample_id] = compute_divg(output_d['scores'],expl_results[sample_id]['scores'])
        else:
            o_s = output_d
            e_s = expl_results[sample_id]
            e_o_s = edit_output_results[sample_id]
            e_e_s = edit_expl_results[sample_id]
            if o_s['scores'].shape[0] != e_o_s['scores'].shape[0] or e_s['scores'].shape[0] != e_e_s['scores'].shape[0]: # do the averaging here to match
                try:
                    o_s,e_o_s = average_subj_scores(o_s,e_o_s)
                    e_s,e_e_s = average_subj_scores(e_s,e_e_s)
                except Exception as e:
                    continue
            else:
                o_s,e_s,e_o_s,e_e_s = output_d['scores'],e_s['scores'],e_o_s['scores'],e_e_s['scores']
            output_divg = compute_divg(o_s,e_o_s,compare = 'o_o') 
            expl_divg = compute_divg(e_s,e_e_s,compare = 'e_e') 
            # output_divg = o_s - e_o_s
            # expl_divg = e_s - e_e_s
            score_d[sample_id] = expl_divg

    # plot individually and compare across token/layer (all)
    output_results = {k:v for k,v in output_results.items() if k in random_keys and k in score_d}
    
    for sort_key in ['high']:
        
        # sorted_ids = [k for k,v in sorted(score_d.items(),key = lambda x:x[1],reverse = sort_key == 'high')]
        # output_results = {k: output_results[k] for k in sorted_ids}
        
        for plot_type in ['all']:
            count = 0
            plot_dir = f"plots/{model_name}/{dataset_name}/{expl_type}_original/{plot_type}/{sort_key}"
            edit_plot_dir = f"plots/{model_name}/{dataset_name}/{expl_type}_paraphrase/{plot_type}/{sort_key}"
            if plot_type != 'overall_layer':
                os.makedirs(plot_dir,exist_ok=True)

            if plot_type == 'overall_layer':
                overall_tracker = defaultdict(list)
            for i,(sample_id,output_dir) in enumerate(output_results.items()):
                
                expl_dir = expl_results[sample_id]
                e_output_dir = edit_output_results[sample_id]
                e_expl_dir = edit_expl_results[sample_id]
                # find the position of the answer strings
                choices = (known_ds[sample_id]['choices'])
                answer_index = alpha_to_int(known_ds[sample_id]['answer'])
                answer_string = choices[answer_index]
                formatted_answer_string = f"({known_ds[sample_id]['answer']}) {answer_string}"
                answer_range = find_token_range(tokenizer,output_dir['input_tokens'],formatted_answer_string)
                
                ## print the results
                if plot_type == 'all':
                    print (f"sample_id {sample_id}, {rank}: {score_d[sample_id]:.3f}")

                if plot_type == 'all':
                    output_plot_path = os.path.join(plot_dir,f'{sample_id}_output.png')
                    expl_plot_path = os.path.join(plot_dir,f'{sample_id}_expl.png')
                    plot_trace_heatmap(output_dir,output_plot_path,modelname=model_name)
                    plot_trace_heatmap(expl_dir,expl_plot_path,modelname=model_name)
                    if rank == 'FEC':
                        e_output_plot_path = os.path.join(edit_plot_dir,f'{sample_id}_output.png')
                        e_expl_plot_path = os.path.join(edit_plot_dir,f'{sample_id}_expl.png')
                        plot_trace_heatmap(e_output_dir,e_output_plot_path,modelname=model_name)
                        plot_trace_heatmap(e_expl_dir,e_expl_plot_path,modelname=model_name)

                else:
                    output_plot_path = os.path.join(plot_dir,f'{sample_id}_both.png') 
                    plot_trace_barplot_joined(output_dir,expl_dir,answer_range,output_plot_path,type_ = plot_type)
                count += 1
            if plot_type == 'overall_layer':
                output_plot_path = os.path.join('/'.join(plot_dir.split('/')[:-1]),f'overall_layer.png') 
                plot_overall_layer(**overall_tracker,save_path=output_plot_path)

            


if __name__ == '__main__':
    main()

