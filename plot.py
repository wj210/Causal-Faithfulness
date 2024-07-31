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

def avg_scores(dicts):
    s = []
    s.extend(d['scores'] for d in dicts)
    out = dicts[0]
    out['scores'] = np.mean(s,axis = 0)
    return out
    

def main():
    # dataset_name = 'esnli'
    dataset_name = 'csqa'
    model_name = 'gemma-2B-chat'
    # model_name = 'gpt2'
    # expl_type='cot'
    expl_type = 'post_hoc'
    ds_type = 'original'
    # ablate ='subject_1'
    ablate = ''
    seed = 0 # any random seed
    seed = 0 if ablate != '' else seed
    prediction_dir = f'prediction/{model_name}/{dataset_name}/{seed}'
    output_path = os.path.join(prediction_dir,f'{expl_type}_output_{ds_type}.pkl')
    expl_path = os.path.join(prediction_dir,f'{expl_type}_expl_{ds_type}.pkl')
    data_path = f"data/{ds_type}/{seed}/{dataset_name}/{model_name}_{expl_type}.jsonl"
    model_path,_ = get_model_path(model_name,expl_type='post_hoc')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    subject_key = 'subject' if ds_type == 'original' else f"{ds_type}_subject"

    if ablate != "": # just to change naming
        output_path = output_path.replace(ds_type,ablate)
        expl_path = expl_path.replace(ds_type,ablate)
        ds_type = ablate


    choice_key = [] 

    with open(output_path,'rb') as f:
        output_results = pickle.load(f)
    with open(expl_path,'rb') as f:
        expl_results = pickle.load(f)
    with open(data_path,'r') as f:
        known_ds = [json.loads(l) for l in f]
    known_ds = {d['sample_id']:d for d in known_ds}

    # plot individually and compare across token/layer (all)
    for plot_type in ['all','token','layer','overall_layer']:
        count = 0
        plot_dir = f"plots/{model_name}/{dataset_name}/{expl_type}_{ds_type}/{plot_type}"
        if plot_type != 'overall_layer':
            os.makedirs(plot_dir,exist_ok=True)

        if plot_type == 'overall_layer':
            overall_tracker = defaultdict(list)
        for i,(sample_id,output_dir) in enumerate(output_results.items()):
            expl_dir = expl_results[sample_id]
            expl_dir['kind'] = None
            output_dir['kind'] = None
            # find the position of the answer strings
            choices = untuple_dict((known_ds[sample_id]['choices']),choice_key)
            answer_index = alpha_to_int(known_ds[sample_id]['answer'])
            answer_string = choices[answer_index]
            formatted_answer_string = f"({known_ds[sample_id]['answer']}) {answer_string}"
            answer_range = find_token_range(tokenizer,output_dir['input_tokens'],formatted_answer_string)

            # subj = known_ds[sample_id][subject_key]
            # subj_range = find_token_range(tokenizer,output_dir['input_tokens'],subj)
            # expl_dir['subject_range'] = subj_range
            # output_dir['subject_range'] = subj_range

            if plot_type == 'overall_layer':
                overall_tracker['o_list'].append(output_dir)
                overall_tracker['expl_list'].append(expl_dir)
                overall_tracker['answer_ranges'].append(answer_range)
                continue

            if plot_type == 'all':
                output_plot_path = os.path.join(plot_dir,f'{sample_id}_output.png')
                expl_plot_path = os.path.join(plot_dir,f'{sample_id}_expl.png')
                plot_trace_heatmap(output_dir,output_plot_path,modelname=model_name)
                plot_trace_heatmap(expl_dir,expl_plot_path,modelname=model_name)
            else:
                output_plot_path = os.path.join(plot_dir,f'{sample_id}_both.png') 
                plot_trace_barplot_joined(output_dir,expl_dir,answer_range,output_plot_path,type_ = plot_type)
            count += 1
            if count >=10:
                break
        if plot_type == 'overall_layer':
            output_plot_path = os.path.join('/'.join(plot_dir.split('/')[:-1]),f'overall_layer.png') 
            plot_overall_layer(**overall_tracker,save_path=output_plot_path)


if __name__ == '__main__':
    main()

