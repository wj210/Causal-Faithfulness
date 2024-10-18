import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to PYTHONPATH
sys.path.append(parent_dir)
from utils.plot_utils import plot_trace_heatmap,plot_trace_barplot_joined,plot_overall_layer
from utils.model_data_utils import get_model_path
import pickle
import numpy as np
from collections import defaultdict
from score_utils.compute_scores import compute_divg
import random
from argparse import ArgumentParser

def avg_scores(dicts):
    s = []
    s.extend(d['scores'] for d in dicts)
    out = dicts[0]
    out['scores'] = np.mean(s,axis = 0)
    return out

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name',type=str,default='csqa')
    parser.add_argument('--model_name',type=str,default='llama3-8B-chat')
    parser.add_argument('--type',type=str,default='diff_prob')
    parser.add_argument('--mode',type=str,default='STR')
    args = parser.parse_args()

    seed = 0 # any random seed
    random.seed(seed)
    prediction_path = f'prediction/{args.model_name}/{args.dataset_name}/{seed}/causal_{args.mode}.pkl'
    with open(prediction_path,'rb') as f:
        causal_results = pickle.load(f)

    causal_results = [[k]+v for k,v in causal_results.items()]
    random.shuffle(causal_results)

    for plot_type in ['all']:
        plot_dir = f"plots/{args.model_name}/{args.dataset_name}/{plot_type}_{args.type.split('diff_')[-1]}_{args.mode}"
        if plot_type != 'overall_layer':
            os.makedirs(plot_dir,exist_ok=True)

        if plot_type == 'overall_layer':
            overall_tracker = defaultdict(list)
        for i,(sample_id, ans_score, expl_score) in enumerate(causal_results):
            if np.linalg.norm(ans_score[args.type]) < 1e-8 or np.linalg.norm(expl_score[args.type]) < 1e-8:
                continue
            causal_score = compute_divg(ans_score[args.type],expl_score[args.type])['single']

            ## print the results
            if plot_type == 'all':
                print (f"{args.model_name}: sample_id {sample_id}: {causal_score:.3f}")

            if plot_type == 'all':
                ans_plot_path = os.path.join(plot_dir,f'{sample_id}_ans.png')
                expl_plot_path = os.path.join(plot_dir,f'{sample_id}_expl.png')
                plot_trace_heatmap(ans_score,ans_plot_path,type_ = args.type)
                plot_trace_heatmap(expl_score,expl_plot_path,type_ = args.type)
            else:
                ans_plot_path = os.path.join(plot_dir,f'{sample_id}_both.png') 
                plot_trace_barplot_joined(output_dir,expl_dir,answer_range,ans_plot_path,type_ = plot_type)
            
            if i > 10:
                break

        if plot_type == 'overall_layer':
            ans_plot_path = os.path.join('/'.join(plot_dir.split('/')[:-1]),f'overall_layer.png') 
            plot_overall_layer(**overall_tracker,save_path=ans_plot_path)

if __name__ == '__main__':
    main()

