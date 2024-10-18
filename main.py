import os, re, json
import torch, numpy,random
from utils.causal_trace import *
from utils.extra_utils import *
from utils.prediction import *
from utils.model_data_utils import ModelAndTokenizer,get_model_path
from utils.causal_faithfulness import compute_causal_values_STR,compute_causal_values_STR_ood
import argparse
import pickle
from utils.attacks import compute_cf_edit,get_plaus_score
torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    
    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gemma2-2B-chat",
    )
    aa("--dataset_name", default="csqa")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--num_seed", default=1, type=int)
    aa("--corrupted_samples",default = 5,type = int)
    aa("--num_expl",default = 3,type = int)
    aa("--batch_size",default = 32,type = int)
    aa("--mode",default = 'STR',type = str,choices = ['STR','GN'])
    aa("--metric",default = 'causal',type = str,choices = ['causal','cc_shap','mistake','paraphrase','cf_edit','plausibility','biased','early_answering'])
    aa("--openai_api_key", default="openai_key.txt")
    aa("--ablate_noise", default="") # if ablate, set to s1 ... s5
    aa("--window", default=1,type = int)
    aa("--ood_analysis", action = 'store_true')
    args = parser.parse_args()

    if args.ood_analysis:
        args.mode = 'ood'
        print ('Running OOD analysis')

    ## Load model ##
    
    if args.metric != 'plausibility':
        model_path = get_model_path(args.model_name)
        mt = ModelAndTokenizer(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=(torch.bfloat16),
        m_name = args.model_name
        )
    
    ## Sanity Checks ##
    for seed in range(args.num_seed):
        seed_all(seed) # seed here
        ## Path
        prediction_dir = f'prediction/{args.model_name}/{args.dataset_name}/{seed}'
        os.makedirs(prediction_dir,exist_ok=True)
        pred_ds_path = f'data/{seed}/{args.dataset_name}/{args.model_name}.jsonl'
        if args.ood_analysis:
            pred_ds_path = pred_ds_path.replace('.jsonl',f'_ood_STR.jsonl') # only for STR

        ## load pred dict ##
        with open(pred_ds_path,'r') as f:
            pred_ds = [json.loads(l) for l in f.readlines()]

        if args.metric == 'causal':
            causal_path = os.path.join(prediction_dir,f'causal_{args.mode}.pkl')
            if args.window > 1:
                causal_path = causal_path.replace('.pkl',f'_{args.window}.pkl')
            ## Check for remaining samples via sample_id, only run again for samples not collected ##
            if os.path.exists(causal_path):
                with open(causal_path,'rb') as f:
                    all_causal_scores = pickle.load(f)
                original_existing_ids = set(all_causal_scores.keys())
                pred_ds = [d for d in pred_ds if d['sample_id'] not in original_existing_ids]
            else:
                all_causal_scores = {}
            
            ## Get causal score for the remaining ones
            if len(pred_ds) > 0:
                print (f'Computing remaining {len(pred_ds)} original samples')
                if args.mode == 'STR':
                    causal_scores = compute_causal_values_STR(pred_ds,mt,args)
                else:
                    causal_scores = compute_causal_values_STR_ood(pred_ds,mt,args)
                all_causal_scores.update(causal_scores)
                with open(causal_path,'wb') as f:
                    pickle.dump(all_causal_scores,f)
            else:
                print (f'Already computed {causal_path}')
            
        ## Run for other test - CC-SHAP/Mistake/Paraphrase/CF edits ##
        else:
            if args.metric == 'cc_shap':
                from utils.cshap import run_cc_shap
                run_cc_shap(mt,pred_ds,args,prediction_dir)
            elif args.metric == 'cf_edit': 
                compute_cf_edit(pred_ds,mt,prediction_dir,args)
            elif args.metric == 'plausibility':
                assert args.dataset_name != 'csqa' , 'Plausibility not supported for CSQA since no gold explanation'
                base_ds = [json.loads(l) for l in open(f"data/{args.dataset_name}_cf.jsonl",'r')]
                get_plaus_score(pred_ds,base_ds,args,prediction_dir)


if __name__ == "__main__":
    main()