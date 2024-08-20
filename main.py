import os, re, json
import torch, numpy,random
from utils.causal_trace import *
from utils.extra_utils import *
from utils.prediction import *
from utils.model_data_utils import ModelAndTokenizer,get_model_path
import argparse
import pickle

from utils.attacks import run_semantic_attacks,paraphrase_instruction,compute_causal_values,compute_cf_edit,get_plaus_score,eval_biasing_features
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
        default="llama3-8B-chat",
    )
    aa("--dataset_name", default="csqa")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--num_seed", default=3, type=int)
    aa("--num_samples",default = 100,type = int)
    aa("--corrupted_samples",default = 5,type = int)
    aa("--num_expl",default = 3,type = int)
    aa("--batch_size",default = 32,type = int)
    aa("--expl_type",default = 'cot',type = str,choices = ['post_hoc','cot']) 
    aa("--metric",default = 'causal',type = str,choices = ['fec','feac','cc_shap','mistake','paraphrase','cf_edit','plausibility','biased','early_answering'])
    aa("--openai_api_key", default="openai_key.txt")
    aa("--ablate_noise", default="") # if ablate, set to s1 ... s5
    args = parser.parse_args()

    if args.metric in ['paraphrase','mistake','plausibility']:
        os.environ['OPENAI_API_KEY'] = open(args.openai_api_key,'r').read().strip()

    """
    In the case of different seed runs, we may have different difference scores, we find the common samples with the highest average difference score.
    """
    known_paths,known_edit_paths = [],[]
    edit_type = 'paraphrase' if args.metric != 'cf_edit' else 'cf'

    for seed in range(args.num_seed):
        ## original
        known_ds_dir = f'data/original/{seed}/{args.dataset_name}'
        known_paths.append(f"{known_ds_dir}/{args.model_name}_{args.expl_type}.jsonl")
        known_edit_paths.append(f"data/consistency/{seed}/{args.dataset_name}/{args.model_name}_{args.expl_type}_paraphrase.jsonl")

    ori_preds,known_subjects = get_common_samples(known_paths,type_ = 'original')
    

    ## ensure for each sample id, the ori and edit predictions are the same ##
    if args.metric == 'FEC':
        edit_preds,edit_subjects = get_common_samples(known_edit_paths,type_ = "paraphrase")
        applicable_ids = set()
        for ok,op in ori_preds.items():
            if ok not in edit_preds:
                continue
            ep = edit_preds[ok]
            add = True
            for ori_p,edit_p in zip(op,ep):
                if ori_p != edit_p:
                    add = False
                    break
            if add:
                applicable_ids.add(ok)
    else:
        applicable_ids = set(ori_preds.keys())

    
    ## randomly pick samples
    random.seed(42)
    selected_ids = random.sample(applicable_ids,args.num_samples)
    choice_key = []

    ## Load model ##
    model_path,use_fs = get_model_path(args.model_name,args.expl_type)
    mt = ModelAndTokenizer(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.bfloat16),
    m_name = args.model_name
    )
    
    ## Sanity Checks ##
    if args.metric in ['paraphrase','mistake','biased','early_answering']:
        assert args.expl_type == 'cot', 'Only cot is supported for semantic attacks'
    elif args.metric == 'cf_edit':
        assert args.expl_type == 'post_hoc', 'only compute cf edit for post_hoc type of explanations'
    
    if args.ablate_noise != "":
        assert args.ablate_noise != 's3', 'choose a noise level other than s3'
        path_flag = f"_{args.ablate_noise}"
        args.noise_level = args.ablate_noise # change the noise level to the ablated noise level
    else:
        path_flag = ''

    for seed in range(args.num_seed):
        seed_all(seed) # seed here
        ## Path
        prediction_dir = f'prediction/{args.model_name}/{args.dataset_name}/{seed}'
        os.makedirs(prediction_dir,exist_ok=True)

        known_ds_dir = f'data/original/{seed}/{args.dataset_name}'
        known_ds_path = f"{known_ds_dir}/{args.model_name}_{args.expl_type}.jsonl"
        orig_output_causal_path = os.path.join(prediction_dir,f'{args.expl_type}_output_original{path_flag}.pkl') #**
        orig_expl_causal_path = os.path.join(prediction_dir,f'{args.expl_type}_expl_original{path_flag}.pkl') #**
        ##
        
        ## Known/edit path for each seed ##
        with open(known_ds_path,'r') as f:
            known_ds = [json.loads(l) for l in f.readlines()]
        original_noise_level = float(args.noise_level[1:]) * collect_embedding_std(mt, known_subjects)

        if args.metric in ['fec','feac']:
            ## Check for remaining samples via sample_id, only run again for samples not collected ##
            if os.path.exists(orig_output_causal_path):
                with open(orig_output_causal_path,'rb') as f:
                    orig_output_store = pickle.load(f)
                with open(orig_expl_causal_path,'rb') as f:
                    orig_expl_store = pickle.load(f)
                original_existing_ids = set(orig_output_store.keys())
                remaining_original_ids = set(selected_ids) - original_existing_ids

            else:
                remaining_original_ids = selected_ids
                orig_output_store = {}
                orig_expl_store = {}
            
            ## Get causal score for the remaining ones
            if len(remaining_original_ids) > 0:
                print (f'Computing remaining {len(remaining_original_ids)} original samples')
                known_ds_dict = {d['sample_id']:d for d in known_ds}
                remaining_original_ds = [known_ds_dict[i] for i in remaining_original_ids]

                o_store,e_store = compute_causal_values(remaining_original_ds,mt,choice_key,use_fs,'original',original_noise_level,args)
                orig_output_store.update(o_store)
                orig_expl_store.update(e_store)

                with open(orig_output_causal_path,'wb') as f:
                    pickle.dump(orig_output_store,f)
                with open(orig_expl_causal_path,'wb') as f:
                    pickle.dump(orig_expl_store,f)
            

            if args.metric == 'fec':
                edited_known_ds_path = f"data/consistency/{seed}/{args.dataset_name}/{args.model_name}_{args.expl_type}_{edit_type}.jsonl"
                edit_output_causal_path = os.path.join(prediction_dir,f'{args.expl_type}_output_{edit_type}.pkl') #**
                edit_expl_causal_path = os.path.join(prediction_dir,f'{args.expl_type}_expl_{edit_type}.pkl') #**
                with open(edited_known_ds_path,'r') as f:
                    edited_known_ds = [json.loads(l) for l in f.readlines()]
                edited_noise_level = float(args.noise_level[1:]) * collect_embedding_std(mt, edit_subjects)
                if os.path.exists(edit_output_causal_path):
                    with open(edit_output_causal_path,'rb') as f:
                        edit_output_store = pickle.load(f)
                    with open(edit_expl_causal_path,'rb') as f:
                        edit_expl_store = pickle.load(f)
                    edit_existing_ids = set(edit_output_store.keys())
                    remaining_edit_ids = set(selected_ids) - edit_existing_ids
                else:
                    remaining_edit_ids = selected_ids
                    edit_output_store = {}
                    edit_expl_store = {}

                if len(remaining_edit_ids) > 0 and args.ablate_noise == "":
                    print (f'Computing remaining {len(remaining_edit_ids)} {edit_type} samples')
                    edited_known_ds_dict = {d['sample_id']:d for d in edited_known_ds}
                    remaining_edit_ds = [edited_known_ds_dict[i] for i in remaining_edit_ids]
                    o_store,e_store = compute_causal_values(remaining_edit_ds,mt,choice_key,use_fs,edit_type,edited_noise_level,args)
                    edit_output_store.update(o_store)
                    edit_expl_store.update(e_store)

                    with open(edit_output_causal_path,'wb') as f:
                        pickle.dump(edit_output_store,f)
                    with open(edit_expl_causal_path,'wb') as f:
                        pickle.dump(edit_expl_store,f)
        
        ## Run for other test - CC-SHAP/Mistake/Paraphrase/CF edits ##
        else:
            known_ds_dict=  {d['sample_id']:d for d in known_ds}
            selected_ds = [known_ds_dict[i] for i in selected_ids]
            if args.metric in ['mistake','paraphrase','early_answering']:
                run_semantic_attacks(selected_ds,mt,choice_key,args,attack = args.metric,save_dir = prediction_dir,seed = seed,use_fs =use_fs)
            elif args.metric == 'biased':
                eval_biasing_features(selected_ds,mt,prediction_dir)
            elif args.metric == 'cc_shap':
                from utils.cshap import run_cc_shap
                run_cc_shap(mt,selected_ds,choice_key,args,prediction_dir)
            elif args.metric == 'cf_edit':
                cf_ds_dict = {d['sample_id']:d for d in edited_known_ds}
                selected_ids = [s for s in selected_ids if s in cf_ds_dict.keys() and cf_ds_dict[s]['valid_cf_edit']]
                remaining_edit_ids = args.num_samples - len(selected_ids) # issue wif cf_edit: some samples are not valid
                for k,v in cf_ds_dict.items(): # gather more instances until args.num_samples
                    if k not in selected_ids:
                        if v['valid_cf_edit']:
                            selected_ids.append(k)
                    if len(selected_ids) >= args.num_samples:
                        break
                
                cf_selected_ds = {k:v for k,v in cf_ds_dict.items() if k in selected_ids}
                compute_cf_edit(cf_selected_ds,prediction_dir,args)
            elif args.metric == 'plausibility':
                get_plaus_score(selected_ds,args,prediction_dir,seed)

if __name__ == "__main__":
    main()