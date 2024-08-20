import os, re, json
import torch, numpy,random
from utils.causal_trace import *
from utils.extra_utils import *
from utils.prediction import *
from utils.model_data_utils import ModelAndTokenizer,get_model_path,load_hf_ds
import argparse
import pickle
from utils.attacks import paraphrase_instruction
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
    aa("--seed", default=0, type=int)
    aa("--num_samples",default = 100,type = int)
    aa("--corrupted_samples",default = 5,type = int)
    aa("--num_expl",default = 3,type = int)
    aa("--batch_size",default = 32,type = int)
    aa("--expl_type",default = 'cot',type = str,choices = ['post_hoc','cot']) 
    aa("--openai_api_key", default="openai_key.txt")
    args = parser.parse_args()
    seed_all(args.seed) # seed here


    model_path,use_fs = get_model_path(args.model_name,args.expl_type)
    known_ds_dir = f'data/original/{args.seed}/{args.dataset_name}'
    os.makedirs(known_ds_dir,exist_ok=True)
    
    # if int(args.noise_level[-1]) != 3:
    ablate = True
    add_str = f"_{args.noise_level}"
    # else:
    #     ablate = False
    #     add_str = ''
        
    
    known_ds_path = f"{known_ds_dir}/{args.model_name}_{args.expl_type}{add_str}.jsonl"

    os.environ['OPENAI_API_KEY'] = open(args.openai_api_key,'r').read().strip()

    
    choice_key = []

    ## Load model ##
    mt = ModelAndTokenizer(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.bfloat16),
    m_name = args.model_name
    )

    ## Get known dataset for the original (no edits) - get low score/high score for each sample and filter out unknowns according to model/exp ##
    if not os.path.exists(known_ds_path):
        ds,choice_key = load_hf_ds(args.dataset_name,args.seed)
        checked_ds,_ = check_subject(ds)
        checked_ds = filter_samples(checked_ds)
        print (f'omitted {len(ds)- len(checked_ds)} samples, left {len(checked_ds)} samples')
        known_ds,_ = get_known_dataset(checked_ds,mt,batch_size = args.batch_size,choice_key = choice_key,model_name = args.model_name,args = args,use_fs = use_fs,ds_type ='original')
        os.makedirs('data',exist_ok=True) # known_ds sorted by difference
        # save the known ds according to ds_name and model
        with open(known_ds_path,'w') as f:
            for d in known_ds:
                f.write(json.dumps(d)+'\n')
    else:
        with open(known_ds_path,'r') as f:
            known_ds = [json.loads(l) for l in f]
        print (f'{known_ds_path} exists!!')
    
    # if ablate:
    #     exit('No need edits')

    ## Edited (paraphrase/counterfactual)
    edit_path = f'data/consistency/{args.dataset_name}.jsonl'
    edited_known_dir = f"data/consistency/{args.seed}/{args.dataset_name}"
    os.makedirs(edited_known_dir,exist_ok=True)
    edit_types = ['paraphrase','cf'] if args.expl_type == 'post_hoc' and not ablate else ['paraphrase']
    for edit_type in edit_types:
        generate_edit = False
        os.makedirs(os.path.dirname(edit_path),exist_ok=True)
        if not os.path.exists(edit_path):
            generate_edit = True
            ds,choice_key = load_hf_ds(args.dataset_name,args.seed)
            checked_ds,_ = check_subject(ds)
            edited_ds = filter_samples(checked_ds)
        else:
            with open(edit_path,'r') as f:
                edited_ds = [json.loads(l) for l in f]
        if f'{edit_type}_subject' not in edited_ds[0]:
            generate_edit = True

        if generate_edit: # reload the original ds to get the paraphrased/counterfactual dataset
            edited_ds = paraphrase_instruction(edited_ds,choice_key,args,edit_path,edit_type)
        
        ## for cf, need to add in the pred from the original dataset for checking of counterfactual conditions.
        if edit_type == 'cf':
            known_ds = {d['sample_id']:d for d in known_ds}
            for ed in edited_ds:
                ed['pred'] = known_ds[ed['sample_id']]['pred']

        edited_known_ds_path = f"{edited_known_dir}/{args.model_name}_{args.expl_type}_{edit_type}{add_str}.jsonl"
        if not os.path.exists(edited_known_ds_path):
            edited_known_ds,_ = get_known_dataset(edited_ds,mt,batch_size = args.batch_size,choice_key = choice_key,model_name = args.model_name,args = args,use_fs = use_fs,ds_type = edit_type)
            with open(edited_known_ds_path,'w') as f:
                for d in edited_known_ds:
                    f.write(json.dumps(d)+'\n')
        else:
            print (f'{edited_known_ds_path} exists!!')

if __name__ == '__main__':
    main()