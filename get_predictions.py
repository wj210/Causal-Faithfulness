import os, re, json
import torch, numpy,random
from utils.causal_trace import *
from utils.extra_utils import *
from utils.prediction import *
from utils.model_data_utils import ModelAndTokenizer,get_model_path,load_hf_ds
from utils.attacks import paraphrase_instruction
import argparse
from transformers import AutoTokenizer
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
    aa("--seed", default=0, type=int)
    aa("--num_samples",default = 100,type = int)
    aa("--num_expl",default = 3,type = int)
    aa("--batch_size",default = 32,type = int)
    aa("--openai_api_key", default="openai_key.txt")
    aa("--mode", default="STR",choices = ['STR','GN']) # either token replacement or gaussian noise
    aa("--ood_analysis", action = 'store_true')
    args = parser.parse_args()
    seed_all(args.seed) # seed here

    # assert os.path.exists(args.openai_api_key) , "Please provide the openai api key to make edits on the input"
    # os.environ['OPENAI_API_KEY'] = open(args.openai_api_key,'r').read().strip()

    model_path = get_model_path(args.model_name)
    known_ds_dir = f'data/{args.seed}/{args.dataset_name}'
    os.makedirs(known_ds_dir,exist_ok=True)
    if not args.ood_analysis:
        ph_ds_path = f"{known_ds_dir}/{args.model_name}.jsonl"
    else:
        ph_ds_path = f"{known_ds_dir}/{args.model_name}_ood_{args.mode}.jsonl"
        args.num_samples = -1
    cf_ds_path = f"data/{args.dataset_name}_cf.jsonl"

    os.makedirs(os.path.dirname(ph_ds_path),exist_ok=True)

    ## Load model ##
    mt = ModelAndTokenizer(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.bfloat16),
    m_name = args.model_name
    )
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2B-it")
    
    ## Get known dataset for the original (no edits) - get low score/high score for each sample and filter out unknowns according to model/exp ##
    if not os.path.exists(ph_ds_path):
        if not os.path.exists(cf_ds_path):
            ds = load_hf_ds(args.dataset_name,args.seed,tokenizer = mt.tokenizer)
            if args.dataset_name != 'comve':
                checked_ds,_ = check_subject(ds) # check if subject exist in question
                checked_ds = filter_samples(checked_ds) # remove questiosn where no of tokens after subject is too little
                print (f'omitted {len(ds)- len(checked_ds)} samples, left {len(checked_ds)} samples')
                sorted_ds = sort_by_earliest_subject(checked_ds,tokenizer) # sort by the earliest subject
                cf_ds = paraphrase_instruction(sorted_ds,args,cf_ds_path)
            else:
                cf_ds = ds
        else:
            with open(cf_ds_path,'r') as f:
                cf_ds = [json.loads(l) for l in f][:args.num_samples]
        if not args.ood_analysis:
            ph_ds = get_answer_and_explanation(cf_ds,mt,batch_size = args.batch_size,model_name = args.model_name,args = args)
        else:
            if args.mode == 'STR':
                ph_ds = get_answer_and_explanation_ood(cf_ds,mt,batch_size = args.batch_size,model_name = args.model_name,args = args)
            else:
                ph_ds = get_answer_and_explanation_GN(cf_ds,mt,batch_size = args.batch_size,model_name = args.model_name,args = args)
        # save the known ds according to ds_name and model
        with open(ph_ds_path,'w') as f:
            for d in ph_ds:
                f.write(json.dumps(d)+'\n')
    else:
        print (f'{ph_ds_path} exists!!')

if __name__ == '__main__':
    main()