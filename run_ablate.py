import os, re, json
import torch
from utils.causal_trace import *
from utils.extra_utils import *
from utils.prediction import TorchDS
from utils.model_data_utils import ModelAndTokenizer,get_model_path
import argparse
import pickle
from utils.attacks import compute_causal_values
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
    aa("--replace", default=0, type=int)
    aa("--noise_level",default = 's3',type = str)
    aa("--corrupted_samples",default = 5,type = int)
    aa("--batch_size",default = 32,type = int)
    aa("--expl_type",default = 'cot',type = str,choices = ['post_hoc','cot']) 
    aa("--ablate",default = 'subject_1',type = str)
    args = parser.parse_args()
    seed_all(0)

    model_path,use_fs = get_model_path(args.model_name,args.expl_type)
    ds_dir = f'data/original/0'
    known_ds_path = f"{ds_dir}/{args.model_name}_{args.dataset_name}_{args.expl_type}.jsonl"

    mt = ModelAndTokenizer(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16) if 'gpt2' not in args.model_name else torch.float32,
    m_name = args.model_name
    )

    if args.dataset_name == 'csqa':
        choice_key = ['text']
    else:
        choice_key = []

    with open(known_ds_path,'r') as f:
        known_ds = [json.loads(l) for l in f]
    
    if 'noise' in args.ablate:
        noise_std = float(args.ablate.split('_')[-1])
    elif 'subject' in args.ablate: # else is subject
        noise_std = float(args.noise_level[1:])
        num_to_add = int(args.ablate.split('_')[-1])
        if 'back' in args.ablate:
            direction = 'back'
        else:
            direction = 'front'
        for d in known_ds:
            adjusted_subj = add_extra_tokens_to_subject(d['subject'],d['question'],mt.tokenizer,num_to_add,direction)
            d['subject'] = adjusted_subj
    else:
        raise ValueError(f'Unknown ablation type {args.ablate}')

    noise_level = noise_std * collect_embedding_std(mt, [k['subject'] for k in known_ds])
    prediction_dir = f'prediction/{args.model_name}/{args.dataset_name}/0'
    non_ablated_path = os.path.join(prediction_dir,f'{args.expl_type}_output_original.pkl')
    non_ablated_ds = pickle.load(open(non_ablated_path,'rb'))
    sample_ids = set(non_ablated_ds.keys())
    
    known_ds = [d for d in known_ds if d['sample_id'] in sample_ids]
    """
    1) get the sample_ids of the original predictions
    2) re-compute the low_score of the outputs p*(o)
    3) compute the IE of both the outputs and expls
    """
    new_ds = []
    corrupted_ds = TorchDS(known_ds,mt.tokenizer,choice_key,args.model_name,use_fs = use_fs,ds_name = args.dataset_name,expl = None if args.expl_type == 'post_hoc' else args.expl_type,corrupt = True,ds_type = 'original')
    for s in tqdm(corrupted_ds.batched_ds,total = len(corrupted_ds),desc = f'Ablating samples for {args.ablate}'):
        prompt = s['input_ids']
        subject = s['subject']
        answer_t = tokenize_single_answer(s['answer'],mt.tokenizer,args.model_name)
        corrupt_range = find_token_range(mt.tokenizer, prompt, subject,find_sub_range=use_fs)
        low_score,_ = trace_with_patch(
            mt.model, prompt.repeat(1+args.corrupted_samples,1).to(mt.model.device), [], answer_t, corrupt_range, noise=noise_level, uniform_noise=False,past_kv = None ,num_samples = args.corrupted_samples
        )
        corrupted_ds.ds[s['sample_id']]['low_score'] = low_score.item()
        corrupted_ds.ds[s['sample_id']]['difference'] = corrupted_ds.ds[s['sample_id']]['high_score'] - low_score.item()
        new_ds.append(corrupted_ds.ds[s['sample_id']])
    new_ds = sorted(new_ds,key = lambda x: x['difference'],reverse=True)
    
    
    ablated_output,ablated_expl = compute_causal_values(new_ds,mt,choice_key,use_fs,'original',noise_level,args)

    o_path = os.path.join(prediction_dir,f'{args.expl_type}_output_{args.ablate}.pkl')
    e_path = os.path.join(prediction_dir,f'{args.expl_type}_expl_{args.ablate}.pkl')

    with open(o_path,'wb') as f:
        pickle.dump(ablated_output,f)
    with open(e_path,'wb') as f:
        pickle.dump(ablated_expl,f)

if __name__ == "__main__":
    main()