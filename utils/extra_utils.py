import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils
import re
import time
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import numpy as np
import random
from collections import defaultdict
import json

def seed_all(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def join_choices(choices):
    return "\n".join([f"({chr(i+97).upper()}) {c}" for i,c in enumerate(choices)])

def alpha_to_int(alpha):
    return ord(alpha.lower()) - 97

def int_to_alpha(i):
    return chr(i+97).upper()

def format_input(x,tokenizer,fs = []):
    if not isinstance(x,list):
        x = [{'role':'user','content':x}]
    x = fs + x
    # if x[0]['role'] != 'system' and 'gemma' not in tokenizer.name_or_path:
    #     x.insert(0,{'role':'system','content':"You are a helpful assistant who follows instruction well."})
    x = tokenizer.apply_chat_template(x,tokenize = False,add_generation_prompt = True)
    return x

def untuple_dict(x,ks):
    if len(ks) > 0:
        for k in ks:
            x = x[k]
    else:
        return x
    return x

def unroll_list(x):
    if isinstance(x[0],list):
        return sum(x,[])
    else:
        return x

def list_of_alpha(n):
    return [chr(i+97).upper() for i in range(n)]

def parse_bracket(s):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, s)
    if len(matches) >= 1:
        a = matches[0]
    else:
        if '(' in s:
            a_pos  = s.index('(')
            if a_pos >= len(s) - 1:
                return None
            a = s[a_pos+1]
        else:
            return None
    return a.upper()

def get_cot_answer(s):
    s_behind = s.split('The best answer')
    if len(s_behind) == 1:
        return None
    else:
        s_behind = s_behind[1].strip()
    alpha_ans = parse_bracket(s_behind)
    if not isinstance(alpha_ans,str):
        return None
    return alpha_ans

def truncate_to_last_period(input_string):
    last_period_index = input_string.rfind('.')
    # If no period is found, return the original string
    if last_period_index == -1:
        return input_string
    # Truncate the string up to and including the last period
    truncated_string = input_string[:last_period_index + 1]
    return truncated_string

def check_subject(ds,type_ = 'original'):
    if type_ == 'original':
        subject_key = 'subject'
        question_key = 'question'
    else:
        subject_key = f'{type_}_subject'
        question_key = f'{type_}_question'
    checked_ds = []
    for d in ds:
        if d[subject_key] in d[question_key] and d[subject_key] != 'None': # May have none for cf/paraphrase
            checked_ds.append(d)
    return checked_ds,len(ds) - len(checked_ds)


def filter_samples(ds):
    out_ds = []
    for d in ds:
        take = True
        instr_split = remove_punctuations(d['question']).split()
        sub = remove_punctuations(d['subject']).split()
        for i,s in enumerate(instr_split):
            if sub[0] in s:
                if (len(sub) > 1 and " ".join(sub) == " ".join(instr_split[i:i+len(sub)])) or len(sub) == 1:
                    if len(instr_split) - (i+len(sub)) <= 3: # if the subject is in the last 3 tokens, we dont take
                        take = False
                        break
        if take:
            out_ds.append(d)
    return out_ds

def top_n_indices(tensor, N):
    flattened_tensor = tensor.flatten()
    top_n_values, top_n_indices_flat = torch.topk(flattened_tensor, N)
    top_n_indices_2d = []
    for idx in top_n_indices_flat:
        row = idx // tensor.size(1)
        col = idx % tensor.size(1)
        top_n_indices_2d.append((row.item(), col.item()))
    return top_n_indices_2d, top_n_values

def clean_explanation(e):
    if 'Question' in e:
        return e.split('Question')[0].strip()
    else:
        e = e.split('\n\n')[0].strip()
    return e

def tokenize_single_answer(x,tokenizer,model_name):
    if 'llama3'in model_name or 'gpt2' in model_name or 'phi2' in model_name or 'gemma' in model_name:
        return tokenizer.encode(x,add_special_tokens=False)[0]
    else:
        return tokenizer.encode('\n'+x,add_special_tokens=False)[2:][0] # other models have leading char infront. remove it

def cal_cost(model_name,in_tokens,out_tokens):
    if model_name == 'gpt-4':
        cost = in_tokens * (10/1e6) + (out_tokens * (30/1e6))
    elif model_name == 'gpt-4o':
        cost = in_tokens * (5/1e6) + (out_tokens * (15/1e6))
    elif model_name == 'gpt-3.5-turbo-0125':
        cost = in_tokens * (0.5/1e6) + (out_tokens * (1.5/1e6))
    elif model_name == 'gpt-3.5-turbo-instruct':
        cost = in_tokens * (1.5/1e6) + (out_tokens * (2/1e6))
    else:
        raise NotImplementedError
    return cost

def async_process(fn,inps,workers=10,msg=''):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        if msg == '':
            out = list(tqdm(executor.map(fn,inps),total = len(inps)))
        else:
            out = list(tqdm(executor.map(fn,inps),total = len(inps),desc = msg))
    return out

def openai_call(message,model,max_tokens,temperature=0.,n=1):
    client = OpenAI()
    max_calls = 5
    num_calls = 0
    while True:
        if num_calls > max_calls:
            return None,None
        try:
            if 'instruct' in model.lower():
                prompt = ''
                for m in message:
                    prompt += m['content']
                    if m['role'] == 'assistant':
                        prompt += '\n\n'
                response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n = n,
                )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                if n > 1:
                    return [r.text for r in response.choices],cost
                else:
                    return response.choices[0].text,cost
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                if n > 1:
                    return [r.message.content for r in response.choices],cost
                else:
                    return response.choices[0].message.content,cost
        except Exception as e:
            num_calls += 1
            time.sleep(num_calls**2)
            print(f'Failing Openai call due to {e}, remaining calls: {max_calls - num_calls}')


def get_common_samples(paths,type_):
    """
    pick only samples that are correct in all paths
    """
    common_samples = defaultdict(list)
    common_subjects = {}
    for p in paths:
        with open(p,'r') as f:
            data = [json.loads(l) for l in f]
        for d in data:
            if d['sample_id'] not in common_subjects and d['correct']:
                if type_ == 'original':
                    common_subjects[d['sample_id']] = d['question']
                else:
                    common_subjects[d['sample_id']] = d[f'{type_}_question']
            common_samples[d['sample_id']].append(d['pred'])
            
    
    common_samples = {k:v for k,v in common_samples.items() if len(v) == len(paths)}
    common_subjects = [v for k,v in common_subjects.items() if k in common_samples] # get the subjects of the common samples to get noise

    return common_samples,common_subjects

def remove_punctuations(s):
    return re.sub(r'[^\w\s]','',s)



