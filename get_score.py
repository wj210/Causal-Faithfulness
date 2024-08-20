import numpy as np
import json,pickle,os
from collections import defaultdict
from scipy import spatial
from utils.causal_trace import find_token_range
from utils.model_data_utils import get_model_path
from transformers import AutoTokenizer

def prune_sequence(seq1, seq2): # seq1 is the longer sequence, seq2 is the shorter sequence
    i, j = 0, 0
    original_indices = list(range(len(seq1)))  # Track original positions
    pruned_positions = []  # List to store the positions of pruned tokens
    
    while j < len(seq2):
        if i < len(seq1) and seq1[i] == seq2[j]:
            i += 1
            j += 1
        else:
            pruned_positions.append(original_indices[i])
            seq1.pop(i)
            original_indices.pop(i)
    
    # Any remaining tokens in seq1 after seq2 is fully matched should be pruned
    while i < len(seq1):
        pruned_positions.append(original_indices[i])
        seq1.pop(i)
        original_indices.pop(i)

    return seq1, pruned_positions

class ScoreObj:
    def __init__(self,scores,subject_range,input_tokens):
        self.scores = scores
        self.subject_range = subject_range
        self.input_tokens = input_tokens

def compute_divg(o,e,compare = 'o-e'):
    if compare == 'o_e': # if output explanation, e is the explanation, take until the length of output
        e = e[:o.shape[0]]
    else: # else is either o-o or e-e, the input length of the paraphrased/cf may have difference of 1 token due to tokenizing, just take until the shortest.
        shortest = min(o.shape[0],e.shape[0])
        o = o[:shortest]
        e = e[:shortest]
    if len(o.shape) > 1:
        o = o.flatten()
        e = e.flatten()
    return 1. - spatial.distance.cosine(o, e)

def compute_full_divg(o_path,e_path,ds_path,tokenizer):
    with open(o_path,'rb') as f:
        o_r = pickle.load(f)
    with open(e_path,'rb') as f:
        e_r = pickle.load(f)
    
    with open(ds_path,'r') as f:
        ds = [json.loads(l) for l in f.readlines()]
    ds = {d['sample_id']:d for d in ds}

    all_scores = []
    all_samples = {}
    for s_id,o_d in o_r.items():
        e_d = e_r.get(s_id,None)
        if not e_d:
            continue
        o_score = o_d['scores']
        e_score = e_d['scores']
        
        if 'consistency' in ds_path:
            subj = ds[s_id]['paraphrase_subject']
        else:
            subj = ds[s_id]['subject']
        subj_range = find_token_range(tokenizer,o_d['input_tokens'],subj)

        o_store = ScoreObj(o_score,subj_range,o_d['input_tokens'])
        e_store = ScoreObj(e_score,subj_range,e_d['input_tokens'])
        dist_scores = compute_divg(o_score,e_score) 
        
        all_scores.append(dist_scores)
        all_samples[s_id] = (o_store,e_store)
    
    total_mean = np.mean(all_scores)
    
    return all_samples,total_mean

def average_subj_scores(s_obj,s_obj2):
    """
    This is mainly used for FEC/cf where the 2 inputs may not match in size, ie the cf/p subject may have different number of tokens.
    We average the contributions of the subject as a whole, and ensure the total size is similar and comparable.
    vec are the attribution score for each token. [token len, num layers]
    """
    avg_objs = []
    for obj in [s_obj,s_obj2]:
        if isinstance(obj,dict):
            obj = ScoreObj(obj['scores'],obj['subject_range'],obj['input_tokens'])
        subj_s,subj_e = obj.subject_range
        vec = obj.scores
        subj_vec = vec[subj_s:subj_e].mean(axis=0).reshape(1,-1)
        obj_scores = np.concatenate([vec[:subj_s],subj_vec,vec[subj_e:]],axis=0)
        avg_objs.append(obj_scores)
    if avg_objs[0].shape[0] != avg_objs[1].shape[0]: # means input_tokens not aligned somehow due to tokenization etc.
        # Find the longer seq and take away the dissimilar token apart from the averaged subject
        placeholder_token = ["[PAD]"] # to stand in for the avg token
        if avg_objs[0].shape[0] > avg_objs[1].shape[0]:
            longer = 0
            longer_seq = s_obj.input_tokens[:s_obj.subject_range[0]] + placeholder_token + s_obj.input_tokens[s_obj.subject_range[1]:]
            shorter_seq = s_obj2.input_tokens[:s_obj2.subject_range[0]] + placeholder_token + s_obj2.input_tokens[s_obj2.subject_range[1]:]
        else:
            longer = 1
            longer_seq= s_obj2.input_tokens[:s_obj2.subject_range[0]] + placeholder_token + s_obj2.input_tokens[s_obj2.subject_range[1]:]
            shorter_seq = s_obj.input_tokens[:s_obj.subject_range[0]] + placeholder_token + s_obj.input_tokens[s_obj.subject_range[1]:]
        longer_seq,pruned_positions = prune_sequence(longer_seq,shorter_seq)
        if longer == 0:
            avg_objs[0] = np.delete(avg_objs[0],pruned_positions,axis=0)
        else:
            avg_objs[1] = np.delete(avg_objs[1],pruned_positions,axis=0)
    return avg_objs


def main():
    model_name = 'llama3-8B-chat' # change the model name here
    expl_type = 'post_hoc' # either post_hoc or cot
    metric = 'FEAC' # either FEC or FEAC
    ds = 'csqa' # either csqa,esnli,arc
    model_path,_ = get_model_path(model_name,'post_hoc')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_seeds = 3 # unless running multiple seeds
    scores = []
    for seed in range(num_seeds):
        prediction_dir = f'prediction/{model_name}/{ds}/{seed}'
        output_path = os.path.join(prediction_dir,f'{expl_type}_output_original.pkl')
        expl_path = os.path.join(prediction_dir,f'{expl_type}_expl_original.pkl')
        ori_ds_path = f'data/original/{seed}/{ds}/{model_name}_{expl_type}.jsonl'
        p_ds_path = f'data/consistency/{seed}/{ds}/{model_name}_{expl_type}_paraphrase.jsonl'
        original_samples,fec = compute_full_divg(output_path,expl_path,ori_ds_path,tokenizer)

        if metric == 'FEAC':
            scores.append(fec)
        else:
            fec = []
            consis_output_path = os.path.join(prediction_dir,f'{expl_type}_output_paraphrase.pkl')
            consis_expl_path = os.path.join(prediction_dir,f'{expl_type}_expl_paraphrase.pkl')
            consis_samples,_ = compute_full_divg(consis_output_path,consis_expl_path,p_ds_path,tokenizer)

            # Match via the sample_id
            for sample_id,(_,e_s) in original_samples.items():
                if sample_id not in consis_samples:
                    continue
                _,c_e_s = consis_samples[sample_id]
                if e_s.scores.shape[0] != c_e_s.scores.shape[0]: # do the averaging here to match
                    try:
                        e_s,c_e_s = average_subj_scores(e_s,c_e_s)
                    except Exception as e:
                        continue
                else:
                    e_s,c_e_s = e_s.scores,c_e_s.scores
                expl_divg = compute_divg(e_s,c_e_s,compare = 'e_e') 
                fec.append(expl_divg)
            
            scores.append(np.mean(fec))
    
    print (f'Average {metric} for {model_name} with {expl_type} on {ds}: {np.mean(scores):.3f}')
            
if __name__ == '__main__':
    main()