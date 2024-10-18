import numpy as np
import string
from utils.extra_utils import *
from utils.causal_trace import find_token_range,generate_sequence_probs,collect_embedding_std,trace_with_patch
from utils.fewshot import fs_examples

post_hoc_prompt = "Based on the answer, explain in a short sentence why?"
cot_prompt = "Please verbalize shortly and concisely about how you are thinking about the problem, then give your answer in the format 'The best answer is: (X)'. It's very important that you stick to this format."
sep_token = '[SEP]' # just to separate dialog for chat models.

gen_kwargs = {'max_new_tokens':64,
              'do_sample':True,
              'temperature':1.0,
              'top_p':0.95
              }

def format_mcq(question,choices,answer=None,expl_=None,is_chat = True):
    """
    Note this does not add explanation generation, only the instructions or answer
    For Normal: x or x:y if answer provided
    For Post-hoc: x+y
    For CoT: only x, answer is not added since it is after explanation.
    """
    base_prompt = "Question: {question}\n\nChoices:\n{choices}\n\nAnswer: ".format_map({'question':question,'choices':choices}) 
    if not expl_: # just prompt for answer
        if not is_chat:
            base_prompt += 'The best answer is ('
    else: # prompt for post_hoc explanation given answer
        assert  answer is not None, 'Answer is required for post_hoc explanation'
        if not isinstance(answer,str):
            answer = chr(answer+97).upper()
        if not is_chat:
            base_prompt += f'The best answer is ({answer}).'
        else:
            base_prompt += f"{sep_token}The best answer is ({answer})."

    if expl_:
        if is_chat:
            base_prompt += sep_token + post_hoc_prompt
        else:
            base_prompt += '\n' + post_hoc_prompt
    
    if is_chat: # format into list of dicts
        base_prompt = base_prompt.split(sep_token)
        o = []
        for i,p in enumerate(base_prompt):
            if i%2 == 0:
                o.append({'role':'user','content':p})
            else:
                o.append({'role':'assistant','content':p})
        base_prompt = o
    return base_prompt

class TorchDS(torch.utils.data.Dataset):
    def __init__(self,ds,tokenizer,model_name,use_fs = False,ds_name='csqa',expl = None,corrupt=False,mode = 'STR'):
        self.ds = ds
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.alpha_tokenized = {a: tokenize_single_answer(a,tokenizer,model_name) for a in list_of_alpha(26)} # llama3 do not have leading char, others have.
        self.use_fs = use_fs
        self.ds_name = ds_name
        # if self.use_fs:
        self.fs = format_fs(fs_examples[self.ds_name],chat =False,expl = expl) if 'chat' not in self.model_name else []
        # self.fs = [] if 'chat' in self.model_name else ''
        self.is_chat = True if 'chat' in model_name else False
        self.expl = expl
        self.corrupt = corrupt
        self.mode = mode
        if mode == 'ood':
            assert ds_name == 'comve' and not expl, 'OOD mode only supported for comve and only answers'
        if self.expl:
            self.tokenizer.padding_side = 'left'
        self.setup()
    
    def setup(self):
        additional_suffix = self.get_additional_prompt()
        self.batched_ds = []
        ds_copy = {}
        if self.mode != 'ood':
            answer_key = 'answer' if not self.corrupt and not self.expl else 'pred' # only when evaluating acc, do we use answer else always pred
        else:
            answer_key = 'answer'
        for sample_id,d in enumerate(self.ds):
            question = d['question']
            choices = d.get('choices',None)
            joined_choices = join_choices(choices)
            num_choices = len(choices)
            answer = d[answer_key] 
            if d.get('sample_id',None) is not None: # to maintain the tracking
                sample_id = d['sample_id']
            else:
                d['sample_id'] = sample_id
            formatted_question = format_mcq(question,joined_choices,is_chat = self.is_chat,expl_=self.expl,answer = answer if self.expl else None)
            if 'chat' in self.model_name:
                formatted_question = format_input(formatted_question,self.tokenizer,fs = self.fs)
            else:
                formatted_question = self.fs + formatted_question
            formatted_input = formatted_question+additional_suffix
            tokenized_context = torch.tensor(self.tokenizer.encode(formatted_input),dtype=torch.long) 
            input_length = tokenized_context.shape[0]
            data_dict = {'input_ids': tokenized_context,'input_length':input_length,'answer':answer,'num_choices':num_choices,'subject':d['subject'],'sample_id': sample_id}
            data_dict['prompt'] = formatted_input

            if self.mode in ['STR','ood']: # add the cf to see performance degrade
                cf_input = formatted_input.replace(question,d['cf_question'])
                cf_tokenized_context = torch.tensor(self.tokenizer.encode(cf_input),dtype=torch.long)
                cf_input_length = cf_tokenized_context.shape[0]
                data_dict['cf_input_ids'] = cf_tokenized_context
                data_dict['cf_input_length'] = cf_input_length
                if self.mode == 'ood':
                    data_dict['cf_answer'] = d['cf_answer']

            self.batched_ds.append(data_dict)
            ds_copy[sample_id] = d
        self.ds = ds_copy
    

    def get_additional_prompt(self):
        if not self.expl:
            if 'chat' in self.model_name:
                x = 'The best answer is ('
            else:
                x = ''
        else:
            x = ' Because'
        if 'chat' in self.model_name:
            x = x.strip()
        return x
    
    def __len__(self):
        return len(self.batched_ds)
    
    def __getitem__(self,idx):
        return self.batched_ds[idx]
    
    def collate_fn(self,batch):
        input_token_ids = [b['input_ids'] for b in batch]
        input_length = [b['input_length'] for b in batch]
        answer = [b['answer'] for b in batch]
        num_choices = [b['num_choices'] for b in batch]
        sample_id = [b['sample_id'] for b in batch]
        prompts = [b['prompt'] for b in batch]
        subjects = [b['subject'] for b in batch]
        out = {'answer':answer,
                'input_length':input_length,
                "num_choices":num_choices,
                'sample_id':sample_id,
                'subject':subjects
                }
        if not self.expl:
            padded_input = pad_sequence(input_token_ids,batch_first=True,padding_value=self.tokenizer.pad_token_id)
            if 'cf_input_ids' in batch[0]:
                out['cf_input_length'] = [b['cf_input_length'] for b in batch]
                out['cf_input_ids']= pad_sequence([b['cf_input_ids'] for b in batch],batch_first=True,padding_value=self.tokenizer.pad_token_id)
                
        else:
            padded_input = self.tokenizer(prompts,padding='longest' if len(prompts) > 1 else None,return_tensors='pt',truncation=False)
        out['input_ids'] = padded_input
        out['prompt'] = prompts
        if 'cf_answer' in batch[0]:
            out['cf_answer'] = [b['cf_answer'] for b in batch]
        return out
    
    def get_normalized_llh(self,logits,label_ids,input_len): # only used for sequence output
        """
        logits should be a tensor of shape (batch,seq_len,vocab_size)
        label_ids should be the corresponding choice ids = a list of tensor, each tensor = (seq_len,)
        input_len = list of int, each int = length of the unpadded logits
        """
        logprobs = torch.nn.functional.log_softmax(logits,dim=-1)
        loglikelihoods = []
        for lp,label,leng in zip(logprobs,label_ids,input_len):
            lp = lp[:leng]
            lp = lp[-label.shape[0]:]
            lp = torch.gather(lp,1,label.unsqueeze(1)).squeeze(-1)
            normalized_lp = lp.sum()/label.shape[0]
            loglikelihoods.append(normalized_lp.numpy())
        return loglikelihoods
    

def format_fs(fs,chat = True,expl = None):
    out = []
    for f in fs:
        instr = format_mcq(f['question'],join_choices(f['choices']),answer =f['answer'],is_chat = chat,expl_=expl)
        if expl:
            suffix = f' Because {f["explanation"]}'
        else:
            suffix = f"{f['answer']})."
        if chat:
            out.extend(instr)
            if suffix is not None:
                out.append({'role':'assistant','content':suffix.strip()})
        else:
            out.append(instr)
            if suffix is not None:
                out[-1] += suffix
    if not chat:
        out = "\n\n".join(out) + "\n\n"
    return out


def decompose_cot(seq,scores,tokenizer):
    expl_end,_ = find_token_range(tokenizer,seq,'The best answer')
    if expl_end is None:
        expl_probs = np.mean(scores.exp().cpu().numpy())
        return tokenizer.decode(seq,skip_special_tokens=True),expl_probs,'',''
    else:
        expl_probs = np.mean(scores[:expl_end].exp().cpu().numpy())
        expl = tokenizer.decode(seq[:expl_end],skip_special_tokens=True)
    cot_answer_range = seq[expl_end:]
    if scores != None:
        scores = scores[expl_end:]
    else:
        scores = None
    cot_answer_start,_ = find_token_range(tokenizer,cot_answer_range,'(')
    if not cot_answer_start or cot_answer_start+1 >= cot_answer_range.shape[0]:
        return expl,expl_probs,'',''
    else: # return both the answer and the score
        return expl,expl_probs,tokenizer.decode(cot_answer_range[cot_answer_start+1],skip_special_tokens=True),scores[cot_answer_start+1].exp().item() if scores is not None else None
    
def top_p_sampling(logits, temperature=1.0, top_p=0.95):
    # Apply temperature scaling
    logits = logits / temperature
    # Sort the logits and calculate cumulative probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    # Filter out tokens with cumulative probabilities above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    if sorted_indices_to_remove.any():
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
    # Set logits of tokens to be removed to a large negative value
    sorted_logits[sorted_indices_to_remove] = -float('Inf')
    
    # Sample from the filtered distribution
    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    # Convert the sampled index back to the original indices
    return sorted_indices[sampled_index]

def get_pred(mt,prompts,num_choices,inp_lens=None,ans_poss = None):
    if isinstance(prompts,str):
        prompts = [prompts]
    if not torch.is_tensor(prompts):
        inp = [torch.tensor(mt.tokenizer.encode(p),dtype=torch.long) for p in prompts]
        inp_lens = [i.shape[0] for i in inp]
        if len(inp) > 1:
            inps = pad_sequence(inp,batch_first=True,padding_value=mt.tokenizer.pad_token_id)
        else:
            inps = inp[0].unsqueeze(0)
    else:
        inps = prompts
        assert inp_lens is not None, 'Input lens is required for tensor input'
    with torch.no_grad():
        logits = mt.model(inps.to(mt.model.device))["logits"]
    if not isinstance(num_choices,list):
        num_choices = [num_choices for _ in range(inps.shape[0])] # in the event when num choices is passed in as a list and diff btween samples
    preds = []
    to_zip = zip(logits,inp_lens,num_choices) if not ans_poss else zip(logits,inp_lens,num_choices,ans_poss)

    for i,zipped in enumerate(to_zip):
        logit = zipped[0]
        l = zipped[1]
        nc = zipped[2]
        if len(zipped) == 4:
            ans_pos = zipped[3]
        else:
            ans_pos = None
        logit = logit[:l][-1].detach().cpu()
        probs = torch.nn.functional.softmax(logit,dim=-1)
        choice_logits = []
        choice_probs = []
        if ans_pos is not None:
            if not isinstance(ans_pos,list):
                ans_pos = alpha_to_int(ans_pos)
            else:
                ans_pos = [alpha_to_int(a) for a in ans_pos]
        
        for c in list_of_alpha(nc):
            choice_id = tokenize_single_answer(c,mt.tokenizer,mt.model_name)
            choice_probs.append(probs[choice_id])
            choice_logits.append(logit[choice_id])
        choice_logits = torch.stack(choice_logits)
        choice_probs = torch.stack(choice_probs)
        if not isinstance(ans_pos,list):
            sampled_index = torch.multinomial(choice_probs,1)
        else: # we use greedy for ood analysis.
            sampled_index = torch.argmax(choice_logits)
        if ans_pos is None:
            sampled_logit = choice_logits[sampled_index]
            sampled_p = choice_probs[sampled_index]
        else:
            sampled_logit = choice_logits[ans_pos]
            sampled_p = choice_probs[ans_pos]
        ans_string = int_to_alpha(sampled_index.item())
        if ans_pos and isinstance(ans_pos,list):
            preds.append((ans_string,sampled_p.tolist(),sampled_logit.tolist()))
        else:
            preds.append((ans_string,sampled_p.item(),sampled_logit.item()))

    return preds

def generate_cot_response(input_ids,gen_kwargs,mt): # if input_ids is a list of tensors, all of them are assumed to have same shape.
    if not isinstance(input_ids,dict):
        attn_mask = torch.ones_like(input_ids).to(input_ids.device)
        input_ids = {'input_ids':input_ids,'attention_mask':attn_mask}
    with torch.no_grad():
        outputs = mt.model.generate(**input_ids,**gen_kwargs,return_dict_in_generate=True,output_scores=True,pad_token_id=mt.tokenizer.eos_token_id)
    seq_scores = mt.model.compute_transition_scores(outputs.sequences,outputs.scores,normalize_logits=True)
    out = []
    for seq_logits,seq_score,inp in zip(outputs.sequences,seq_scores,input_ids['input_ids']):
        output_seqs = seq_logits[inp.shape[-1]:]
        if 'chat' not in mt.model_name: # clean the explanation by finding Question:
            cot_end,_  = find_token_range(mt.tokenizer,output_seqs,'Question:')
            if cot_end is not None:
                output_seqs = output_seqs[:cot_end]
        if output_seqs.size()[0] == 0:
            out.append(['','',''])
            continue
        out.append(decompose_cot(output_seqs,seq_score,mt.tokenizer))
    return out # list of (cot expl, cot answer, ans prob)

def get_cot_prompt(input_ids,expl,tokenizer):
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    input_str = tokenizer.decode(input_ids,skip_special_tokens=True)
    if expl[-1] not in string.punctuation:
        expl += '.'
    return input_str + expl + ' The best answer is ('

def clean_nonchat_gen(x):
    if 'Question:' in x:
        return x.split('Question:')[0].strip()
    elif 'Q:' in x:
        return x.split('Q:')[0].strip()
    return x
    

def get_answer_and_explanation(ds,mt,batch_size = 8,model_name = 'llama3',args=None):
    """
    Given a dataset, 
    1) Get the acc score, original and corrupted score of answer and explanation (both prob and logit)
    keys to include: question, choices, answer, pred, explanation subject, sample_id, cf_answer, cf_question, cf_subject, prob and logit of high and low answer and explanation
    """
    ds_w_ans = []
    ds = TorchDS(ds,mt.tokenizer,model_name,ds_name = args.dataset_name,expl = None,corrupt = False,mode = args.mode) 
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=ds.collate_fn,drop_last = False)
    for batch in tqdm(dl,total = len(dl),desc = f'Getting accuracy and high score for answer, model: {mt.model_name}'):
        answer  = batch['answer']
        num_choices = batch['num_choices']
        sample_id = batch['sample_id']
        result_collection = {sid:ds.ds[sid] for sid in sample_id}
        # print (mt.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=True))
        # exit()
        joint_input_length = [(batch['input_ids'],batch['input_length'])]
        if 'cf_input_ids' in batch:
            joint_input_length.append((batch['cf_input_ids'],batch['cf_input_length']))
        
        ori_ans_store = []
        for s_id,(input_ids,input_len) in enumerate(joint_input_length):
            if torch.is_tensor(input_ids):
                input_ids = input_ids.to(mt.model.device)
            else:
                input_ids = {k:v.to(mt.model.device) for k,v in input_ids.items()}

            ### Post-Hoc answer ###
            preds = get_pred(mt,input_ids,num_choices,input_len,ans_poss = ori_ans_store if s_id == 1 else None)
            for p,a,sid in zip(preds,answer,sample_id):
                if s_id == 0:
                    result_collection[sid]['high_prob'] = p[1]
                    result_collection[sid]['high_logit'] = p[2]
                    result_collection[sid]['correct'] = p[0].upper() == a
                    result_collection[sid]['pred'] = p[0].upper()
                    ori_ans_store.append(p[0].upper())
                else: # store the corrupt results
                    result_collection[sid]['low_prob'] = p[1]
                    result_collection[sid]['low_logit'] = p[2]
                    result_collection[sid]['incorrect'] = p[0].upper() != a
                    result_collection[sid]['cf_pred'] = p[0].upper()
        ds_w_ans.extend(list(result_collection.values()))


    print (f'{args.model_name} Acc score: {sum([d["correct"] for d in ds_w_ans])/len(ds_w_ans):.2f}')
    if 'incorrect' in ds_w_ans[0]:
        print (f'{args.model_name} Corrupt Acc score: {1 - (sum([d["incorrect"] for d in ds_w_ans])/len(ds_w_ans)):.2f}')
        print (f'{args.model_name} Prob diff: {np.mean([(d["high_prob"]-d["low_prob"])/d["high_prob"] for d in ds_w_ans]):.2f}')
        print (f'{args.model_name} Logit diff: {np.mean([d["high_logit"]-d["low_logit"] for d in ds_w_ans]):.2f}')

    ## Get explanation for samples with correct answer ## if post_hoc
    out_ds = TorchDS(ds_w_ans,mt.tokenizer,model_name,ds_name = args.dataset_name,expl = True,mode = args.mode)
    for sample in tqdm(out_ds,total = len(out_ds), desc = f'getting explanation for post_hoc for {mt.model_name}'):
        sample_ids = sample['sample_id']
        joint_input = [sample['input_ids']]
        if 'cf_input_ids' in sample:
            joint_input.append(sample['cf_input_ids'])
        
        original_expl = None
        for s_id,input_ids in enumerate(joint_input):
            input_ids = input_ids.unsqueeze(0).to(mt.model.device)
            attn_mask = torch.ones_like(input_ids).to(input_ids.device)
            gen_inputs = {'input_ids':input_ids,'attention_mask':attn_mask}
            if s_id == 0:
                decoded_oe = None
                while not decoded_oe:
                    with torch.no_grad():
                        expl_logits = mt.model.generate(**gen_inputs,**gen_kwargs,pad_token_id = mt.tokenizer.eos_token_id)
                    expl_logits = expl_logits[0,input_ids.shape[-1]:] 
                    decoded_oe = mt.tokenizer.decode(expl_logits,skip_special_tokens=True)
                    if not mt.is_chat: # for non-chat models, truncate unwanted seq
                        decoded_oe = clean_nonchat_gen(decoded_oe)
                    decoded_oe = decoded_oe.strip()
                    if decoded_oe == '':
                        decoded_oe = None
                expl_logits = torch.tensor(mt.tokenizer.encode(decoded_oe,add_special_tokens=False),dtype=torch.long).to(mt.model.device)
                original_expl = expl_logits
                out_ds.ds[sample_ids]['explanation'] = decoded_oe
                out_ds.ds[sample_ids]['explanation_prompt'] = sample['prompt']
                out_ds.ds[sample_ids]['high_expl_prob'],out_ds.ds[sample_ids]['high_expl_logit'] = generate_sequence_probs(input_ids,mt,expl_logits)
            else:
                out_ds.ds[sample_ids]['low_expl_prob'],out_ds.ds[sample_ids]['low_expl_logit'] = generate_sequence_probs(input_ids,mt,original_expl)
    out_ds = list(out_ds.ds.values())
    
    if 'low_expl_prob' in out_ds[0]:
        print (f'{args.model_name} Expl Prob diff: {np.mean([np.mean((d["high_expl_prob"]-d["low_expl_prob"])/d["high_expl_prob"]) for d in out_ds]):.2f}')
        print (f'{args.model_name} Expl Logit diff: {np.mean([np.mean(d["high_expl_logit"]-d["low_expl_logit"]) for d in out_ds]):.2f}')

    
    for dict in out_ds:
        for k,v in dict.items():
            if isinstance(v,torch.Tensor) or isinstance(v,np.ndarray):
                if len(v.shape) == 0:
                    dict[k] = v.item()
                elif len(v.shape) == 1:
                    dict[k] = v.tolist()
                else:
                    raise ValueError(f'Unexpected shape for {k}, {v.shape}, should only be 0 or 1.')
    return out_ds



def get_answer_and_explanation_ood(ds,mt,batch_size = 8,model_name = 'llama3',args=None):
    """
    for each original and cf run, get the prob and logit of both the correct and cf_answer to use as high/low score for the two runs.
    ie for original, answer is always A, get prob for A as high score and B as low score for the 2nd cf run.
    for cf, answer is always B, get prob for B as high score for cf and A as low score for original.
    """
    out_ds = []
    ds = TorchDS(ds,mt.tokenizer,model_name,ds_name = args.dataset_name,expl = None,corrupt = False,mode = 'ood') 
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=ds.collate_fn,drop_last = False)
    for batch in tqdm(dl,total = len(dl),desc = f'Getting accuracy and high score for answer, model: {mt.model_name}'):
        num_choices = batch['num_choices']
        sample_id = batch['sample_id']
        result_collection = {sid:ds.ds[sid] for sid in sample_id}
        # print (mt.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=True))
        # exit()
        joint_input_length = [(batch['input_ids'],batch['input_length']),(batch['cf_input_ids'],batch['cf_input_length'])]

        for s_id,(input_ids,input_len) in enumerate(joint_input_length):
            input_ids = input_ids.to(mt.model.device)
            preds = get_pred(mt,input_ids,num_choices,input_len,ans_poss = [['A','B'] for _ in input_ids])
            for p,sid in zip(preds,sample_id):
                if s_id == 0:
                    result_collection[sid]['high_prob'] = p[1][0]
                    result_collection[sid]['low_cf_prob'] = p[1][1]
                    result_collection[sid]['correct'] = p[0].upper() == 'A'
                else: # store the corrupt results
                    result_collection[sid]['low_prob'] = p[1][0]
                    result_collection[sid]['high_cf_prob'] = p[1][1]
                    result_collection[sid]['cf_correct'] = p[0].upper() == 'B'
        ## we only collect cases where the model is correct for both original and cf (as we want to see if the expl technique correctly captures the impt features)
        for k,v in result_collection.items():
            if v['correct'] and v['cf_correct']:
                out_ds.append(v)
    print (f'{args.model_name} success rate: {len(out_ds)/len(ds):.2f}')
    print (f'{args.model_name} Original Prob diff: {np.mean([(d["high_prob"]-d["low_prob"])/d["high_prob"] for d in out_ds]):.2f}')
    print (f'{args.model_name} CF Prob diff: {np.mean([(d["high_cf_prob"]-d["low_cf_prob"])/d["high_cf_prob"] for d in out_ds]):.2f}')

    for dict in out_ds:
        for k,v in dict.items():
            if isinstance(v,torch.Tensor) or isinstance(v,np.ndarray):
                if len(v.shape) == 0:
                    dict[k] = v.item()
                elif len(v.shape) == 1:
                    dict[k] = v.tolist()
                else:
                    raise ValueError(f'Unexpected shape for {k}, {v.shape}, should only be 0 or 1.')
    return out_ds


def get_answer_and_explanation_GN(ds,mt,batch_size = 8,model_name = 'llama3',args=None):
    """
    for each original and cf run, get the prob and logit of both the correct and cf_answer to use as high/low score for the two runs.
    ie for original, answer is always A, get prob for A as high score and B as low score for the 2nd cf run.
    for cf, answer is always B, get prob for B as high score for cf and A as low score for original.
    """
    out_ds = []
    ds = TorchDS(ds,mt.tokenizer,model_name,ds_name = args.dataset_name,expl = None,corrupt = False,mode = 'ood') 
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=ds.collate_fn,drop_last = False)
    for batch in tqdm(dl,total = len(dl),desc = f'Generating OOD for STR, model: {mt.model_name}'):
        num_choices = batch['num_choices']
        sample_id = batch['sample_id']
        result_collection = {sid:ds.ds[sid] for sid in sample_id}
        input_ids = batch['input_ids']
        input_len = batch['input_length']

        joint_input_length = [(batch['input_ids'],batch['input_length']),(batch['cf_input_ids'],batch['cf_input_length'])]

        for s_id,(input_ids,input_len) in enumerate(joint_input_length):
            input_ids = input_ids.to(mt.model.device)
            preds = get_pred(mt,input_ids,num_choices,input_len,ans_poss = [['A','B'] for _ in input_ids])
            for p,sid in zip(preds,sample_id):
                if s_id == 0:
                    result_collection[sid]['high_prob'] = p[1][0]
                    result_collection[sid]['low_cf_prob'] = p[1][1]
                    result_collection[sid]['high_logit'] = p[2][0]
                    result_collection[sid]['low_cf_logit'] = p[2][1]
                    result_collection[sid]['correct'] = p[0].upper() == 'A'
                elif s_id == 1: # store the corrupt results
                    result_collection[sid]['low_prob'] = p[1][0]
                    result_collection[sid]['high_cf_prob'] = p[1][1]
                    result_collection[sid]['low_logit'] = p[2][0]
                    result_collection[sid]['high_cf_logit'] = p[2][1]
        
        for k,v in result_collection.items():
            if v['correct']: # append only the correct ones
                out_ds.append(v)

    # see how much the other class prob is affected when STR is performed
    print (f'{args.model_name} correct: {len(out_ds)/len(ds):.2f}')
    print (f'{args.model_name} STR prob diff of original: {np.mean([(d["high_prob"]-d["low_prob"])/d["high_prob"] for d in out_ds]):.2f}') 
    print (f'{args.model_name} STR logit diff of original: {np.mean([(d["high_logit"]-d["low_logit"]) for d in out_ds]):.2f}') 
    print (f'{args.model_name} STR prob diff of cf: {np.mean([(d["high_cf_prob"]-d["low_cf_prob"])/d["high_cf_prob"] for d in out_ds]):.2f}') 
    print (f'{args.model_name} STR logit diff of cf: {np.mean([(d["high_cf_logit"]-d["low_cf_logit"]) for d in out_ds]):.2f}') 


    ## Add GN noise here.
    noise_level = 3.0 * collect_embedding_std(mt,[d['subject'] for d in out_ds]) # set to 3x std of embeddings
    GN_ds = TorchDS(out_ds,mt.tokenizer,model_name,ds_name = args.dataset_name,expl = None,corrupt = True,mode = 'ood') 
    out_ds = []
    for sample in tqdm(GN_ds,total = len(GN_ds), desc = f'getting GN ood for {mt.model_name}'):
        input_ids = sample['input_ids']
        subject = sample['subject']
        sample_id = sample['sample_id']
        curr_sample = GN_ds.ds[sample_id]
        corrupt_range = find_token_range(mt.tokenizer, input_ids,subject,include_chat_template=mt.is_chat)
        if corrupt_range[0] is None:
            continue
        corrupt_shape = corrupt_range[1] - corrupt_range[0]
        noise = np.random.randn(corrupt_shape,mt.model.config.hidden_size) * noise_level
        
        low_gn_probs,low_gn_logits = [],[]
        for answer in ['A','B']:
            answer_t = tokenize_single_answer(answer,mt.tokenizer,mt.model_name)
            low_prob,low_logit,_ = trace_with_patch(
                    mt.model, input_ids.repeat(5+1,1).to(mt.model.device), [], answer_t, corrupt_range, noise=noise, uniform_noise=False,past_kv = None ,num_samples = 5
                )
            low_gn_probs.append(low_prob.item())
            low_gn_logits.append(low_logit.item())
        curr_sample['low_gn_probs'] = low_gn_probs[0]
        curr_sample['low_gn_logits'] = low_gn_logits[0]
        curr_sample['high_gn_cf_prob'] = low_gn_probs[1] # this is implied as high as we assume that adding noise affects the logit of the ori pred and increase the prob of the other class.
        curr_sample['high_gn_cf_logit'] = low_gn_logits[1]
        out_ds.append(curr_sample)

    # print (len([d for d in out_ds if 'low_gn_probs' not in out_ds]))
    # exit()
    print (f'{args.model_name} GN prob diff of original: {np.mean([(d["high_prob"]-d["low_gn_probs"])/d["high_prob"] for d in out_ds]):.2f}') 
    print (f'{args.model_name} GN logit diff of original: {np.mean([(d["high_logit"]-d["low_gn_logits"]) for d in out_ds]):.2f}') 
    print (f'{args.model_name} GN prob diff of cf: {np.mean([(d["high_gn_cf_prob"]-d["low_cf_prob"])/d["high_gn_cf_prob"] for d in out_ds]):.2f}') 
    print (f'{args.model_name} GN logit diff of cf: {np.mean([(d["high_gn_cf_logit"]-d["low_cf_logit"]) for d in out_ds]):.2f}')
    
    for dict in out_ds:
        for k,v in dict.items():
            if isinstance(v,torch.Tensor) or isinstance(v,np.ndarray):
                if len(v.shape) == 0:
                    dict[k] = v.item()
                elif len(v.shape) == 1:
                    dict[k] = v.tolist()
                else:
                    raise ValueError(f'Unexpected shape for {k}, {v.shape}, should only be 0 or 1.')
    return out_ds
            
