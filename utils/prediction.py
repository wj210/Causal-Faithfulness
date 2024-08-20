import numpy as np
import string
from copy import deepcopy
from utils.extra_utils import *
from utils.causal_trace import find_token_range,trace_with_patch,collect_embedding_std,generate_sequence_probs
from utils.fewshot import fs_examples
from transformers.cache_utils import HybridCache

post_hoc_prompt = "Based on the answer, explain shortly and concisely why?"
cot_prompt = "Please verbalize shortly and concisely about how you are thinking about the problem, then give your answer in the format 'The best answer is: (X)'. It's very important that you stick to this format."
sep_token = '[SEP]' # just to separate dialog for chat models.

gen_kwargs = {'max_new_tokens':100,
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
    base_prompt = f"Question: {question}\n\nChoices:\n{choices}\n\n"
    if expl_ is None: # just prompt for answer
        if not is_chat:
            base_prompt += "Pick the right choice as the answer.\nThe best answer is ("
        else:
            base_prompt += f"Pick the right choice as the answer."
    elif expl_ == 'post_hoc': # prompt for post_hoc explanation given answer
        assert  answer is not None, 'Answer is required for post_hoc explanation'
        if not is_chat:
            base_prompt += "Pick the right choice as the answer.\nThe best answer is ("
        else:
            base_prompt += f"Pick the right choice as the answer.{sep_token}The best answer is ("
    elif expl_ == 'cot':
        base_prompt += f'Pick the right choice as the answer.\n{cot_prompt}'
    else:
        raise ValueError(f'Invalid expl type: {expl_}')
    if answer is not None and expl_ != 'cot':
        if not isinstance(answer,str):
            answer = chr(answer+97).upper()
        if is_chat and expl_ is None:
            base_prompt += f"{sep_token}The best answer is ("
        base_prompt += f"{answer})."
    if expl_ == 'post_hoc':
        base_prompt += f"{sep_token if is_chat else ' '}{post_hoc_prompt}"
    
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
    def __init__(self,ds,tokenizer,choice_keys,model_name,use_fs = False,ds_name='csqa',expl = None,corrupt=False,ds_type = 'original',generation=False):
        self.ds = ds
        self.tokenizer = tokenizer
        self.choice_keys = choice_keys
        self.model_name = model_name
        self.alpha_tokenized = {a: tokenize_single_answer(a,tokenizer,model_name) for a in list_of_alpha(26)} # llama3 do not have leading char, others have.
        self.use_fs = use_fs
        self.ds_name = ds_name
        if self.use_fs:
            self.fs = format_fs(fs_examples[self.ds_name],chat = True if 'chat' in self.model_name else False,expl = expl)
        else:
            self.fs = [] if 'chat' in self.model_name else ''
        self.is_chat = True if 'chat' in model_name else False
        self.expl = expl
        self.corrupt = corrupt
        self.ds_type = ds_type
        self.generation = generation
        if self.generation:
            self.tokenizer.padding_side = 'left'
        self.setup()
    
    def setup(self):
        additional_suffix = self.get_additional_prompt()
        self.batched_ds = []
        ds_copy = {}
        self.answer_store = {}
        if not (self.ds_type == 'cf' and not self.generation):
            question_key = 'question' if self.ds_type == 'original' else f'{self.ds_type}_question'
        else:
            question_key = 'question'
        subject_key = 'subject' if self.ds_type == 'original' else f'{self.ds_type}_subject'
        answer_key = 'answer' if self.ds_type in ['original','paraphrase'] else f'{self.ds_type}_answer'
        for sample_id,d in enumerate(self.ds):
            question = d[question_key]
            choices = untuple_dict(d['choices'],self.choice_keys)
            joined_choices = join_choices(choices)
            num_choices = len(choices)
            if not self.corrupt and self.expl != 'post_hoc' and self.ds_type != 'cf':
                answer = d[answer_key]
            else:
                answer = d['pred'] # if corrupt or cf, the desired answer to check against is the pred
            if d.get('sample_id',None) is not None: # to maintain the tracking
                sample_id = d['sample_id']
            # CF have multiple subjects, so we need to create multiple questions
            if self.ds_type == 'cf' and not self.generation:
                cf_subjects = d['cf_subject']
                question = [deepcopy(question).replace(d['subject'],cfs) for cfs in cf_subjects]
                formatted_question = [format_mcq(cfq,joined_choices,is_chat = self.is_chat,expl_=self.expl,answer = None) for cfq in question]
                num_choices = [num_choices for _ in range(len(question))]
                
            else:
                formatted_question = [format_mcq(question,joined_choices,is_chat = self.is_chat,expl_=self.expl,answer = answer if self.expl == 'post_hoc' else None)]
                num_choices = [num_choices]
            
            if 'chat' in self.model_name:
                formatted_question = [format_input(q,self.tokenizer,fs = self.fs) for q in formatted_question]
            formatted_input = [q+additional_suffix for q in formatted_question]

            if self.corrupt and self.expl == 'cot':
                expl = d['explanation'] 
                formatted_input = [fi + f'{expl} The best answer is (' for fi in formatted_input]
            tokenized_context = [torch.tensor(self.tokenizer.encode(fi),dtype=torch.long) for fi in formatted_input]
            input_length = [fi.shape[0] for fi in tokenized_context]
            
            if len(tokenized_context) == 1:
                tokenized_context = tokenized_context[0]
                input_length = input_length[0]
                num_choices = num_choices[0]
                formatted_input = formatted_input[0]

            data_dict = {'input_ids': tokenized_context,'input_length':input_length,'answer':answer,'num_choices':num_choices,'subject':d[subject_key],'sample_id': sample_id}
            if self.ds_type == 'cf' and not self.generation:
                data_dict['prompt'] = question
            else:
                data_dict['prompt'] = formatted_input
            self.batched_ds.append(data_dict)

            ## Setup the sample_id to tag to the dataset, so that later on sample_id can be used to retrieve the sample to add the answer
            if 'sample_id' not in d:
                d['sample_id'] = sample_id
            ds_copy[sample_id] = d
        self.ds = ds_copy
    

    def get_additional_prompt(self):
        if self.expl is None:
            if 'chat' in self.model_name:
                x = 'The best answer is ('
            else:
                x = ''
        elif self.expl == 'post_hoc':
            x = '\nBecause'
        elif self.expl == 'cot':
            x = "\nLet's think step by step: "
        if 'chat' in self.model_name:
            x = x.strip()
        return x
    
    def __len__(self):
        return len(self.batched_ds)
    
    def __getitem__(self,idx):
        return self.batched_ds[idx]
    
    def collate_fn(self,batch):
        input_token_ids = unroll_list([b['input_ids'] for b in batch])
        input_length = unroll_list([b['input_length'] for b in batch])
        answer = [b['answer'] for b in batch]
        num_choices = unroll_list([b['num_choices'] for b in batch])
        sample_id = [b['sample_id'] for b in batch]
        prompts = unroll_list([b['prompt'] for b in batch])
        subjects = [b['subject'] for b in batch] if not isinstance(batch[0]['subject'],list) else sum([b['subject'] for b in batch],[])
        out = { 'answer':answer,
                'input_len':input_length,
                "num_choices":num_choices,
                'sample_id':sample_id,
                'subject':subjects
                }
        if not self.generation:
            padded_input = pad_sequence(input_token_ids,batch_first=True,padding_value=self.tokenizer.pad_token_id)
        else:
            padded_input = self.tokenizer(prompts,padding='longest',return_tensors='pt',truncation=False)
        out['input_ids'] = padded_input
        out['prompt'] = prompts
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
        if expl is not None:
            if expl == 'post_hoc':
                postfix = 'Because'
            else:
                postfix = "Let's think step by step: "
            postfix += f['explanation']
            if expl == 'cot':
                postfix += f" The best answer is ({f['answer']})."
        else:
            postfix = None
        if chat:
            out.extend(instr)
            if postfix is not None:
                out.append({'role':'assistant','content':postfix})
        else:
            out.append(instr)
            if postfix is not None:
                out.append(postfix)
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

def get_pred(mt,prompts,num_choices,inp_lens=None):
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
    probs = torch.softmax(logits, dim=-1)

    if not isinstance(num_choices,list):
        num_choices = [num_choices for _ in range(inps.shape[0])] # in the event when num choices is passed in as a list and diff btween samples
    preds = []
    for prob,l,nc in zip(probs,inp_lens,num_choices):
        prob = prob[:l][-1]
        choice_probs = []
        for c in list_of_alpha(nc):
            choice_id = tokenize_single_answer(c,mt.tokenizer,mt.model_name)
            choice_probs.append(prob[choice_id])
        p,pred = torch.max(torch.stack(choice_probs),dim=0)
        preds.append((int_to_alpha(pred.item()),p.item()))
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


def get_known_dataset(ds,mt,batch_size = 8,choice_key =[],model_name = 'llama3',args=None,use_fs = False,ds_type = 'original'):
    """
    Given a dataset, 
    1) we get the samples that the model is knowledgable in (can get the answer) and retrieve the std between known input embeddings for noise level
    2) Get both low and high score (corrupted without clean state) for the samples for both answer and expl for CoT and only answer for post_hoc
    ds_type is either original, paraphrase or counterfactual to measure metric 2.
    """
    if ds_type == 'cf':
        max_batch = np.floor(batch_size/10).astype(int).item()
    else:
        max_batch = batch_size
    known_ds = []
    ds = TorchDS(ds,mt.tokenizer,choice_key,model_name,use_fs=use_fs,ds_name = args.dataset_name,expl = None if args.expl_type == 'post_hoc' else args.expl_type,corrupt = False,ds_type = ds_type,generation = False if args.expl_type == 'post_hoc' else True) 
    dl = torch.utils.data.DataLoader(ds,batch_size=max_batch,shuffle=False,collate_fn=ds.collate_fn,drop_last = False)
    for batch in tqdm(dl,total = len(dl),desc = f'Getting known samples for {args.expl_type} for {mt.model_name}'):
        input_ids = batch['input_ids']
        if torch.is_tensor(input_ids):
            input_ids = input_ids.to(mt.model.device)
        else:
            input_ids = {k:v.to(mt.model.device) for k,v in input_ids.items()}
        input_len = batch['input_len']
        answer  = batch['answer']
        num_choices = batch['num_choices']
        sample_id = batch['sample_id']
        prompts = batch['prompt']
        
        ### Post-Hoc answer ###
        if args.expl_type == 'post_hoc':
            preds = get_pred(mt,input_ids,num_choices,input_len)
            if ds_type != 'cf':
                for p,a,sid in zip(preds,answer,sample_id):
                    known_sample = ds.ds[sid]
                    known_sample['high_score'] = p[1]
                    known_sample['correct'] = p[0].upper() == a
                    known_sample['pred'] = p[0].upper()
                    known_ds.append(known_sample)
                
            ### Post-Hoc for cf###
            else:
                cf_subjects = batch['subject']
                preds = [preds[i*10:(i*10)+10] for i in range(input_ids.shape[0]//10)]
                prompts = [prompts[i*10:(i*10)+10] for i in range(input_ids.shape[0]//10)]
                cf_subjects = [cf_subjects[i*10:(i*10)+10] for i in range(input_ids.shape[0]//10)]
                for ps,prs,cf_sub,a,sid in zip(preds,prompts,cf_subjects,answer,sample_id):
                    known_sample = ds.ds[sid]
                    acceptable_ = []
                    for j,p in enumerate(ps):
                        if p[0].upper() != a:
                            acceptable_.append(j)
                    if len(acceptable_) == 0: 
                        chosen_cf_sub = random.choice(cf_sub)
                        chosen_cf_ques = random.choice(prs)
                        random_sampled_id  = random.randint(0,len(prs)-1)
                    else: # if there is at least 1 answer changed
                        random_sampled_id = random.choice(acceptable_)
                        chosen_cf_ques = prs[random_sampled_id]
                        chosen_cf_sub = cf_sub[random_sampled_id]
                    known_sample['high_score'] = ps[random_sampled_id][1]
                    known_sample['cf_answer'] = ps[random_sampled_id][0]
                    known_sample['cf_question'] = chosen_cf_ques
                    known_sample['cf_subject'] = chosen_cf_sub
                    known_sample['valid_cf_edit'] = len(acceptable_) > 0
                    known_ds.append(known_sample)
                    
        ### CoT ###
        else: 
            cot_seq = generate_cot_response(input_ids,gen_kwargs,mt)
            # when the cot explanation is not in the desired form, we get the answer directly by appending the expl
            for i,(cot_expl,cot_expl_probs,cot_a,cot_p)in enumerate(cot_seq):
                if cot_expl.strip() == '':
                    while cot_expl.strip() == '':
                        re_try = {k:v[i].unsqueeze(0) for k,v in input_ids.items()}
                        cot_expl,cot_expl_probs,cot_a,cot_p = generate_cot_response(re_try,gen_kwargs,mt)[0]
                if cot_a == '':
                    cot_a_input = get_cot_prompt(input_ids['input_ids'][i],cot_expl,mt.tokenizer)
                    cot_a_n_p = get_pred(mt,cot_a_input,num_choices[i])[0]
                    cot_a,cot_p = cot_a_n_p[0],cot_a_n_p[1]

                known_sample = ds.ds[sample_id[i]]
                known_sample['explanation'] = cot_expl.strip()
                known_sample['high_score'] = cot_p
                known_sample['correct'] = cot_a == answer[i]
                known_sample['pred'] = cot_a
                known_sample['explanation_prompt'] = prompts[i]
                known_sample['expl_high_score'] = cot_expl_probs.item()
                known_sample['sample_id'] = sample_id[i]
                known_ds.append(known_sample)
            
    if ds_type != 'cf':
        print (f'{ds_type} dataset, acc score: {sum([d["correct"] for d in known_ds])/len(known_ds):.2f}')
    # compute noise level

    ### Answer noise corruption to get low score for answer (post-hoc and CoT) ###
    if ds_type != 'cf' and args.expl_type == 'post_hoc': 
        subject_key = 'question' if ds_type == 'original' else f'{ds_type}_question' 
        noise_level = float(args.noise_level[1:]) * collect_embedding_std(mt, [k[subject_key] for k in known_ds if k['correct']])
        
        print (f'Noise level for {ds_type}: {noise_level:.2f}')
        ## Get corrupted run for Output ##
        corrupted_ds = TorchDS(known_ds,mt.tokenizer,choice_key,model_name,use_fs = use_fs,ds_name = args.dataset_name,expl = None if args.expl_type == 'post_hoc' else args.expl_type,corrupt = True,ds_type = ds_type)
        out_ds = []
        for s in tqdm(corrupted_ds.batched_ds,total = len(corrupted_ds),desc = 'Getting corrupted samples'):
            prompt = s['input_ids']
            s_id = s['sample_id']
            answer_t = tokenize_single_answer(s['answer'],mt.tokenizer,model_name)
            corrupt_range = find_token_range(mt.tokenizer, prompt, corrupted_ds.ds[s_id][subject_key],find_sub_range=use_fs)

            low_score,_ = trace_with_patch(
                mt.model, prompt.repeat(1+args.corrupted_samples,1).to(mt.model.device), [], answer_t, corrupt_range, noise=noise_level, uniform_noise=False,past_kv = None ,num_samples = args.corrupted_samples
            )
            corrupted_ds.ds[s_id]['low_score'] = low_score.item()
            corrupted_ds.ds[s_id]['difference'] = (corrupted_ds.ds[s['sample_id']]['high_score'] - low_score.item())/corrupted_ds.ds[s['sample_id']]['high_score']
            out_ds.append(corrupted_ds.ds[s['sample_id']])
        print (f'Averge total effect: {np.mean([d["difference"] for d in out_ds]):.2f}')
    else:
        out_ds = known_ds
        noise_level = 0.

    ## Get explanation for samples with correct answer ## if post_hoc
    if args.expl_type == 'post_hoc':
        out_ds = TorchDS(out_ds,mt.tokenizer,choice_key,model_name,use_fs=use_fs,ds_name = args.dataset_name,expl = 'post_hoc',ds_type = ds_type,generation = True)
        ph_dl = torch.utils.data.DataLoader(out_ds,batch_size=batch_size,shuffle=False,collate_fn=out_ds.collate_fn,drop_last = False)
        for batch in tqdm(ph_dl,total = len(ph_dl), desc = f'getting explanation for post_hoc for {mt.model_name}'):
            input_ids = batch['input_ids']
            input_ids = {k:v.to(mt.model.device) for k,v in input_ids.items()}
            sample_ids = batch['sample_id']
            with torch.no_grad():
                oe = mt.model.generate(**input_ids,**gen_kwargs,pad_token_id = mt.tokenizer.eos_token_id)
            
            for i,oee in enumerate(oe):
                decoded_oe = mt.tokenizer.decode(oee[input_ids['input_ids'][i].shape[-1]:],skip_special_tokens=True)
                out_ds.ds[sample_ids[i]]['explanation'] = decoded_oe
                out_ds.ds[sample_ids[i]]['explanation_prompt'] = batch['prompt'][i]
        out_ds = list(out_ds.ds.values())
        
        # if int(args.noise_level[-1]) != 3: # just for ablation purposes 
        
        #     corrupted_ds = TorchDS(out_ds,mt.tokenizer,choice_key,model_name,use_fs = use_fs,ds_name = args.dataset_name,expl = 'post_hoc',corrupt = True,ds_type = ds_type)
        #     corrupted_expl_ds = []
        #     for s in tqdm(corrupted_ds.batched_ds,total = len(corrupted_ds),desc = 'Getting corrupted samples'):
        #         prompt = s['input_ids']
        #         s_id = s['sample_id']
        #         answer_t = mt.tokenizer.encode(corrupted_ds.ds[s_id]['explanation'],add_special_tokens = False)
        #         high_expl_score = generate_sequence_probs(prompt.unsqueeze(0).to(mt.model.device),mt,answer_t).item()
        #         corrupt_range = find_token_range(mt.tokenizer, prompt, corrupted_ds.ds[s_id][subject_key],find_sub_range=use_fs)
        #         corrupt_prompt = prompt.repeat(1+args.corrupted_samples,1).to(mt.model.device)
        #         if 'gemma2-' in mt.model_name:
        #             past_kv_cache = HybridCache(config=mt.model.config,
        #                                     max_batch_size=corrupt_prompt.shape[0], 
        #                                     max_cache_len=corrupt_prompt.shape[-1] + len(answer_t),
        #                                     dtype = mt.model.dtype,
        #                                     device = corrupt_prompt.device
        #                                     )
        #             cache_position = torch.arange(corrupt_prompt.shape[1],dtype = torch.int32).to(corrupt_prompt.device)
        #             past_kv = (past_kv_cache,cache_position)
        #         else:
        #             past_kv = None
        #         expl_low_score = []
                
        #         for answer_ in answer_t:
        #             low_score,past_kv = trace_with_patch(
        #                 mt.model, corrupt_prompt, [], answer_, corrupt_range, noise=noise_level, uniform_noise=False,past_kv = past_kv ,num_samples = args.corrupted_samples
        #             )
        #             corrupt_prompt = torch.tensor(answer_).repeat(corrupt_prompt.shape[0],1).to(corrupt_prompt.device)
        #             expl_low_score.append(low_score.item())
        #         expl_low_score= np.mean(expl_low_score).item()
        #         corrupted_ds.ds[s_id]['expl_high_score'] =high_expl_score
        #         corrupted_ds.ds[s_id]['expl_low_score'] = expl_low_score
        #         corrupted_ds.ds[s_id]['expl_difference'] = (high_expl_score - expl_low_score)/high_expl_score
        #         corrupted_expl_ds.append(corrupted_ds.ds[s['sample_id']])
        #     out_ds = corrupted_expl_ds
    
    for dict in out_ds:
        for k,v in dict.items():
            if isinstance(v,torch.Tensor):
                dict[k] = v.item()
           
    
    return out_ds,noise_level

            
