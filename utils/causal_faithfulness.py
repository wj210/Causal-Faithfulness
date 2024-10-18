from utils.extra_utils import *
from utils.prediction import *
from utils.causal_trace import *
from time import time


def trace_with_patch_STR(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    past_kv = None,
):
    """
    Runs a single causal trace, 1st input is clean, 2nd is corrupted/CF.
    ** NEW **
    This function is modified to do batch processing, where the states at different layer/tok pos is batched together (Not different samples).
    Thus num_samples (the number of corrupted duplicated samples) is used to gauge where's the next sample.
    """
    nb = inp.shape[0]//2 # batch size (how many forward runs)

    # patch_spec = defaultdict(list)
    patch_spec = []
    for t,l in states_to_patch: 
        pc = defaultdict(list)
        if not isinstance(l,list):
            pc[l].append(t)
        else:
            for ll in l:
                pc[ll].append(t)
        
        patch_spec.append(pc)
    if states_to_patch != []:
        assert len(patch_spec) == nb, f"number of patching states should be equal to number of corrupted runs., {len(patch_spec)} != {nb}"
    else:
        patch_spec = [defaultdict(list)]
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    if 'gemma-2-' in model.name_or_path: # gemma2 requires the cache position to be passed
        if past_kv is not None:
            past_kv_cache = past_kv[0]
            cache_position = past_kv[1]
            if cache_position.shape[0] > 1:
                past_kv = None
            else:
                past_kv = past_kv_cache
        else:
            past_kv_cache = None

    def patch_rep(x, layer):
        if past_kv is not None: # if answer seq is more than 1, we do not need to patch since we are just continuing the run.
            return x
        if all([layer not in pc for pc in patch_spec]):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for i,pc in enumerate(patch_spec):
            for t in pc[layer]: # only when layer is in pc
                clean_pos = i*2
                h[clean_pos+1, t] = h[clean_pos, t] # 0 is always the clean one
        return x

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + sum([list(pc.keys()) for pc in patch_spec],[]),
        edit_output=patch_rep,
    ) as td:
        if 'gemma-2-' in model.name_or_path:
            if past_kv_cache is not None:
                outputs_exp = model(inp,use_cache = True,past_key_values = past_kv_cache,cache_position=cache_position)
                next_cache_position = cache_position[-1:] + 1
                out_kv = (past_kv_cache,next_cache_position)
            else:
                outputs_exp = model(inp)
                out_kv = None # if answer length is only 1 no need to cache
        else:
            if past_kv is not None:
                outputs_exp = model(inp,use_cache = True,past_key_values = past_kv)
            else:
                outputs_exp = model(inp,use_cache = True)
            out_kv = outputs_exp.past_key_values

    # We report softmax probabilities and logit for the answers_t token predictions of interest. # average across all corrupted runs 
    probs = []
    logits = []
    for j in range(1,outputs_exp.logits.shape[0],2): # corrup starts from 1
        curr_sample_logits = outputs_exp.logits[j, -1, :]
        probs.append(torch.softmax(curr_sample_logits, dim=-1)[answers_t])
        logits.append(curr_sample_logits[answers_t])
    probs = torch.stack(probs)
    logits = torch.stack(logits)

    return probs,logits, out_kv

def trace_important_states_STR(
    model,
    num_layers,
    inp,
    answer_t,
    token_range=None, # ranges to patch list of size 2
    batch_size = 32,
    use_kv_caching = True,
    window = 1,
):
    table_prob = []
    table_logit = []
    if isinstance(num_layers,int):
        layer_range = range(num_layers)
    else:
        layer_range = deepcopy(num_layers)
        num_layers = len(layer_range)
        
    pos_to_edit = []
    for tnum in token_range:
        for layer in layer_range:
            if window == 1:
                pos_to_edit.append((tnum, layername(model, layer)))
            else: # patch in multiple layers
                window_layers = [layername(model,L) for L in range(max(0, layer - window // 2), min(num_layers, layer - (-window // 2)))]
                pos_to_edit.append((tnum, window_layers))

    for i in range(0,len(pos_to_edit),batch_size):
        take_size = min(batch_size,len(pos_to_edit) - i)
        inp_rolled = inp.repeat(take_size,1)
        batched_prob,batched_logit  = [],[]
        if use_kv_caching:
            if 'gemma-2-' in model.name_or_path:
                past_kv_cache = HybridCache(config=model.config,
                                        batch_size=inp_rolled.shape[0], 
                                        max_cache_len=inp_rolled.shape[-1] + len(answer_t),
                                        dtype = model.dtype,
                                        device = model.device
                                        )
                cache_position = torch.arange(inp_rolled.shape[1],dtype = torch.int32).to(inp_rolled.device)
                past_kv = (past_kv_cache,cache_position)
            else:
                past_kv = None
        else:
            past_kv = None
        for answer in answer_t:
            p,l,past_kv = trace_with_patch_STR(
                            model,
                            inp_rolled,
                            pos_to_edit[i:i+batch_size],
                            answer,
                            past_kv = past_kv,
                        )
            batched_prob.append(p)
            batched_logit.append(l)
            if use_kv_caching:
                inp_rolled = torch.tensor(answer).repeat(inp_rolled.shape[0],1).to(inp_rolled.device)
            else:
                inp_rolled = add_column_with_int(inp_rolled,answer)
                past_kv = None

        table_prob.append(torch.stack(batched_prob).T)
        table_logit.append(torch.stack(batched_logit).T)
        if 'gemma-2-' in model.name_or_path and past_kv is not None:
            past_kv[0].reset()
            torch.cuda.empty_cache()  # Optionally clear CUDA cache if using GPU
    
    table_prob = torch.cat(table_prob,dim = 0)
    table_logit = torch.cat(table_logit,dim = 0)
    table_prob = table_prob.view(len(token_range),num_layers,-1)
    table_logit = table_logit.view(len(token_range),num_layers,-1)
    return table_prob,table_logit

def calculate_STR_causal(
    mt,
    prompt,
    subject,
    answer = None, # if answer provided, no need to predict again,
    input_until = None,
    batch_size = 32,
    scores = {},
    window = 1,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    causal_token_range contains list of toks to patch
    """
    if input_until is None:
        print ('Warning: Ensure the input length for both clean and cf is the same if end position of tracing is not provided.')
        input_until = len(prompt[0])
    original_range = find_token_range(mt.tokenizer, prompt[0], subject[0],include_chat_template = mt.is_chat,find_sub_range = not mt.is_chat) # list of list, outer = number of words, inner = token pos
    cf_range = find_token_range(mt.tokenizer, prompt[1], subject[1],include_chat_template = mt.is_chat,find_sub_range = not mt.is_chat)

    all_t_range = [] # for patching
    
    for tok_range_n_prompt in [(original_range,prompt[0]),(cf_range,prompt[1])]:
        tok_range,p_prompt = tok_range_n_prompt
        if isinstance(input_until,str):
            end_pos = find_token_range(mt.tokenizer, p_prompt, input_until,include_chat_template = mt.is_chat,find_sub_range = not mt.is_chat)[0]
        else:
            end_pos = tok_range[0][0] + input_until
        tok_range = range(tok_range[0],end_pos)
        all_t_range.append(tok_range)

    assert all_t_range[0] == all_t_range[1], "both original and cf causal token range must have the same number of tokens"

    if len(answer) == 1: # for explanation
        answer_t = tokenize_single_answer(answer,mt.tokenizer,mt.model_name)
    else:
        answer_t = mt.tokenizer.encode(answer,add_special_tokens=False)
    if not isinstance(answer_t,list):
        answer_t = [answer_t]

    ## For logging ##
    original_input_tokens = [decode_tokens(mt.tokenizer, prompt[0])[j] for j in all_t_range[0]]
    cf_input_tokens = [decode_tokens(mt.tokenizer, prompt[1])[j] for j in all_t_range[1]]
    
    ## high and low score ##
    low_prob,low_logit = scores['low_prob'],scores['low_logit']
    high_prob,high_logit = scores['high_prob'],scores['high_logit']

    ## Patch input is concatenated (clean and cf) need to take note of difference in length.
    patch_input = torch.stack(prompt).to(mt.model.device)
    ## Patching ##
    patched_prob,patched_logit = trace_important_states_STR( # input,mask and token range are size of 2.
        mt.model,
        mt.num_layers,
        patch_input,
        answer_t,
        token_range=all_t_range[0], # since both is the same just pass in one
        batch_size = batch_size,
        use_kv_caching = True if len(answer_t) > 1 else False,
        window = window,
        )
    diff_prob = patched_prob.detach().cpu() - low_prob
    diff_logit = patched_logit.detach().cpu() - low_logit

    normalized_diff_prob = (diff_prob / (high_prob - low_prob)).mean(-1)
    normalized_diff_logit = (diff_logit / (high_logit - low_logit)).mean(-1)

    diff_prob = diff_prob.mean(-1)
    diff_logit= diff_logit.mean(-1)

    return dict(
        diff_prob=diff_prob,
        diff_logit=diff_logit,
        normalized_diff_prob=normalized_diff_prob,
        normalized_diff_logit=normalized_diff_logit,
        low_prob=low_prob,
        low_logit=low_logit,
        high_prob=high_prob,
        high_logit=high_logit,
        input_ids=prompt,
        input_tokens=[original_input_tokens,cf_input_tokens],
        subject_range=[original_range,cf_range],
        answer=answer)

def compute_causal_values_STR(ds,mt,args):
    store = {}
    ds = TorchDS(ds,mt.tokenizer,args.model_name,ds_name = args.dataset_name,expl =False,corrupt = True,mode = 'STR')
    starting_t = time()
    for sample in tqdm(ds,total = len(ds),desc = f'Getting attributions for {args.dataset_name}, {args.model_name}'):
        curr_store = []
        sample_id = sample['sample_id']
        curr_sample = ds.ds[sample_id]
        subject = sample['subject']
        cf_subject = curr_sample['cf_subject']
        input_until = "\n\nAnswer: "
        if curr_sample['explanation'] == "":
            continue
        for gen_type in ['answer','expl']:
            if gen_type == 'answer':
                prompts = [sample['input_ids'],sample['cf_input_ids']]
                answer = sample['answer']
                retrieved_scores = {k:curr_sample[k] for k in ['low_prob','low_logit','high_prob','high_logit']}
            else:
                cf_question = curr_sample['cf_question']
                ori_ques = curr_sample['question']
                cf_prompt = curr_sample['explanation_prompt'].replace(ori_ques,cf_question)
                prompts = [torch.tensor(mt.tokenizer.encode(p),dtype=torch.long) for p in [curr_sample['explanation_prompt'],cf_prompt]]
                answer = curr_sample['explanation']
                retrieved_scores = {k.replace('_expl',''):curr_sample[k] for k in ['low_expl_prob','low_expl_logit','high_expl_prob','high_expl_logit']}

            retrieved_scores = {k:torch.tensor(v,dtype = torch.float32) for k,v in retrieved_scores.items()}
            try:
                result = calculate_STR_causal(
                            mt,
                            prompts,
                            [subject,cf_subject],
                            answer=answer,
                            input_until = input_until,
                            batch_size = args.batch_size,
                            scores = retrieved_scores,
                            window = args.window,
                        )
            except Exception as e:
                print (f'Error in sample {sample_id}, {gen_type}')
                continue
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            curr_store.append(numpy_result)

        if len(curr_store) == 2:
            store[sample_id] = curr_store
        
    total_time_taken = time() - starting_t
    print (f'Total time taken for {args.model_name}: {total_time_taken/3600:.2f}hr, per sample: {total_time_taken/len(ds):.3f}s')
    return store


def compute_causal_values_STR_ood(ds,mt,args):
    store = {}
    ds = TorchDS(ds,mt.tokenizer,args.model_name,ds_name = args.dataset_name,expl =False,corrupt = True,mode = 'ood')
    starting_t = time()
    for sample in tqdm(ds,total = len(ds),desc = f'Getting attributions for {args.dataset_name}, {args.model_name}'):
        curr_store = []
        sample_id = sample['sample_id']
        curr_sample = ds.ds[sample_id]
        input_until = "\n\nAnswer: "
        
        for gen_type in ['original','cf']:
            if gen_type == 'original':
                subject = curr_sample['subject']
                cf_subject = curr_sample['cf_subject']
                prompts = [sample['input_ids'],sample['cf_input_ids']]
                answer = curr_sample['answer']
                retrieved_scores = {k:curr_sample[k] for k in ['low_prob','high_prob']}
            else:
                subject = curr_sample['cf_subject']
                cf_subject = curr_sample['subject']
                prompts = [sample['cf_input_ids'],sample['input_ids']]
                answer = curr_sample['cf_answer']
                retrieved_scores = {k.replace('_cf',''):curr_sample[k] for k in ['low_cf_prob','high_cf_prob']}
            
            retrieved_scores['high_logit'] = 1.0 # arbitrary
            retrieved_scores['low_logit'] = 0.0

            retrieved_scores = {k:torch.tensor(v,dtype = torch.float32) for k,v in retrieved_scores.items()}
            try:
                result = calculate_STR_causal(
                            mt,
                            prompts,
                            [subject,cf_subject],
                            answer=answer,
                            input_until = input_until,
                            batch_size = args.batch_size,
                            scores = retrieved_scores,
                            window = args.window,
                        )
            except Exception as e:
                print (f'Error in sample {sample_id}, {gen_type}')
                continue
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            curr_store.append(numpy_result)

        if len(curr_store) == 2:
            store[sample_id] = curr_store
        
    total_time_taken = time() - starting_t
    print (f'Total time taken for {args.model_name}: {total_time_taken/3600:.2f}hr, per sample: {total_time_taken/len(ds):.3f}s')
    return store