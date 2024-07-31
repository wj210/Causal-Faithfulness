import re
from collections import defaultdict
from copy import deepcopy
import numpy
import torch
import utils.nethook as nethook
from transformers.cache_utils import HybridCache
from utils.extra_utils import tokenize_single_answer

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    num_samples = 10,
    past_kv = None,
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.

    ** NEW **
    This function is modified to do batch processing, where the states at different layer/tok pos is batched together (Not different samples).
    Thus num_samples (the number of corrupted duplicated samples) is used to gauge where's the next sample.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    nb = (inp.shape[0]-1)//num_samples

    # patch_spec = defaultdict(list)
    patch_spec = []
    for t, l in states_to_patch:
        pc = defaultdict(list)
        pc[l].append(t)
        patch_spec.append(pc)
        # patch_spec[l].append(t)
    if states_to_patch != []:
        assert len(patch_spec) == nb, f"number of patching states should be equal to number of corrupted runs., {len(patch_spec)} != {nb}"
    else:
        patch_spec = [defaultdict(list)]
    

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    
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
        if past_kv is not None:
            return x
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = [noise_fn(
                    torch.from_numpy(prng(num_samples, e - b, x.shape[2]))
                ).to(x.device) for _ in range(nb)] # get a random noise for each corrupted sample

                for ni,i in enumerate(range(1,x.shape[0],num_samples)): # use diff noise for diff sample
                    if replace:
                        x[i:i+num_samples, b:e] = noise_data[ni]
                    else:
                        x[i:i+num_samples, b:e] += noise_data[ni]
            return x
        # if layer not in patch_spec:
        if all([layer not in pc for pc in patch_spec]):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for i,pc in enumerate(patch_spec):
            for t in pc[layer]: # only when layer is in pc
                pos_start = i*(num_samples) + 1
                h[pos_start:pos_start+num_samples, t] = h[0, t] # 0 is always the clean one
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        # [embed_layername] + list(patch_spec.keys()) + additional_layers,
        [embed_layername] + sum([list(pc.keys()) for pc in patch_spec],[]) + additional_layers,
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

    # We report softmax probabilities for the answers_t token predictions of interest. # average across all corrupted runs
    probs = []
    for j in range(1,outputs_exp.logits.shape[0],num_samples):
        probs.append(torch.softmax(outputs_exp.logits[j:j+num_samples, -1, :], dim=1).mean(dim=0)[answers_t])
    probs = torch.stack(probs)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced, out_kv

    return probs, out_kv


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    answer = None, # if answer provided, no need to predict again,
    input_until = None,
    batch_size = 32,
    use_kv_caching = True, # use it for sequential outputs to save computation (should set to fp16 or 32),
    scores = None,
    use_fs = False # if use_fs need to ensure find_token_range args find_sub_range is True
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    # inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    if isinstance(prompt,str):
        inp = torch.stack([torch.tensor(mt.tokenizer.encode(p)) for p in [prompt]* (samples + 1)],dim = 0).to(mt.model.device)
    else:
        inp  = prompt.repeat(samples+1,1).to(mt.model.device)
    e_range = find_token_range(mt.tokenizer, inp[0], subject,find_sub_range = use_fs) # find pos of the subject in the input prompt
    token_range = range(e_range[0],inp.shape[1]) # replacing the previous tokens are pointless.

    if len(answer) == 1: # for explanation
        answer_t = tokenize_single_answer(answer,mt.tokenizer,mt.model_name)
    else:
        answer_t = mt.tokenizer.encode(answer,add_special_tokens=False)
    if not isinstance(answer_t,list):
        answer_t = [answer_t]

    if input_until is not None: # exclude the last part of the prompt after choices
        if isinstance(input_until,str):
            input_range = find_token_range(mt.tokenizer, inp[0], input_until,find_sub_range = use_fs)
            token_range = range(e_range[0],input_range[0]) 
        else:
            token_range = range(e_range[0],e_range[0]+ input_until)

    final_input_tokens = [decode_tokens(mt.tokenizer, inp[0])[j] for j in token_range]
    
    
    low_score_store = []

    ## Low score (No patching) ##
    low_score_inp = deepcopy(inp)

    ## if using gemma2, we need to define a cache (issue with the current transformers)
    if 'gemma2' in mt.model_name:
        past_kv_cache = HybridCache(config=mt.model.config,
                                   max_batch_size=low_score_inp.shape[0], 
                                  max_cache_len=2048,
                                  dtype = torch.bfloat16
                                  )
        cache_position = torch.arange(low_score_inp.shape[1],dtype = torch.int32).to(low_score_inp.device)
        low_past_kv = (past_kv_cache,cache_position)
    else:
        low_past_kv = None # DO KV CACHEING FOR EFFICIENCY
    
    if scores is None:
        for ans_token in answer_t:
            low_score,low_past_kv = trace_with_patch(
                mt.model, low_score_inp, [], ans_token, e_range, noise=noise, uniform_noise=uniform_noise,past_kv = low_past_kv if use_kv_caching and len(answer_t) > 1 else None,num_samples = samples)
            low_score_inp = torch.tensor(ans_token).repeat(low_score_inp.shape[0],1).to(low_score_inp.device)
            low_score_store.append(low_score.item())
        base_score = generate_sequence_probs(inp[:1],mt,answer_t)
        if 'gemma2' in mt.model_name:
            del low_past_kv
            torch.cuda.empty_cache()

    else:
        low_score_store = scores['low_score']
        base_score = scores['high_score']

    if len(answer_t) > 1:
        low_score = torch.tensor(low_score_store).mean()
    else:
        low_score = low_score_store[0] if isinstance(low_score_store,list) else low_score_store
        low_score = torch.tensor(low_score)

    ## Patching ##
    differences = trace_important_states(
        mt.model,
        mt.num_layers if '70b' not in mt.model_name.lower() else range(0,mt.num_layers,2), # for 70b models, we skip every other layer
        inp,
        e_range,
        answer_t,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace,
        token_range=token_range,
        batch_size = batch_size,
        use_kv_caching = True if use_kv_caching and len(answer_t) > 1 else False,
        num_samples = samples
        )
    differences = differences.detach().cpu()
    
    if len(answer_t) > 1:
        sum_differences = differences.mean(dim = -1)
    else:
        sum_differences = differences.squeeze(-1)
    sum_differences = sum_differences - low_score ## add in the difference p(o*,h,i) - p(o*)
    

    return dict(
        scores=sum_differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp[0],
        input_tokens=final_input_tokens,
        subject_range=[0,e_range[1]- e_range[0]],
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    batch_size = 32,
    use_kv_caching = True,
    num_samples = 10
):
    ntoks = inp.shape[1]
    table = []
    if isinstance(num_layers,int):
        layer_range = range(num_layers)
    else:
        layer_range = deepcopy(num_layers)
        num_layers = len(layer_range)
        

    pos_to_edit = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        for layer in layer_range:
            pos_to_edit.append((tnum, layername(model, layer)))
    
    for i in range(0,len(pos_to_edit),batch_size):
        take_size = batch_size if i+batch_size <= len(pos_to_edit) else len(pos_to_edit) - i
        ## for batch, we only repeat 1: onwards, 0 is the clean input
        inp_rolled = inp[1:].repeat(take_size,1)
        inp_rolled = torch.cat([inp[:1],inp_rolled],dim = 0)
        all_r = []
        if use_kv_caching:
            if 'gemma-2-' in model.name_or_path:
                past_kv_cache = HybridCache(config=model.config,
                                        max_batch_size=inp_rolled.shape[0], 
                                        max_cache_len=2048,
                                        dtype = torch.bfloat16
                                        )
                cache_position = torch.arange(inp_rolled.shape[1],dtype = torch.int32).to(inp_rolled.device)
                past_kv = (past_kv_cache,cache_position)
            else:
                past_kv = None
        else:
            past_kv = None
        for answer in answer_t:
            r,past_kv = trace_with_patch(
                            model,
                            inp_rolled,
                            pos_to_edit[i:i+batch_size],
                            answer,
                            tokens_to_mix=e_range,
                            noise=noise,
                            uniform_noise=uniform_noise,
                            replace=replace,
                            past_kv = past_kv,
                            num_samples = num_samples
                        )
            all_r.append(r)
            if use_kv_caching:
                inp_rolled = torch.tensor(answer).repeat(inp_rolled.shape[0],1).to(inp_rolled.device)
            else:
                inp_rolled = add_column_with_int(inp_rolled,answer)
                past_kv = None

        table.append(torch.stack(all_r).T)
        if 'gemma-2-' in model.name_or_path and past_kv is not None:
            del past_kv
            torch.cuda.empty_cache()  # Optionally clear CUDA cache if using GPU
    
    table = torch.cat(table,dim = 0)
    table = table.view(len(token_range),num_layers,len(answer_t))
    return table

def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model,"model"): # for llama models
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind == 'self_attn'
        return f"model.layers.{num}{'' if kind is None else '.'+ kind}"
    assert False, "unknown transformer structure"

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring,allow_lowercase=True,find_sub_range=False):
    """
    Note in few_shot, there might be repeatable subject tokens. They key here is to find a restricted frame.
    """
    if not isinstance(token_array[0],str): # can either be input_ids or decoded list of tokens
        toks = decode_tokens(tokenizer, token_array)
    else:
        toks = token_array
    toks = [t.strip() for t in toks] # get rid of spaces just in case there are any
    if find_sub_range:
        tok_to_start_from = find_subject_range(toks)
    else:
        tok_to_start_from = 0
    toks = toks[tok_to_start_from:]
    whole_string = "".join(toks)
    substring = "".join([s.strip() for s in substring.split()]) # to standardize no spaces
    if allow_lowercase:
        whole_string = whole_string.lower()
        substring = substring.lower()
    if substring not in whole_string:
        return None, None
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start+tok_to_start_from, tok_end+tok_to_start_from)

def find_subject_range(token_array):
    """
    Given a text, use a common key to filter out unwanted text
    key = Question:
    [text][text][key-2][text]...[key-1]**... -> get position of ** after last occurence of the key
    token_array = list of tokens for the string 
    """
    key_to_cut = 'Question:'
    joined_token_array = "".join(token_array)
    joined_token_key = "".join([s.strip() for s in key_to_cut.split()])
    no_keys = len(joined_token_array.split(joined_token_key)) -1
    total_tokens_traversed = 0
    for _ in range(no_keys):
        key_loc = joined_token_array.index(joined_token_key)
        loc = 0
        tok_end = 0
        for i,t in enumerate(token_array): # from the end
            loc += len(t)
            if loc >= key_loc + len(joined_token_key): # first one includes the end of the key
                tok_end = i + 1
                break
        total_tokens_traversed += tok_end
        token_array = token_array[tok_end:] # truncate away the 1st key
        joined_token_array = "".join(token_array)
    return total_tokens_traversed # return back total tokens traversed from the back. ,ie end of key1

def generate_sequence_probs(inps,mt,answers,return_type = 'mean'):
    kv = None
    base_score = []
    for a in answers:
        with torch.no_grad():
            out = mt.model(inps,use_cache=True,past_key_values = kv)
        kv = out.past_key_values
        probs = torch.softmax(out.logits[0,-1],dim= 0)[a]
        inps = torch.tensor(a).repeat(inps.shape[0],1).to(inps.device)
        base_score.append(probs.item())
    if return_type == 'mean':
        return numpy.mean(base_score)
    else:
        return numpy.exp(numpy.log(base_score).sum())

def collect_embedding_std(mt, subjects):
    alldata = []
    from tqdm import tqdm
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def add_column_with_int(tensor, int_value):
    num_rows = tensor.size(0)
    column_to_add = torch.full((num_rows, 1), int_value).to(tensor.device)
    new_tensor = torch.cat((tensor, column_to_add), dim=1)
    return new_tensor

def align_two_token_lists(target,check): # both have to be list of tokens
    for i,t in enumerate(check):
        if target[i] != t:
            return False
    return True

def add_extra_tokens_to_subject(subject,prompt,tokenizer,add = 1,direction='front'):
    """
    add additional tokens to the subject from the prompt, where the subject is part of.
    """
    tokenized_prompt = tokenizer.encode(prompt)
    subject_range = find_token_range(tokenizer,tokenized_prompt,subject)
    if direction == 'front':
        if add < 0 and subject_range[1] - subject_range[0] == 1: # if the subject is only 1 token
            return subject
        subject_range = (subject_range[0],subject_range[1]+add)
    else:
        if subject_range[0] - add < 0: # if the subject is at the start of the prompt
            return subject
        subject_range = (subject_range[0]-add,subject_range[1])
    new_sub = tokenizer.decode(tokenized_prompt[subject_range[0]:subject_range[1]])
    return new_sub


# if __name__ == "__main__":
#     main()
