from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from utils.extra_utils import join_choices,alpha_to_int,int_to_alpha,tokenize_single_answer
from utils.prediction import format_mcq,format_input
from torch.nn.utils.rnn import pad_sequence
from utils.model_data_utils import load_hf_ds
import numpy as np
import random
from tqdm import tqdm
from time import time
from transformers.cache_utils import HybridCache, DynamicCache

# seed = 2
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)

def get_num_layers(model):
    total_layers = []
    if hasattr(model, "transformer"):
        prefix = 'transformer.h.'
    elif hasattr(model, "gpt_neox"):
        prefix = 'gpt_neox.layers.'
    elif hasattr(model, "model"):
        prefix = 'model.layers.'
    else:
        raise ValueError("Unknown model type")
    total_layers.extend([n.split(prefix)[-1].split('.')[0] for n,_ in model.named_parameters() if prefix in n])
    
    total_layers = [int(x) for x in total_layers]
    num_layers = max(total_layers) + 1
    return num_layers

# m_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
m_path = "google/gemma-2-27b-it"
# m_path = 'bartowski/gemma-2-27b-it-GGUF'
gguf_file = "gemma-2-27b-it-Q6_K_L.gguf"
# m_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# m_path = 'microsoft/Phi-3-mini-4k-instruct'
# m_path = "EleutherAI/gpt-neo-1.3B"


# m_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# m_path = "casperhansen/llama-3-70b-instruct-awq"
tokenizer = AutoTokenizer.from_pretrained(m_path)
    
if '27B' not in m_path and '70B' not in m_path:
    m = AutoModelForCausalLM.from_pretrained(m_path,torch_dtype = torch.bfloat16 if 'gpt2' not in m_path else torch.float32,attn_implementation="flash_attention_2" if 'gpt2' not in m_path else None,device_map = 'auto',trust_remote_code = True).eval()
else:
    quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4")
    m = AutoModelForCausalLM.from_pretrained(m_path,quantization_config = quant_config,torch_dtype = torch.bfloat16,attn_implementation="flash_attention_2" if 'gpt2' not in m_path else None,device_map = 'auto',trust_remote_code = True,low_cpu_mem_usage=True).eval()
chat = True if 'gpt2' not in m_path else False
# chat=False
# data_path = "data/esnli_test.jsonl"
data,_ = load_hf_ds('esnli')

if isinstance(data,list):
    random.shuffle(data)
    data = data[:50]
else:
    data = data.shuffle(seed=42).select(list(range(50)))

# random.shuffle(data)
scores = []
# m.forward = torch.compile(m.forward, mode="reduce-overhead", fullgraph=True)
# m._supports_cache_class = True
# m.generation_config.cache_implementation = None
for j,d in tqdm(enumerate(data),total=len(data)):
    joined_choice = join_choices(d['choices'])
    prompt = format_mcq(d['question'],joined_choice,is_chat=chat)
    if chat:
        prompt = format_input(prompt,tokenizer) + 'The correct answer is ('
        # prompt = format_input(prompt,tokenizer) + "Let's think step by step: "
    all_inps = []
    all_labels= []
    ans = d['answer']
    tokenized_prompt = torch.tensor(tokenizer.encode(prompt),dtype = torch.long).repeat(4,1).to(m.device)
    t = time()
    cache = HybridCache(config=m.config, max_batch_size=tokenized_prompt.shape[0], max_cache_len=100
                        ,dtype = m.dtype,
                        device = m.device
                        )
    cache_position = torch.arange(tokenized_prompt.shape[1],dtype = torch.int32).to(m.device)
    cache = None
    all = []
    for _ in range(10):
        t = time()
        with torch.no_grad():
            out = m(tokenized_prompt,use_cache = True,past_key_values = cache,cache_position = cache_position)
            # out = m(tokenized_prompt,use_cache = True,past_key_values = cache)
        # out = m.generate(tokenized_prompt,do_sample=True,max_new_tokens=1,pad_token_id = tokenizer.eos_token_id)
        next_token = out.logits[:, -1].argmax(dim=-1)
        # cache = out.past_key_values
        tokenized_prompt = next_token.unsqueeze(-1)
        cache_position = cache_position[-1:] + 1
        # all.append(next_token.item())
        print (time()-t)
    # print (tokenizer.decode(all))
    exit()
    
        
    # print(next_token, repr(tokenizer.decode(next_token.item())))

    # concatted = torch.cat([tokenized_prompt, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
    # cache_position_concatted = torch.cat([cache_position, cache_position_next], dim=-1)

    # with torch.no_grad():
    #     out2_manual = m(concatted, cache_position=cache_position_concatted)
    #     out2_opt = m(next_token.unsqueeze(0).unsqueeze(0), cache_position=cache_position_next, use_cache=True, past_key_values=cache)

    # print(tokenizer.decode(out2_manual.logits[0, -1].argmax()), tokenizer.decode(out2_opt.logits[0, -1].argmax()))
    # exit()
    # pred = tokenizer.decode(out[0,tokenized_prompt.shape[1]:],skip_special_tokens=True)
    # scores.append(int(pred == ans))

# print (np.mean(scores))


