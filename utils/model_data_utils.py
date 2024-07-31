from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import utils.nethook as nethook
from datasets import load_dataset,concatenate_datasets
from utils.fewshot import subject_extract_fs
from utils.extra_utils import openai_call,join_choices,filter_samples
import pandas as pd
from tqdm import tqdm
import os,json
import random
import torch

def get_model_path(model_name,expl_type):
    use_fs = False if ('chat' in model_name or 'post_hoc' in expl_type) else True
    # use_fs = False
    if 'llama3' in model_name:
        if 'chat' not in model_name:
            model_path = f"meta-llama/Meta-Llama-3-{model_name.split('-')[-1]}"
        else:
            model_path = f"meta-llama/Meta-Llama-3-{model_name.split('-')[-2]}-Instruct"
    elif 'llama2' in model_name:
        if 'chat' in model_name:
            chat_string = '-chat'
            llama_size = f"{model_name.split('-')[-2]}".lower()
        else:
            chat_string = ''
            llama_size = f"{model_name.split('-')[-1]}".lower()
        model_path = f"meta-llama/Llama-2-{llama_size}{chat_string}-hf"
    elif 'gemma-' in model_name:
        model_path = f"google/gemma-{model_name.split('-')[-2].lower()}-it"
    elif 'gemma2' in model_name:
        model_path = f"google/gemma-2-{model_name.split('-')[-2].lower()}-it"
    elif 'gpt2' in model_name:
        model_path = model_name
    elif 'phi3-chat':
        model_path = "microsoft/Phi-3-mini-4k-instruct"
    return model_path,use_fs

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        m_name = None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if model is None:
            assert model_name is not None
            if '27b' in model_name.lower():
                quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4")
            else:
                quant_config = None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if "gpt2" not in model_name else None,
                device_map = 'auto',
                quantization_config = quant_config
            )
            nethook.set_requires_grad(False, model)
            model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.num_layers = self.get_num_layers()
        self.model_name = m_name
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
    
    def get_num_layers(self):
        total_layers = []
        if hasattr(self.model, "transformer"):
            prefix = 'transformer.h.'
        elif hasattr(self.model, "gpt_neox"):
            prefix = 'gpt_neox.layers.'
        elif hasattr(self.model, "model"):
            prefix = 'model.layers.'
        else:
            raise ValueError("Unknown model type")
        total_layers.extend([n.split(prefix)[-1].split('.')[0] for n,_ in self.model.named_parameters() if prefix in n])
        
        total_layers = [int(x) for x in total_layers]
        num_layers = max(total_layers) + 1
        return num_layers
    


def load_hf_ds(ds_name,seed=0):
    """
    Standardized keys: question, choices, subject, answer
    esnli select if the subject - rationale tokens of the premise is continuous ie 2,3,4 instead of 2,7,8. (there is 3 rationales per example)
    """
    random.seed(42) # we fix the seed here to sample the same samples.
    if ds_name == 'csqa':
        dataset_path = "niurl/eraser_cose"
        choice_key = []
        ds = load_dataset(dataset_path)
        def map_fn(d):
            d['choices'] = [c.strip() for c in d['choices'].split('[sep]')]
            subject_spans = d['evidence_span'][0]
            d['subject'] = " ".join(d['question'][subject_spans[0]:subject_spans[1]])
            d['question'] = " ".join(d['question'])
            return d
        all_ds = []
        for split in ['test','val']:
            split_ds = ds[split]
            split_ds = split_ds.rename_column('classification','answer').rename_column('query','choices')
            split_ds = split_ds.map(map_fn,remove_columns=['evidence_span'],num_proc=8)
            all_ds.append(split_ds)
        ds = concatenate_datasets(all_ds)
    elif ds_name == 'esnli':
        def check_continuous(l,i):
            return True if l[i] == l[i-1] +1 else False
        esnli_cached_path = 'data/esnli_test.jsonl'
        if not os.path.exists(esnli_cached_path):
            ds_path = 'data/esnli_test.csv'
            ds = pd.read_csv(ds_path).to_dict(orient='records')
            acceptable_ds = []
            esnli_format = """Suppose "{sent0}". Can we infer that "{sent1}"?"""
            esnli_choices = ['Yes','No']
            esnli_answer_map = {'entailment':'A','contradiction':'B'}
            for d in ds:
                if d['gold_label'] == 'neutral': # does not have rationale tokens
                    continue
                out_ds = {}
                out_ds['question'] = esnli_format.format(sent0=d['Sentence1'],sent1=d['Sentence2'])
                out_ds['choices'] = esnli_choices
                out_ds['answer'] = esnli_answer_map[d['gold_label']]
                out_ds['correct_explanation'] = [d[f'Explanation_{i}'] for i in range(1,4) if d[f'Explanation_{i}'].strip() != ''] # check for plausibility.
                acceptable_subjects = []
                for i in range(1,4):
                    if d[f'Sentence1_Highlighted_{i}'] != '{}': # we get the rationale tokens only for hypothesis
                        sub_indices = [int(i) for i in d[f'Sentence1_Highlighted_{i}'].split(',')]
                        is_continuous = all([check_continuous(sub_indices,j) for j in range(1,len(sub_indices))])
                        if is_continuous:
                            acceptable_subjects.append(sub_indices)
                    
                # get the subject whose token is the earliest.
                if len(acceptable_subjects) == 0:
                    continue
                else:
                    selected_subjects = sorted(acceptable_subjects,key = lambda x:x[0])[0]
                out_ds['subject'] = " ".join([d['Sentence1'].split()[i] for i in selected_subjects])
                acceptable_ds.append(out_ds)
            with open(esnli_cached_path,'w') as f:
                for d in acceptable_ds:
                    f.write(json.dumps(d)+'\n')
        else:
            with open(esnli_cached_path,'r') as f:
                acceptable_ds = [json.loads(l) for l in f]
    
        ds = random.sample(acceptable_ds,1000) # to get known_ds
        choice_key= []
    elif ds_name == 'arc':
        choice_key = []
        formatted_path = 'data/arc_test.json'
        if not os.path.exists(formatted_path):
            ds = load_dataset('allenai/ai2_arc','ARC-Challenge',split = 'test')
            formatted_ds =[]
            for d in ds:
                dd = {
                    'question':d['question'],
                    'choices': d['choices']['text'],
                    'answer': d['answerKey']
                }
                formatted_ds.append(dd)
            ds_w_subject,total_cost = extract_subject_from_questions(formatted_ds)
            print (f'Total cost for extracting subjects: {total_cost:.2f}')
            ds = ds_w_subject
            with open(formatted_path,'w') as f:
                for d in ds:
                    f.write(json.dumps(d)+'\n')
        else:
            with open(formatted_path,'r') as f:
                ds = [json.loads(l) for l in f]
            
    else:
        raise ValueError(f"Dataset {ds_name} not found.")
    random.seed(seed) # seed back to the original seed
    return ds,choice_key


def extract_subject_from_questions(ds):
    """
    Given a question, extract the subject tokens that are important for the answer.
    """
    header = [
        {'role':'user','content':"Extract a set of rationale tokens from the question that are important for deriving the answer. It is important that the tokens are a set of contiguous tokens that are concise and present in the question.\nYou should also avoid extracting tokens that ends at the end of the question.\nDo you understand the instruction?"},
        {'role':'assistant','content':"Yes, I understand the instruction. I will extract a short set of contiguous tokens that are important."},
    ]

    def format_subject_prompt(question,choices,answer):
        return f"Question: {question}\n\n{choices}\n\nAnswer: {answer}\n\nRationale Tokens: "
    
    formatted_fs = [[{'role':'user','content':format_subject_prompt(fs['question'],fs['choices'],fs['answer'])},{'role':'assistant','content':fs['subject']}]
                    for fs in subject_extract_fs]
    formatted_fs = header + sum(formatted_fs,[])
    total_cost = 0.
    new_d = []
    for d in tqdm(ds,total =len(ds),desc = 'Extracting subjects'):
        num_tries = 0
        question = d['question']
        answer = d['answer']
        choices = join_choices(d['choices'])
        extracted_tokens = None
        prompt_message = formatted_fs + [{'role':'user','content':format_subject_prompt(question,choices,answer)}]
        while not extracted_tokens and num_tries < 3:
            num_tries += 1
            extracted_tokens,cost = openai_call(prompt_message,'gpt-4o',max_tokens=10,temperature = 1.0)
            if extracted_tokens not in question:
                if extracted_tokens.lower() in question.lower():
                    extracted_tokens = extracted_tokens.lower()
                else:
                    extracted_tokens = None 
            total_cost += cost
            # check if the subject is towards the end.
            # if extracted_tokens:
            #     d['subject'] = extracted_tokens
            #     if len(filter_samples([d])) == 0: 
            #         extracted_tokens = None 
            if num_tries == 3:
                break
        if not extracted_tokens:
            continue
        d['subject'] = extracted_tokens
        new_d.append(d)
    return new_d,total_cost



