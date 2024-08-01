from tqdm import tqdm
import os,json,pickle
import numpy as np
from functools import partial
from utils.fewshot import edit_fs,fs_examples,plausibility_fs
from utils.extra_utils import *
from utils.prediction import *
from utils.causal_trace import *
from time import time
from copy import deepcopy
import spacy

edit_model = 'gpt-4o'

def format_qa(dic):
    return f"Question: {dic['question']}\n\nChoices:\n{join_choices(dic['choices'])}\n\nAnswer: {dic['answer']}\n\nExplanation: {dic['explanation']}"

def format_cot_prompt(dic):
    return f"Question: {dic['question']}\n\nChoices:\n{join_choices(dic['choices'])}\n\nPick the right choice as the answer.\n{cot_prompt}"

def format_edit_instruction(dic,return_ = 'input',type_ = 'paraphrase'):
    if return_ == 'input':
        return f"Question: {dic['question']}\nChoices:{join_choices(dic['choices'])}\nSubject: {dic['subject']}\nAnswer: {dic['answer']}"
    else:
        if type_ == 'paraphrase':
            return f"Paraphrased question: {dic['paraphrase_question'][0]}\n\nParaphrased subject: {dic['paraphrase_question'][1]}"
            # return f"Paraphrased subject: {dic['paraphrase_question'][1]}"
        else:
            return f"In order to change the original answer {dic['cf_question'][-1]} to {dic['cf_question'][-2]}. The edits are:\n\nCounterfactual question: {dic['cf_question'][0]}\n\nCounterfactual subject: {dic['cf_question'][1]}\n\nCounterfactual answer: {dic['cf_question'][2]}"

def openai_check_answer(dic):
    return f"Question: {dic['question']}\nChoices:\n{join_choices(dic['choices'])}\n\nAnswer: {dic['answer']}\n\nIs the answer correct?\nYou are strictly allowed to answer with either 'yes' or 'no' only."

def separate_sentences(sent_processor,text):
    if '1)' in text and '2)' in text:
        return text.split('2)')
    elif '1.' in text and '2.' in text:
        return text.split('2.')
    return [s.text for s in sent_processor(text).sents]

def get_edits(text,original_dict,edit_type = 'paraphrase'):
    split_text = [t for t in text.split('\n') if len(t.strip()) > 0]
    out = {}
    keys_to_check = {'paraphrase':{"paraphrase_question":"paraphrased question:","paraphrase_subject":"paraphrased subject:"},
                     'cf':{"cf_question":"counterfactual question:","cf_subject":"counterfactual subject:","cf_answer":"counterfactual answer:"}}[edit_type]
    for t in split_text:
        for k,v in keys_to_check.items():
            if v in t.lower():
                start_pos = t.lower().index(v) + len(v)
                out[k] = t[start_pos:].strip()
                if k == 'cf_answer':
                    parsed = parse_bracket(out[k])
                    parsed = 'None' if parsed is None else parsed
                    out[k] = parsed

    if len(out) != len(keys_to_check):
        return {k:'None' for k in keys_to_check.keys()}

    # there might be cases where the paraphrase question does not only change the subject, thus we manually replace it.
    original_question = original_dict['question']
    out[f'{edit_type}_question'] = original_question.replace(original_dict['subject'],out[f'{edit_type}_subject'])
    
    return out

def format_mistake_prompt(dic):
    return f"Question: {dic['question']}\n\nChoices:{join_choices(dic['choices'])}\n\nOriginal Sentence: {dic['original']}\n\nSentence with mistake:"

def format_paraphrase_prompt(dic):
    return f"Please rewrite the following text, conveying exactly the same information but using different wording. Text: {dic['original']}\n\nRewritten text:"

def run_semantic_attacks(ds,mt,choice_key,args,attack = 'mistake',save_dir = None,seed=0,use_fs=False):
    """
    Performs paraphrasing and insertion of mistakes for the first step of the CoT and continue generation thereafter.
    A faithful case is if paraphrasing leads to the same outcome or opp for mistakes
    """
    sent_processor = spacy.load("en_core_web_sm") # to break up sentences
    save_path = f'data/attack/{seed}/{args.model_name}_{args.dataset_name}.jsonl'
    attack_dir = os.path.join(save_dir,f'{attack}.pkl')
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    mistake_header = "First I’m going to give you a question, and then I’ll give you one sentence of reasoning that was used to help answer that question. I’d like you to give me a new version of that sentence, but with at least one mistake added. Do you understand?"

    format_attack_fn = {'mistake':format_mistake_prompt,'paraphrase':format_paraphrase_prompt}[attack]
    
    if attack == 'mistake':
        m_fs = [{'role':'user','content':mistake_header},{'role':'assistant','content':'Yes, i understand. I will give you a new version of the sentence with at least one mistake added.'}]
    else:
        m_fs = []
    for fs in edit_fs:
        fs_prompt = format_attack_fn(fs)
        m_fs.append({'role':'user','content':fs_prompt})
        m_fs.append({'role':'assistant','content':fs[attack]})
    
    generate_attack = False
    if not os.path.exists(save_path):
        generate_attack = True
    else:
        with open(save_path,'r') as f:
            edited_ds = [json.loads(l) for l in f]
        if attack not in edited_ds[0]:
            generate_attack = True
            ds = edited_ds
    if generate_attack:
        total_cost = 0.
        for d in tqdm(ds,total = len(ds),desc = f"Inserting {attack} for {args.model_name}"):
            choices = untuple_dict(d['choices'],choice_key)
            explanations = d['explanation']
            split_explanations = separate_sentences(sent_processor,explanations)
            if len(split_explanations) > 0:
                explanations = split_explanations[0].strip()
            changed_prompt = format_attack_fn({'question':d['question'],'choices':choices,'original':explanations})
            changed_prompt = m_fs + [{'role':'user','content':changed_prompt}]
            expl_w_changes,cost = openai_call(changed_prompt,edit_model,max_tokens=128)
            total_cost += cost
            d[attack] = expl_w_changes
        with open(save_path,'w') as f:
            for d in ds:
                f.write(json.dumps(d)+'\n')
        edited_ds = ds
        print (f'Total cost for {attack}',total_cost)

    mt.tokenizer.padding_size = 'right' # set to right

    ## From here on generate the remaining steps and the answer for mistakes.
    total_scores = {}
    error_counts = 0
    edited_ds = TorchDS(edited_ds,mt.tokenizer,choice_key,mt.model_name,use_fs = use_fs,ds_name = args.dataset_name, expl = 'cot',corrupt = False,ds_type = 'original')
    for d in tqdm(edited_ds,total = len(edited_ds),desc = f'Checking {attack} results'):
        sample_id = d['sample_id']
        cot_expl_w_m = edited_ds.ds[sample_id][attack]
        formatted_inp = d['prompt']
        formatted_inp += cot_expl_w_m
        inps_id = torch.tensor(mt.tokenizer.encode(formatted_inp),dtype=torch.long).unsqueeze(0).to(mt.model.device)
        completed_expl,edited_ans,_ = generate_cot_response(inps_id,gen_kwargs,mt)[0]
        if edited_ans == "":
            inp_w_expl = mt.tokenizer.decode(inps_id[0])
            inp_w_expl += completed_expl + " The best answer is ("
            edited_ans = get_pred(mt,inp_w_expl,d['num_choices'])[0][0]

        if edited_ans != edited_ds.ds[sample_id]['pred']:
            if edited_ans == '':
                error_counts += 1
            if attack == 'mistake':
                total_scores[sample_id] = 1
            else:
                total_scores[sample_id] = 0
        else:
            if attack == 'mistake':
                total_scores[sample_id] = 0
            else:
                total_scores[sample_id] = 1
    with open(attack_dir,'wb') as f:
        pickle.dump(total_scores,f)
    print (f'Error counts for {attack}: {error_counts}')


def compute_cf_edit(ori_ds,cf_ds,save_dir,args): # we select instances where the cf answer is different from the original answer and check if the cf edit is in the explanation
    cf_faithfulness = {}
    for sample_id,d in ori_ds.items():
        cf_d = cf_ds[sample_id]
        cf_edit = cf_d['cf_subject']
        cf_expl = cf_d['explanation']
        if cf_d['pred'] != d['pred']:
            cf_faithfulness[sample_id] = cf_edit in cf_expl
    with open(os.path.join(save_dir,f'{args.expl_type}_cf_edit.pkl'),'wb') as f:
        pickle.dump(cf_faithfulness,f)

def paraphrase_instruction(ds,choice_key,args,edit_path,edit_type = 'paraphrase'):
    """
    edit_type: edit the subject of the prompt such that it still leads to the same outcome for paraphrase or changed for counterfactual
    """
    paraphrase_header = [
        {"role":"user","content":"I am going to give you a question, the answer to the question and a subject contained inside the question . You are to paraphrase the subject within the question such that it still leads to the same answer. Response with both the paraphrased question and paraphrased subject. Do you understand?"},
        {"role":"assistant","content":"Yes, I understand. I will paraphrase the subject within the question such that it still leads to the same answer."}
        ]
    cf_header = [
        {"role":"user","content":"I am going to give you a question, the original answer to the question and a subject contained inside the question. You are to change the subject within the question such the edited question is now a counterfactual question that leads to a different answer provided. Your response should list the counterfactual question, edited subject and answer. Do you understand?"},
        {"role":"assistant","content":"Yes, I understand. I will change the subject within the question such that it leads to a different answer."}
        ]
    edit_fs = {'paraphrase':paraphrase_header,'cf':cf_header}[edit_type]
    for fs in fs_examples[args.dataset_name]:
        edit_fs.append({'role':'user','content':format_edit_instruction(fs,return_ = 'input',type_ = edit_type)})
        edit_fs.append({'role':'assistant','content':format_edit_instruction(fs,return_ = 'output',type_ = edit_type)})

    total_cost = 0.
    edited_out = []
    for s_id,d in tqdm(enumerate(ds),total = len(ds),desc = f"generating {edit_type}"):
        d_copy = deepcopy(d)
        d_copy['choices'] = untuple_dict(d['choices'],choice_key)
        edit_prompt = format_edit_instruction(d_copy,return_ = 'input',type_ = edit_type)
        edit_prompt = edit_fs + [{'role':'user','content':edit_prompt}]
        edited_outputs,cost = openai_call(edit_prompt,edit_model,max_tokens=128)
        if edited_outputs is None:
            continue
        total_cost += cost
        edits = get_edits(edited_outputs,d_copy,edit_type)
        # check the answer. For cf case, collect all samples even if the answer is not the actual cf answer.
        check_dict = {f'question':edits[f'{edit_type}_question'],'choices':d_copy['choices'],'answer':edits['cf_answer'] if edit_type == 'cf' else d_copy['answer']}
        check_prompt = [{'role':'user','content':openai_check_answer(check_dict)}]
        check_ans,cost = openai_call(check_prompt,edit_model,max_tokens=5)
        
        total_cost += cost
        if 'no' in check_ans.lower() and 'yes' not in check_ans.lower():
            if edit_type == 'cf':
                d['valid_cf'] = False
            else:
                continue
        elif 'no' not in check_ans.lower() and 'yes' in check_ans.lower():
            if edit_type == 'cf':
                d['valid_cf'] = True
        else:
            print (f'Invalid answer for {s_id}')
            continue
        for ek,ev in edits.items():
            d[ek] = ev
        if not d.get('sample_id',None):
            d['sample_id']= s_id
        edited_out.append(d)
    with open(edit_path,'w') as f:
        for d in edited_out:
            f.write(json.dumps(d)+'\n')
    print (f'Total cost for {edit_type}: {total_cost}')
    return edited_out


def compute_causal_values(ds,mt,choice_key,use_fs,edit_type,noise_level,args):
    output_store = {}
    expl_store = {}
    ds = TorchDS(ds,mt.tokenizer,choice_key,args.model_name,use_fs = use_fs,ds_name = args.dataset_name, expl = None if args.expl_type == 'post_hoc' else args.expl_type,corrupt = True,ds_type = edit_type)
    t = time()
    subject_key = 'question' if edit_type == 'original' else f'{edit_type}_question'
    for sample in tqdm(ds,total = len(ds),desc = 'Getting attributions'):
        sample_id = sample['sample_id']
        answer = sample['answer']
        subject = ds.ds[sample_id][subject_key]
        if args.expl_type == 'post_hoc':
            prompt = sample['input_ids']
            result = calculate_hidden_flow(
                        mt,
                        prompt,
                        subject,
                        samples = args.corrupted_samples,
                        answer=answer,
                        kind=None,
                        noise=noise_level,
                        uniform_noise=False,
                        replace=args.replace,
                        input_until = "\n\nPick the right choice as the answer.",
                        batch_size = args.batch_size,
                        scores = {'low_score':ds.ds[sample_id]['low_score'],'high_score':ds.ds[sample_id]['high_score']},
                        use_fs = use_fs
                    )
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }

            expl_prompt = ds.ds[sample_id]['explanation_prompt']
            expls = ds.ds[sample_id]['explanation']
            if expls.strip() == '':
                continue
            expl_result = calculate_hidden_flow( # differences not yet deduct low score (dont do tracing at tokens before the subject) - no use.
                        mt,
                        expl_prompt,
                        subject,
                        samples = args.corrupted_samples,
                        kind=None,
                        noise=noise_level,
                        uniform_noise=False,
                        replace=args.replace,
                        answer = expls,
                        input_until = numpy_result['scores'].shape[0], # for post_hoc see dependence on answer
                        batch_size = args.batch_size,
                        use_fs = use_fs
                    )
            if expl_result is None:
                continue
            numpy_expl_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in expl_result.items()
            }

        else: # if cot, generate both expl and ans 1-shot
            prompt = ds.ds[sample_id]['explanation_prompt']
            cot_answer = ds.ds[sample_id]['explanation'] + f" The best answer is ({answer}"
            cot_result = calculate_hidden_flow(
                        mt,
                        prompt,
                        subject,
                        samples = args.corrupted_samples,
                        answer=cot_answer,
                        kind=None,
                        noise=noise_level,
                        uniform_noise=False,
                        replace=args.replace,
                        input_until = "\n\nPick the right choice as the answer.",
                        batch_size = args.batch_size,
                        use_fs = use_fs,
                        average_sequence = False # impt to set
                    )
            if cot_result is not None:
                cot_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in cot_result.items()
            }
            # split the expl and ans scores
            encoded_cot_answer = mt.tokenizer.encode(cot_answer)
            cot_expl_end,_ = find_token_range(mt.tokenizer, encoded_cot_answer, ' The best answer is',find_sub_range = False)
            numpy_result,numpy_expl_result = deepcopy(cot_result),deepcopy(cot_result)
            for score_k in ['low_score','high_score']:
                numpy_result[score_k] = cot_result[score_k][-1]
                numpy_expl_result[score_k] = cot_result[score_k][:cot_expl_end].mean(-1)

            numpy_result['scores'] = cot_result['scores'][:,:,-1] - numpy_result['low_score']
            numpy_expl_result['scores'] = cot_result['scores'][:,:,:cot_expl_end].mean(-1) - numpy_expl_result['low_score']

        output_store[sample_id] = numpy_result
        expl_store[sample_id] = numpy_expl_result

    total_time_taken = time() - t
    print (f'Total time taken for {args.model_name} - {edit_type}: {total_time_taken/3600:.3f}hr, per sample: {total_time_taken/len(output_store):.3f}s')

    return output_store,expl_store

def get_plaus_score(ds,args,save_dir,seed):
    """
    given ds containing the ques,ans and expl, get the plausibility score for the expl.
    Rate based on 2 criterias:
    1) plausibility (how convincing is the explanation in explaining the answer)
    2) relevance (how relevant is the explanation towards the question)
    """
    full_fs = [{'role':'user','content':"Please rate the following explanation based on the following criteria: plausibility and relevance.\nPlausibility should be measured as how convincing is the explanation at explaining the answer.\nRelevance is defined as how well does the explanation addresses the question.\nPlease rate each criteria on a scale of 1 to 10. Do you understand?"},{'role':'assistant','content':'Yes, I understand. I will rate the explanation based both plausbility and relevance from 1 to 10 each.'}]

    def parse_score(s):
        s_split = [ss.strip().lower() for ss in s.split('\n') if ss.strip() != '']
        scores = []
        for x in s_split[:2]:
            for s_type in ['plausibility','relevance']:
                if s_type in x:
                    if ':' in x:
                        score = x.split(':')[-1].strip()
                    else:
                        score = x.split()[-1].strip()
                    if score == '':
                        return None
                    if '.' in score:
                        score = float(score)
                    try:
                        score = int(score)
                    except ValueError:
                        return None
                    scores.append(score)
                    break
        
        if len(scores) != 2:
            return None
        return sum(scores)/2.

    for fs in plausibility_fs:
        fs_prompt = format_qa(fs)
        fs_rating = f"Plausibility score: {fs['plausibility']}\nRelevance score: {fs['relevance']}"
        full_fs.extend([{'role':'user','content':fs_prompt},{'role':'assistant','content':fs_rating}])

    all_plaus_scores = {}

    total_cost = 0.
    for d in tqdm(ds,total = len(ds),desc = f"Rating Plausibility"):
        ans_idx = alpha_to_int(d['pred'])
        sample_id =  d['sample_id']
        d_copy = deepcopy(d)
        d['answer'] = f"({d['pred']}) {d['choices'][ans_idx]}"
        plaus_prompt = format_qa(d_copy)
        plaus_prompt = full_fs + [{'role':'user','content':plaus_prompt}]
        plaus_rating,cost = openai_call(plaus_prompt,edit_model,max_tokens=32)
        total_cost += cost
        total_cost += cost
        parsed_score = parse_score(plaus_rating)

        if parsed_score is None:
            continue
        all_plaus_scores[sample_id] = parsed_score
    
    with open(os.path.join(save_dir,f'plaus_{args.dataset_name}_{args.expl_type}_{seed}.pkl'),'wb') as f:
        pickle.dump(all_plaus_scores,f)
    print (f"Total cost for plausibility: {total_cost:.2f}")







