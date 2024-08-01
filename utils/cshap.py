import pickle
import os,sys
import numpy as np
import shap
from scipy import spatial
from utils.extra_utils import *
from utils.prediction import format_mcq
from utils.causal_trace import find_token_range

def get_shap_values(x, y,explainer):
    """ Compute Shapley Values of (y|x) for given x and y """
    if isinstance(y,list):
        x = [x for _ in y]
    else:
        x = [x]
        y = [y]
    out = []
    for xx,yy in zip(x,y):
        out.append(explainer([xx],[yy]).values[0])
    return out
      

def cc_shap_score(ratios_prediction, ratios_explanation):
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    return cosine


def aggregate_values_explanation(shap_value, tokenizer,to_marginalize =' Yes. Why?'):
    """ 
    Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
    take_until is the len of input since the outpt_len include the input as well.
    """
    len_to_marginalize = tokenizer([to_marginalize], return_tensors="pt", padding=False, add_special_tokens=False).input_ids.shape[1]
    add_to_base = np.abs(shap_value[len_to_marginalize:]).sum(axis=0)
    ratios = shap_value / (np.abs(shap_value).sum(axis=0) - add_to_base) * 100
    out = np.mean(ratios,axis=-1)[:len_to_marginalize]
    return out

def compute_cc_shap(tokenizer,values_prediction, values_explanation, marg_pred='', marg_expl=' Yes. Why?'):
    ratios_prediction = aggregate_values_explanation(values_prediction,tokenizer,marg_pred)
    ratios_explanation = aggregate_values_explanation(values_explanation,tokenizer,marg_expl)
    cosine = cc_shap_score(ratios_prediction,ratios_explanation)

    return cosine
# , dist_correl, mse, var, kl_div, js_div,

def cc_shap_measure(tokenizer,explainer,output_args,is_chat,expl_type='post_hoc'):
    """
    We use the already computed explanation and answer to derive the SHAP values for output and expl separately.
    For both post-hoc and cot, the output is the same, but explanation is sampled. Thus the variance lies with the different generated explanation.
    In particular, how each explanation's divg against the output varies among each other.
    """
    question = output_args['question']
    choices = output_args['choices']
    ans = output_args['answer']
    expl = output_args['explanation']
    if expl_type == 'post_hoc':
        output_prompt = format_mcq(question,choices,expl_=None,is_chat=is_chat)
        if is_chat:
            output_prompt = format_input(output_prompt,tokenizer)
            output_prompt += "The best answer is: ("
        output_shap = explainer([output_prompt],[ans]).values[0]

        expl_prompt= format_mcq(question,choices,answer = ans,expl_='post_hoc',is_chat=is_chat)
        if is_chat:
            expl_prompt = format_input(expl_prompt,tokenizer) + 'Because' 
        else:
            expl_prompt = expl_prompt + '\nBecause' 
        expl_shap = explainer([expl_prompt],[expl]).values[0]
        
    elif expl_type == 'cot':
        output_prompt = format_mcq(question,choices,expl_='cot',is_chat=is_chat)
        if is_chat:
            output_prompt = format_input(output_prompt,tokenizer) + "Let's think step by step: " 
        else:
            output_prompt = output_prompt + "\nLet's think step by step: "
        
        cot_seq = expl + f" The best answer is ({ans})."
        cot_values = explainer([output_prompt],[cot_seq]).values[0]
        ## split up the output_shap and expl_shap
        key_to_find = 'The best answer is ('
        encoded_cot_seq = tokenizer.encode(cot_seq)
        expl_loc,ans_loc = find_token_range(tokenizer,encoded_cot_seq,key_to_find)
        output_shap = cot_values[:,ans_loc].reshape(-1,1)
        expl_shap = cot_values[:,:expl_loc]

    else:
        raise ValueError(f'Unknown explanation type {expl_type}')

    marg_pred = output_prompt.split('\n\nPick the right choice as the answer.')[0] 

    cosine = compute_cc_shap(tokenizer,output_shap, expl_shap, marg_pred=marg_pred, marg_expl=marg_pred)
    return 1 - cosine 

def run_cc_shap(mt,ds,choice_key,args,pred_dir):
    save_path = os.path.join(pred_dir,f'{args.expl_type}_ccshap.pkl')
    mt.model.generation_config.is_decoder = True
    mt.model.config.is_decoder = True
    if not os.path.exists(save_path):
        teacher_forcing_model = shap.models.TeacherForcing(mt.model, mt.tokenizer)
        masker = shap.maskers.Text(mt.tokenizer, mask_token="...", collapse_mask_token=True)
        explainer = shap.Explainer(teacher_forcing_model, masker,silent=True)
        out =  {}
        for d in tqdm(ds,total = len(ds),desc = f'Running CC-SHAP for {args.model_name} using {args.expl_type}'):
            output_args = {}
            output_args['question'] = d['question']
            output_args['choices'] = join_choices(untuple_dict(d['choices'],choice_key))
            output_args['answer'] = d['pred']
            output_args['explanation'] = d['explanation']
            sample_id = d['sample_id']
            cos_similarity = cc_shap_measure(mt.tokenizer,explainer,output_args,is_chat=True if 'chat' in args.model_name else False,expl_type=args.expl_type)
            if cos_similarity is not None:
                out[sample_id] = cos_similarity
        
        with open(save_path,'wb') as f:
            pickle.dump(out,f)
    else:
        print (f'Already computed {save_path}')


    


