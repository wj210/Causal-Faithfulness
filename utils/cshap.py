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
    return cosine, True if not (np.linalg.norm(ratios_prediction) < 1e-8 or np.linalg.norm(ratios_explanation) < 1e-8) else False


def aggregate_values_explanation(shap_value, tokenizer,to_marginalize =' Yes. Why?'):
    """ 
    Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
    take_until is the len of input since the outpt_len include the input as well.
    """
    len_to_marginalize = tokenizer([to_marginalize], return_tensors="pt", padding=False, add_special_tokens=False).input_ids.shape[1]
    add_to_base = np.abs(shap_value[len_to_marginalize:]).sum(axis=0)
    ratios = shap_value / (np.abs(shap_value).sum(axis=0) - add_to_base)
    out = np.mean(ratios,axis=-1)[:len_to_marginalize]
    return out

def compute_cc_shap(tokenizer,values_prediction, values_explanation, marg_pred='', marg_expl=' Yes. Why?'):
    ratios_prediction = aggregate_values_explanation(values_prediction,tokenizer,marg_pred)
    ratios_explanation = aggregate_values_explanation(values_explanation,tokenizer,marg_expl)
    return cc_shap_score(ratios_prediction,ratios_explanation)
# , dist_correl, mse, var, kl_div, js_div,

def cc_shap_measure(tokenizer,explainer,output_args,is_chat,batch_size,answer_only=False):
    """
    We use the already computed explanation and answer to derive the SHAP values for output and expl separately.
    """
    question = output_args['question']
    choices = output_args['choices']
    ans = output_args['answer']
    expl = output_args.get('explanation','')
    output_prompt = format_mcq(question,choices,expl_=False,is_chat=is_chat)
    if is_chat:
        output_prompt = format_input(output_prompt,tokenizer)
        output_prompt += "The best answer is: ("
    marg_pred = output_prompt.split('\n\nAnswer:')[0]
    output_shap = explainer([output_prompt],[ans],batch_size = batch_size).values[0]
    if answer_only:
        shap_agg =  aggregate_values_explanation(output_shap,tokenizer,marg_pred)
        sub_position = find_token_range(tokenizer,tokenizer.encode(marg_pred,add_special_tokens=False),output_args['subject'],include_chat_template=is_chat)
        return shap_agg,sub_position

    expl_prompt= format_mcq(question,choices,answer = ans,expl_=True,is_chat=is_chat)
    if is_chat:
        expl_prompt = format_input(expl_prompt,tokenizer) + 'Because' 
    else:
        expl_prompt = expl_prompt + ' Because' 
    expl_shap = explainer([expl_prompt],[expl],batch_size = batch_size).values[0]
     

    cosine,valid = compute_cc_shap(tokenizer,output_shap, expl_shap, marg_pred=marg_pred, marg_expl=marg_pred)
    return 1. - cosine ,valid

def run_cc_shap(mt,ds,args,pred_dir):
    save_path = os.path.join(pred_dir,f'ccshap.pkl') if not args.ood_analysis else os.path.join(pred_dir,f'ccshap_ood.pkl')
    mt.model.generation_config.is_decoder = True
    mt.model.config.is_decoder = True

    if not os.path.exists(save_path):
        generate_test = True
        out =  {}
    else:
        with open(save_path,'rb') as f:
            out = pickle.load(f)
        existing_ids = set(out.keys())
        ds = [d for d in ds if d['sample_id'] not in existing_ids]
        if len(ds) > 0:
            generate_test = True
        else:
            generate_test = False
            exit (f'Already computed {save_path}')

    if generate_test:
        teacher_forcing_model = shap.models.TeacherForcing(mt.model, mt.tokenizer,batch_size = 512)
        masker = shap.maskers.Text(mt.tokenizer, mask_token="...", collapse_mask_token=True)
        explainer = shap.Explainer(teacher_forcing_model, masker,silent=True)
        # for batch_id in tqdm()
        for d in tqdm(ds,total = len(ds),desc = f'Running CC-SHAP for {args.model_name}'):
            sample_id = d['sample_id']
            output_args = {}
            output_args['choices'] = join_choices(d['choices'])

            if not args.ood_analysis:
                output_args['answer'] = d['pred']
                output_args['question'] = d['question']
                output_args['explanation'] = d['explanation']
                if d['explanation'] == "":
                    continue
                cos_similarity = cc_shap_measure(mt.tokenizer,explainer,output_args,is_chat=mt.is_chat,batch_size = args.batch_size)
                if cos_similarity is not None:
                    out[sample_id] = cos_similarity # add a valid check here
            else:
                ood_sample = {}
                for d_type in ['original','cf']:
                    if d_type == 'original':
                        output_args['question'] = d['question']
                        output_args['answer'] = d['answer']
                        output_args['subject'] = d['subject']
                    else:
                        output_args['question'] = d['cf_question']
                        output_args['answer'] = d['cf_answer']
                        output_args['subject'] = d['cf_subject']
                    cos_similarity = cc_shap_measure(mt.tokenizer,explainer,output_args,is_chat=mt.is_chat,batch_size = args.batch_size,answer_only=True)
                    ood_sample[d_type] = cos_similarity
                out[sample_id] = ood_sample

        with open(save_path,'wb') as f:
            pickle.dump(out,f)


    


