from matplotlib import pyplot as plt
import textwrap
import os
import numpy as np
from scipy import spatial
from scipy.stats import rankdata


def plot_trace_heatmap(result, savepdf=None,xlabel=None,type_ = 'diff_logit'):
    score = result[type_]
    answer = result["answer"]
    ori_input = result["input_tokens"][0]
    cf_input = result["input_tokens"][1]
    ori_range = [0,result["subject_range"][0][1]-result["subject_range"][0][0]]
    cf_range = [0,result["subject_range"][1][1]-result["subject_range"][1][0]]
    labels = list(ori_input)
    for i in range(*ori_range): # annotate the corrupted tokens
        labels[i] = labels[i] + "*"

    original_subject_tokens = [ori_input[i] for i in range(*ori_range)]
    corrupted_subject_tokens = [cf_input[i] for i in range(*cf_range)]

    # title_msg = "\n".join([f"Original tokens: {original_subject_tokens}", f"Corrupted tokens: {corrupted_subject_tokens}"])
    
    # Wrap the labels if they are too long
    max_label_length = 5  # Set your desired maximum label length
    wrapped_answer = "\n".join(textwrap.wrap(str(answer), width=200))

    with plt.rc_context(rc={"font.size": 8}):
        fig, ax = plt.subplots(figsize=(14,8), dpi=200)
        h = ax.pcolor(
            score,
            cmap="Purples",
            vmin=0., 
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(score))])
        ax.set_xticks([0.5 + i for i in range(0, score.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, score.shape[1] - 6, 5)),fontsize = 14)
        ax.set_yticklabels(labels,fontsize = 14)
        # Adjust tick parameters for better display
        ax.tick_params(axis='y', which='major', labelsize=14, labelrotation=0)
        # ax.set_xlabel(f"single restored layer",fontsize = 16)
        # cb = plt.colorbar(h)
        # ax.set_title(title_msg)
        if xlabel is not None:
            ax.set_xlabel(xlabel,fontsize = 16)
        # cb.ax.set_title('gen: '+wrapped_answer, y=-0.16, fontsize=14)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def plot_trace_barplot(result,save_path= None,type_ = 'layer'):
    """
    scores are of N,L (N - number of tokens, L - number of layers)
    if layer, average across tokens and vice versa
    """
    score = result['scores']
    answer = result['answer']
    plt.figure(figsize=(12,8))
    if type_ == 'layer':
        score = score.mean(axis = 0)
        labels = range(score.shape[0])
    else:
        score = score.mean(axis = 1)
        labels = list(result["input_tokens"])
        for i in range(*result["subject_range"]): # annotate the corrupted tokens
            labels[i] = labels[i] + "*"
    title = f'Average impact across {type_} for:\n{answer}'
    wrapped_answer = "\n".join(textwrap.wrap(title, width=150))
    plt.bar(labels,score)
    plt.xlabel(type_)
    plt.ylabel('IDE')
    plt.title(wrapped_answer,fontsize = 10)
    if type_ == 'token':
        plt.xticks(rotation = 45,ha = 'right')
    plt.savefig(save_path)
    plt.close()


def average_value_via_indices(array,indices_ranges,axis = 0):
    out = []
    assert axis in [0,1], "axis must be 0 or 1"
    for ir in indices_ranges:
        if axis == 0:
            out.append(np.mean([array[i] for i in range(*ir)],axis = 0))
        else:
            out.append(np.mean([array[:,i] for i in range(*ir)],axis = 1))
    return out

def plot_trace_barplot_joined(output_result,expl_result,answer_range,save_path= None,type_ = 'layer',value_type = 'logit'):
    """
    scores are of N,L (N - number of tokens, L - number of layers)
    if layer, average across tokens and vice versa
    """
    o_score = output_result[f'diff_{value_type}']
    answer = output_result['answer']

    e_score = expl_result['scores']
    expl = expl_result['answer']

    if type_ == 'layer': # we plot out line plot on both the subject and answer.
        ranges_to_get = [output_result['subject_range'],answer_range]
        o_score = average_value_via_indices(o_score,ranges_to_get) # a list of 2 list (subject and answer)
        e_score = average_value_via_indices(e_score,ranges_to_get)
        labels = range(o_score[0].shape[0])
        subj_dist,ans_dist = 0,0
        for i,(o,e) in enumerate(zip(o_score,e_score)):
            d = 1. - spatial.distance.cosine(o,e)
            if i == 0:
                subj_dist = d
            else:
                ans_dist = d
        title_str = f'Answer: {answer}, Subj dist: {subj_dist:.2f}, Ans dist: {ans_dist:.2f}'
    else:
        o_score = o_score.mean(axis = 1)
        e_score = e_score.mean(axis = 1)
        labels = list(output_result["input_tokens"])
        for i in range(*output_result["subject_range"]): # annotate the corrupted tokens
            labels[i] = labels[i] + "*"
        distance = 1. - spatial.distance.cosine(o_score,e_score)
        title_str = f'Answer: {answer}, Distance: {distance:.2f}'

    wrapped_title = "\n".join(textwrap.wrap(f"Pred: {expl}", width=150))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),sharex = True)  # 2 rows, 1 column

    if type_ == 'token':
        ax1.bar(labels,e_score)
        ax2.bar(labels,o_score)
    else:
        for i,s in enumerate(e_score):
            if i == 0:
                lab = 'subject'
            else:
                lab = 'answer'
            ax1.plot(labels,s,label = lab)
        for i,s in enumerate(o_score):
            if i == 0:
                lab = 'subject'
            else:
                lab = 'answer'
            ax2.plot(labels,s,label = lab)

    ax1.set_title(wrapped_title,fontsize = 10)
    ax1.set_ylabel('IDE')
    ax1.legend(fontsize = 14,loc = 'upper right')

    ax2.set_title(title_str,fontsize = 14)
    ax2.set_ylabel('IDE')

    if type_ == 'token':
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_overall_layer(o_list=[],expl_list=[],answer_ranges=[],save_path = None):
    """
    o_list,expl_list are lists of output and explanation results
    """
    total_subj_dist,total_ans_dist = 0,0
    total_rankings =  [np.zeros(expl_list[0]['scores'].shape[-1]) for _ in range(4)] # sort by output/expl subj then output/expl answer
    for o,e,ar in zip(o_list,expl_list,answer_ranges):
        o_score = o['scores']
        e_score = e['scores'][:o_score.shape[0]]
        ranges_to_get = [o['subject_range'],ar]
        o_score = average_value_via_indices(o_score,ranges_to_get) # a list of 2 list (subject and answer)
        e_score = average_value_via_indices(e_score,ranges_to_get)
        subj_dist,ans_dist = 0,0
        for i,(o,e) in enumerate(zip(o_score,e_score)):
            d = 1. - spatial.distance.cosine(o,e)
            if i == 0:
                subj_dist = d
            else:
                ans_dist = d
        total_subj_dist += subj_dist
        total_ans_dist += ans_dist
        combined_scores = [o_score[0],e_score[0],o_score[1],e_score[1]]
        for i,cs in enumerate(combined_scores):
            total_rankings[i] += rankdata(cs)
    
    total_rankings = [tr/len(o_list) for tr in total_rankings]
    total_subj_dist /= len(o_list)
    total_ans_dist /= len(o_list)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),sharex = True)  # 2 rows, 1 column

    bar_width = 0.4
    x_range = np.arange(total_rankings[0].shape[0])

    x_range1 = x_range - bar_width/2
    x_range2 = x_range + bar_width/2

    ax1.bar(x_range1,total_rankings[0],bar_width,label = 'answer',color = 'b',alpha = 0.6)
    ax1.bar(x_range2,total_rankings[1],bar_width,label = 'expl',color = 'r',alpha = 0.6)
    ax1.set_title(f'Average subject ranking: {total_subj_dist:.2f}')
    ax1.legend(fontsize = 14,loc = 'upper right')

    ax2.bar(x_range1,total_rankings[2],bar_width,label = 'answer',color = 'b',alpha = 0.6)
    ax2.bar(x_range2,total_rankings[3],bar_width,label = 'expl',color = 'r',alpha = 0.6)
    ax2.set_title(f'Average correct answer ranking: {total_ans_dist:.2f}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_category_barplot(categories,values,save_path,colors = None,legend_loc = 'upper center'):
    x = np.arange(len(categories))
    bar_width = 0.25
    fig, ax = plt.subplots()
    for i,(label_name,scores) in enumerate(values.items()):
        if not isinstance(scores[0],list) and not isinstance(scores[0],tuple):
            ax.bar(x+i*bar_width + (bar_width * (len(scores)-1)/ 2),scores,width = bar_width,label = label_name,color = colors[i])
        else:
            primary_scores = [s[0] for s in scores]
            secondary_scores = [s[1] for s in scores]
            ax.bar(x+i*bar_width + (bar_width * (len(scores)-1)/ 2),primary_scores,width = bar_width,label = label_name,color = colors[i])
            ax.bar(x+i*bar_width + (bar_width * (len(scores)-1)/ 2), secondary_scores, width=bar_width,color=colors[i], hatch='//', alpha=0.5)

    ax.set_xticks(x + bar_width)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(categories, fontsize=16)
    ax.legend(loc = legend_loc,fontsize=10)
    plt.savefig(save_path)