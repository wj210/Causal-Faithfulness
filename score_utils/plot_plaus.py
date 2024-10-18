import os
import pickle
from compute_scores import compute_divg
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def plot_scatter(points,dataset):
    all_x,all_y = [],[]
    # fig = plt.figure(figsize=(8,8))
    for model_name,points in points.items():
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x,y,label = model_name.split('gemma2-')[-1].strip())
        all_x.extend(x)
        all_y.extend(y)
    corr,p_value = pearsonr(all_x,all_y)
    m = LinearRegression().fit(np.array(all_x).reshape(-1,1),all_y)
    score = m.score(np.array(all_x).reshape(-1,1),all_y)
    title_name = f"Faithfulness - Plausibility, correlation: {corr:.3f}"
    # plt.title(title_name,fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylabel('Plausibility',fontsize=16)
    # plt.xlabel('Faithfulness',fontsize=16)
    plt.legend(loc = 'upper right')
    # plt.legend(bbox_to_anchor=(1., 1.25), loc='upper right',fontsize= 14)
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.75)
    plt.savefig(f'plots/faith_plaus_{dataset}.png')
    return corr,p_value,score


def main():
    dataset_name = 'esnli'
    combined = {}
    for model_name in ['gemma2-2B-chat','gemma2-2B','gemma2-9B-chat','gemma2-9B','gemma2-27B-chat','gemma2-27B']:
        all_scores = []
        prediction_dir = f'prediction/{model_name}/{dataset_name}/0'
        plaus_path = os.path.join(prediction_dir,f'plaus.pkl')
        faithful_path = os.path.join(prediction_dir,f'causal_STR.pkl')

        with open(plaus_path,'rb') as f:
            plaus_results = pickle.load(f)
        with open(faithful_path,'rb') as f:
            faith_results = pickle.load(f)

        for sample_id, plaus_score in plaus_results.items():
            ans,expl_score = faith_results.get(sample_id,(None,None))
            if ans is None:
                continue
            faith_score = compute_divg(ans['diff_prob'],expl_score['diff_prob'],compute_type='single')['single']
            all_scores.append((faith_score,plaus_score))
        
        print (f"Model: {model_name}, faith: {np.mean([s[0] for s in all_scores]):.2f}, plaus: {np.mean([s[1] for s in all_scores]):.2f}")
        combined[model_name] = all_scores
    
    corr,p_value,score = plot_scatter(combined,dataset_name)
    print ('Dataset: {}'.format(dataset_name))
    print (f"Correlation: {corr:.3f}, p-value: {p_value}, R^2: {score:.3f}")

    


if __name__ == '__main__':
    main()

