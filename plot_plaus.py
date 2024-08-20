import os
import pickle
from compute_scores import find_common_sample,compute_divg
from collections import defaultdict
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
        plt.scatter(x,y,label = model_name)
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
    # plt.legend(loc = 'upper right')
    # plt.legend(bbox_to_anchor=(1., 1.25), loc='upper right',fontsize= 14)
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.75)
    plt.savefig(f'plots/faith_plaus_{dataset}.png')
    return corr,p_value,score


def main():
    dataset_name = 'esnli'
    # model_name = 'llama3-8B-chat'
    expl_type='post_hoc'
    
    combined = defaultdict(list)
    
    for model_name in ['llama3-8B-chat','llama3-8B','gemma-2B-chat','gemma2-27B-chat','gemma-2B']:
        all_scores = {'plaus': defaultdict(list),'faithful': defaultdict(list)}
        for seed in range(3):
            prediction_dir = f'prediction/{model_name}/{dataset_name}/{seed}'
            plaus_path = os.path.join(prediction_dir,f'{expl_type}_plaus.pkl')
            faithful_o_path = os.path.join(prediction_dir,f'{expl_type}_output_original.pkl')
            faithful_e_path = os.path.join(prediction_dir,f'{expl_type}_expl_original.pkl')

            sample_ids = find_common_sample(faithful_o_path,plaus_path)

            with open(plaus_path,'rb') as f:
                plaus_results = pickle.load(f)
            with open(faithful_o_path,'rb') as f:
                o_results = pickle.load(f)
            with open(faithful_e_path,'rb') as f:
                e_results = pickle.load(f)
            
            for sample_id in sample_ids:
                all_scores['plaus'][sample_id].append(plaus_results.get(sample_id,None))
                all_scores['faithful'][sample_id].append(compute_divg(o_results[sample_id]['scores'],e_results[sample_id]['scores']))

        
        for k,v in all_scores['plaus'].items():
            v = [x for x in v if x is not None]
            if len(v) != 3: # missing scores
                continue
            plaus = np.mean(v)
            faithful = np.mean(all_scores['faithful'][k])
            combined[model_name].append((faithful,plaus))
    
    corr,p_value,score = plot_scatter(combined,dataset_name)
    print ('Dataset: {}'.format(dataset_name))
    print (f"Correlation: {corr:.3f}, p-value: {p_value}, R^2: {score:.3f}")
    for model_name,scores in combined.items():
        print (f"{model_name}: {np.mean([s[1] for s in scores]):.3f}")

    


if __name__ == '__main__':
    main()

