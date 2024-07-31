import os
import pickle
from compute_scores import find_common_sample,compute_divg
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_scatter(points,expl_type):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    corr,p_value = pearsonr(x,y)
    # fig = plt.figure(figsize=(12,12))
    title_name = f"Faith-plaus for {expl_type}"
    plt.title(title_name)
    plt.scatter(x,y)
    plt.ylabel('Plausibility')
    plt.xlabel('Faithfulness')
    plt.savefig(f'plots/faith_plaus.png')
    return corr,p_value


def main():
    dataset_name = 'csqa'
    # model_name = 'llama3-8B-chat'
    expl_type='post_hoc'
    
    combined = []
    
    for model_name in ['llama3-8B-chat','llama3-8B']:
        all_scores = {'plaus': defaultdict(list),'faithful': defaultdict(list)}
        for seed in range(3):
            prediction_dir = f'prediction/{model_name}/{dataset_name}/{seed}'
            plaus_path = os.path.join(prediction_dir,f'plaus_{dataset_name}_{expl_type}_{seed}.pkl')
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
            combined.append((faithful,plaus))
    
    corr,p_value = plot_scatter(combined,expl_type)
    print (f"Correlation: {corr:.3f}, p-value: {p_value}")

    


if __name__ == '__main__':
    main()

