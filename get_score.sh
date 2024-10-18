model_name='gemma2-9B'
dataset_name='comve'

## Get scores ##
for model_name in gemma2-2B gemma2-2B-chat gemma2-9B gemma2-9B-chat gemma2-27B gemma2-27B-chat
do
  python score_utils/compute_scores.py \
  --dataset_name csqa esnli comve \
  --metric causal cf_edit ccshap \
  --model_name $model_name \
  --causal_type STR STR_10
done

## Plot ##
for model_name in gemma2-2B gemma2-2B-chat gemma2-9B gemma2-9B-chat gemma2-27B gemma2-27B-chat
do
  python score_utils/plot.py \
  --dataset_name $dataset_name \
  --model_name $model_name 
done

 
## OOD only for ComVE ## 

## OOD GN ## 
python score_utils/compute_scores.py \
--dataset_name $dataset_name \
--metric ood_GN \
--model_name 'gemma2-2B-chat' \
--causal_type STR STR_10

## OOD SHAP ##
python score_utils/compute_scores.py \
--dataset_name $dataset_name \
--metric ood \
--model_name 'gemma2-2B-chat' \
--causal_type STR STR_10



