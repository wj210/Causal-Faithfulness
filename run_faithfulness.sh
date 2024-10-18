bs=64
num_samples=100
model_name='gemma-2B'
dataset_name='csqa'

python get_predictions.py \
--dataset_name $dataset_name \
--batch_size $bs \
--model_name $model_name \
--num_samples $num_samples \
--mode STR

python main.py \
--dataset_name $dataset_name \
--batch_size $bs \
--model_name $model_name \