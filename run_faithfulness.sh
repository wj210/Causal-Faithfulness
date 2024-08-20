bs=8
num_samples=100
corrupted_samples=5
expl_type='post_hoc'
model_name='gemma-2B'
metric=causal
dataset_name='csqa'

python main.py \
--dataset_name $dataset_name \
--num_samples $num_samples \
--batch_size $bs \
--corrupted_samples $corrupted_samples \
--expl_type $expl_type \
--model_name $model_name \
--metric $metric \