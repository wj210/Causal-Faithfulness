echo "Running job with CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
cuda_path="cuda_visible_devices.txt"
if [ -f $cuda_path ]; then
  export CUDA_VISIBLE_DEVICES=$(cat $cuda_path)
  num_gpu=$(cat "$cuda_path" | tr ', ' '\n' | grep -c '[0-9]')
  echo "num gpus = $num_gpu"
else
  echo "cuda_visible_devices.txt file not found."
fi

bs=32
num_samples=100
corrupted_samples=5
expl_type='post_hoc'
model_name='gemma-2B-chat'
metric=causal
faithfulness_type=input_output_p
dataset_name='csqa'

for noise_level in s3 
do
  for dataset_name in arc esnli csqa
  do
    python get_known_ds.py \
    --dataset_name $dataset_name \
    --num_samples $num_samples \
    --batch_size $bs \
    --corrupted_samples $corrupted_samples \
    --expl_type $expl_type \
    --model_name $model_name \
    --seed 0 \
    --noise_level $noise_level 
    
  done
done