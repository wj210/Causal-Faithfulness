REVISION=main
max_input_length=3000
max_total_length=3500
port=8082 # set if using tgi
master_port=29490 # set if using tgi
mem_frac=0.9
num_seq=10 # set to 10 for sc-cot
model=meta-llama/Meta-Llama-3-70B-Instruct
sharded=true
requests=160
dtype=bfloat16
export CUDA_VISIBLE_DEVICES=1,2
num_gpu=2
# if using local
text-generation-launcher --model-id $model --num-shard $num_gpu --port $port --max-input-length $max_input_length --master-port $master_port --cuda-memory-fraction $mem_frac --max-best-of $num_seq --sharded $sharded --max-total-tokens $max_total_length --disable-custom-kernels --max-concurrent-requests $requests --dtype $dtype