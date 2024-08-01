import torch
import os
import time
import argparse


def occupy_gpu_memory(gpu_id, memory_to_occupy):
    # Set the device to the specified GPU
    device = torch.device(f'cuda:{gpu_id}')

    # Calculate the size of the tensor to allocate
    num_elements = (memory_to_occupy * 1024 * 1024) // 4  # Assuming 4 bytes per float32
    # Allocate a tensor of zeros on the GPU
    _ = torch.zeros(num_elements, dtype=torch.float32, device=device)

    # Wait indefinitely
    try:
        while True:
            time.sleep(1)  # Sleep for a short time to prevent high CPU usage
    except KeyboardInterrupt:
        print("Exiting and releasing GPU memory.")

if __name__ == "__main__":
    GPU_ID = 0  # Change this to the ID of the GPU you want to occupy
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_no',type = str,default = '0')
    parser.add_argument('--memory',type = int,default = 60000)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.device_no

    occupy_gpu_memory(GPU_ID, args.memory)