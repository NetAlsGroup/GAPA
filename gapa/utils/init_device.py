import os
import numpy as np
import subprocess


def init_device(world_size=None):
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], universal_newlines=True)
        gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
        if world_size is None:
            world_size = len(gpu_memory) // 2
        if world_size < 2 and len(gpu_memory) >= 1:
            world_size = len(gpu_memory)
        best_gpu_index = np.argsort(gpu_memory)[:world_size]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(t) for t in best_gpu_index])
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{best_gpu_index}'
        import torch

        device = f'cuda:{0}'
        torch.cuda.set_device(0)
        print(f"\nTotal device: {best_gpu_index}. Main process in device {best_gpu_index[0]}")
        print(f"World size is {world_size}. Setup in utils.init_device.py\n")
        return device, world_size
    except:
        device = 'cpu'
        print(f"\nDevice in cpu\n")
        return device, 0


def mutil_init_device(world_size=None):
    try:
        # result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        #                                  universal_newlines=True)
        # gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
        # if world_size is None:
        #     world_size = len(gpu_memory)
        # best_gpu_index = np.argsort(gpu_memory)[:world_size]
        # CUDA_VISIBLE_DEVICES=0
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(t) for t in best_gpu_index])
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{best_gpu_index}'
        import torch
        device = f'cuda:{0}'
        # torch.cuda.set_device(0)
        # print(f"\nTotal device: {best_gpu_index}. Main process in device {best_gpu_index[0]}")
        print(f"World size is {world_size}. Setup in utils.init_device.py\n")
        return device, world_size
    except:
        device = 'cpu'
        print(f"\nDevice in cpu\n")
        return device, 0

