import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available ✅")
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name:", torch.cuda.get_device_name(i))
            print(f"GPU {i} memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2))
    else:
        print("CUDA is NOT available ❌")

if __name__ == "__main__":
    check_cuda()