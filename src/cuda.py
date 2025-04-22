import torch

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"{n_gpu} CUDA GPU(s) available:")
    for i in range(n_gpu):
        print(f"  • GPU #{i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPU available.")
