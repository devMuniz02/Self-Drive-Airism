import torch
import sys

print("="*40)
print("      CUDA DIAGNOSTIC TOOL      ")
print("="*40)

# 1. Python & PyTorch Version
print(f"Python Version:    {sys.version.split()[0]}")
print(f"PyTorch Version:   {torch.__version__}")
print(f"CUDA (in PyTorch): {torch.version.cuda}")

# 2. Availability Check
is_available = torch.cuda.is_available()
print(f"\nCUDA Available?    {'✅ YES' if is_available else '❌ NO'}")

if is_available:
    # 3. Device Details
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    print(f"GPU Count:         {torch.cuda.device_count()}")
    print(f"Current GPU:       {device_name} (ID: {device_id})")
    
    # 4. Actual Computation Test
    print("\nTesting GPU computation...", end="")
    try:
        # Create a tensor and move it to GPU
        x = torch.rand(5, 3)
        x = x.to("cuda")
        print(" SUCCESS! ✅")
        print(f"Tensor device is: {x.device}")
    except Exception as e:
        print(f" FAILED ❌")
        print(f"Error: {e}")
else:
    print("\n⚠️  Running on CPU only.")
    print("If you expected a GPU, check that you installed the 'cu' version of PyTorch")
    print("and that your NVIDIA drivers are up to date.")

print("="*40)