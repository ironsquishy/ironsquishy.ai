import torch
import gc
import time

def bytes_to_mb(x):
    return x / (1024 ** 2)

def print_cuda_stats(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"[{stage}] CUDA Memory Allocated: {bytes_to_mb(allocated):.2f} MB")
        print(f"[{stage}] CUDA Memory Reserved:  {bytes_to_mb(reserved):.2f} MB")
    else:
        print(f"[{stage}] CUDA not available")

def clear_cuda(verbose=True):
    if verbose:
        print("\n=== Starting CUDA Cleanup "
        "===")

    print_cuda_stats("BEFORE")

    # Step 1: Python garbage collection
    if verbose:
        print("\n[INFO] Running Python garbage collection...")
    gc.collect()
    time.sleep(0.5)

    # Step 2: Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        if verbose:
            print("[INFO] Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(0.5)
    else:
        print("[WARN] CUDA not available, skipping GPU cleanup")

    print_cuda_stats("AFTER")

    if verbose:
        print("=== CUDA Cleanup Complete ===\n")

# Example usage
if __name__ == "__main__":
    clear_cuda()