import torch 
print("PyTorch version:", torch.__version__) 
print("CUDA available:", torch.cuda.is_available()) 
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU") 
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB") 
