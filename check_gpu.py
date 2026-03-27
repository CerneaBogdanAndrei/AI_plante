import torch
print(f"CUDA disponibil: {torch.cuda.is_available()}")
print(f"Placa detectată: {torch.cuda.get_device_name(0)}")