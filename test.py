import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
number = torch.cuda.device_count()
print(number)