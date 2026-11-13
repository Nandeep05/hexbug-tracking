import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # optional: see if it detects 4060
