import torch
print(torch.__version__)
print(torch.cuda.is_available())
#查看cuda版本
print(torch.version.cuda)
print(torch.cuda.device_count())