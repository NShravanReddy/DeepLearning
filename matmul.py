import torch

tensor1=torch.randn(3,3)
tensor2=torch.randn(3,3)
print(torch.matmul(tensor1,tensor2).size())