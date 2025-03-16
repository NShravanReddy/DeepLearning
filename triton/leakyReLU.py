import triton
import triton.language as tl
import torch
import torch.nn as nn
DEVICE = 'cuda'


@triton.jit
def l_r_k(x_ptr,
          y_ptr,
          alpha,
          N0,
          BLOCK_SIZE:tl.constexpr):
  
  pid=tl.program_id(axis=0)
  block_start= BLOCK_SIZE * pid
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < N0
  x=tl.load(x_ptr+offsets,mask=mask)
  y=tl.maximum(x,alpha * x)
  tl.store(y_ptr+offsets,y,mask=mask)



def l_r_k_h(x:torch.Tensor,alpha:float=1, BLOCK_SIZE=1024)->torch.Tensor:
  y=torch.empty_like(x)
  N0=x.numel()
  grid= lambda meta :(triton.cdiv(N0,meta['BLOCK_SIZE']),)
  assert x.is_cuda and y.is_cuda
  l_r_k[grid](x,y,alpha,N0,BLOCK_SIZE=BLOCK_SIZE)
  return y,alpha

if __name__=='__main__':
  N=1024*1024
  x=torch.randn(N, device='cuda', dtype=torch.float32)

  y_triton,alpha=l_r_k_h(x)
  leaky_relu = nn.LeakyReLU(negative_slope=alpha)
  y_torch = leaky_relu(x)

  print(y_torch)
  print(y_triton)
  print(abs(y_torch-y_triton))