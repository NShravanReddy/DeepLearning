import torch
import triton 
import triton.language as tl
DEVICE='cuda'
@triton.jit
def constant_add_kernal(x_ptr,
                        constant,
                        y_ptr,
                        N0:tl.constexpr,
                        BLOCK_SIZE:tl.constexpr):
  pid=tl.program_id(axis=0)
  block_start= pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0,BLOCK_SIZE)
  mask= offsets < N0
  x= tl.load(x_ptr+offsets,mask=mask)
  y= x+constant
  tl.store(y_ptr+offsets,y,mask=mask)


def constant_add_(x:torch.Tensor, constant:float) -> torch.Tensor:
  N0=x.numel()
  BLOCK_SIZE=N0
  y=torch.empty_like(x)
  grid=lambda meta: (1,)
  constant_add_kernal[grid](x,constant,y,N0,BLOCK_SIZE=BLOCK_SIZE)
  return y

if __name__=='__main__':
  N0=1024
  x=torch.arange(0,N0,device=DEVICE,dtype=torch.float32)
  constant=3.0
  y_torch=x+constant
  y_triton=constant_add_(x,constant)
  print((y_torch,y_triton))