import triton
import triton.language as tl
import torch
import time

@triton.jit
def t_s_k(x_ptr,
          y_ptr,
          N0,
          BLOCK_SIZE: tl.constexpr):
    pid=tl.program_id(axis=0)
    block_start= pid* BLOCK_SIZE
    offsets= block_start+ tl.arange(0,BLOCK_SIZE)
    mask= offsets< N0
    x=tl.load(x_ptr+offsets,mask=mask)
    y= 1/ (1+tl.exp(-x))
    tl.store(y_ptr+offsets,y,mask=mask)


def t_s_k_h(x:torch.Tensor, BLOCK_SIZE=1024) ->torch.Tensor:
    y=torch.empty_like(x)
    N0=x.numel()
    grid= lambda meta:(triton.cdiv(N0,meta['BLOCK_SIZE']),)
    t_s_k[grid](x,y,N0,BLOCK_SIZE=BLOCK_SIZE)
    return y

def benchmark(func, *args, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / n_iters * 1000


if __name__=='__main__':
  N=1024*1024
  x=torch.randn(N, device='cuda', dtype=torch.float32)

  y_triton=t_s_k(x)
  leaky_relu = nn.Sigmoid()
  y_torch = leaky_relu(x)

  print(y_torch)
  print(y_triton)
  print(abs(y_torch-y_triton))