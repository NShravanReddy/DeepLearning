import torch
import triton
import triton.language as tl
import time
torch.cuda.empty_cache()
@triton.jit
def r_f_k(x_ptr,
          y_ptr,
          N:tl.constexpr,
          BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    block_start=pid*BLOCK_SIZE
    offsets= block_start+tl.arange(0,BLOCK_SIZE)
    mask = offsets<N
    x=tl.load(x_ptr+offsets,mask=mask)
    y=tl.maximum(x,0.0)
    tl.store(y_ptr+offsets,y,mask=mask)

@triton.jit
def r_b_k(x_ptr,
          grad_output_ptr,
          grad_input_ptr,
          N:tl.constexpr,
          BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    block_start= pid*BLOCK_SIZE
    offsets = block_start+tl.arange(0,BLOCK_SIZE)
    mask= offsets< N
    x=tl.load(x_ptr+offsets,mask=mask)
    grad_out=tl.load(grad_output_ptr+offsets,mask=mask)
    grad_in=tl.where(x>0,grad_out,0.0)
    tl.store(grad_input_ptr+offsets,grad_in,mask=mask)

class TRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x:torch.Tensor,BLOCK_SIZE: int=1024)->torch.Tensor:
        N=x.numel()
        y=torch.empty_like(x)
        grid=lambda meta: (triton.cdiv(N,meta['BLOCK_SIZE']),)
        r_f_k[grid](x,y,N,BLOCK_SIZE=BLOCK_SIZE)
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE=BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx,grad_output:torch.Tensor)->tuple:
        x,=ctx.saved_tensors
        N=x.numel()
        grad_input=torch.empty_like(x)
  
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        BLOCK_SIZE=ctx.BLOCK_SIZE
        r_b_k[grid](x,grad_output,grad_input,N,BLOCK_SIZE=BLOCK_SIZE)
        return grad_input, None

def t_r(x:torch.Tensor,BLOCK_SIZE:int =1024) -> torch.Tensor:
    return TRF.apply(x,BLOCK_SIZE)


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
    N=1024 * 1024
    torch.cuda.empty_cache()

    x = torch.randn(N, device='cuda', dtype=torch.float32,requires_grad=True)
    BLOCK_SIZE=1024
    y_triton = t_r(x, BLOCK_SIZE)
    loss_triton = y_triton.sum()
    loss_triton.backward()

    
    x_torch = x.detach().clone().requires_grad_()
    y_torch = torch.relu(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    triton_time = benchmark(lambda: triton_relu(x, BLOCK_SIZE))
    torch_time = benchmark(lambda: torch.relu(x))
    print(f"Average execution time (Forward Pass):")
    print(f"  Triton ReLU = {triton_time:.3f} ms")
    print(f"  PyTorch ReLU = {torch_time:.3f} ms")

