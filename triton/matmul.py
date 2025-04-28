import torch 
import triton 
import triton.language as tl

@triton.jit

def gemm_kernal(x_ptr,
               y_ptr,
               r_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr
               ):
    pid= tl.program_id(axis=0)
    block_start= pid * BLOCK_SIZE
    offsets= block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x=tl.load(x_ptr + offsets, mask =mask) #loading to SRAM/l2 cache
    y=tl.load(y_ptr + offsets, mask= mask) #loading to SRAM/l2 cache
    for i in range((x.shape[0])):
        for j in range((y.shape[1])):
            r[i][j] = (x[i] * y[:, j]).sum()




def gemm(x:torch.Tensor,y:torch.Tensor,r:torch.Tensor):
    output=torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid= lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    gemm_kernal[grid](x,y, output, n_elements, BLOCK_SIZE=1024)
    return output

    


x = torch.randn(10, 3, 4)
y = torch.randn(10, 4, 5)
r=torch.zeros(x.shape[0],y.shape[1])

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y=  torch.rand(size, device='cuda')
output_torch = x+y
output_triton = add(x,y)
print(output_torch)
print(output_triton)
print(f' The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch -output_triton))}')
