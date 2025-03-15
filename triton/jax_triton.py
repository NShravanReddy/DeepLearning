import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.langugage as tl
import torch

@triton.jit
def a_k(
    x_ptr,
    y_ptr,
    output_ptr,
    block_size:tl.constexpr,
):
    pid=tl.program_id(axis=0)
    block_start=pid*block_size
    offsets=block_start + tl.arange(0,block_size)
    mask= offsets< 8
    x=tl.load(x_ptr+offsets,mask=mask)
    y=tl.load(x_ptr+offsets,mask=mask)
    output=x+y
    tl.store(output_ptr+offsets,output,mask=mask)

def a_k_h(x: jnp.ndarray, y:jnp.ndarray)->jnp.ndarray:
    output_shape=jax.ShapeDtypeStruct(shape=x.shape,dtype=x.dtype)
    BLOCK_SIZE=8
    assert x.is_cuda and y.is_cuda and output_shape.is_cuda
    grid= (triton.cdiv(x.size, BLOCK_SIZE), )
    return jt.triton_call(
    x,
    y,
    kernel=a_k,
    output_shape=output_shape, 
    grid=grid,
    BLOCK_SIZE=BLOCK_SIZE)

def main(unused_argv):
    x_val=jnp.arange(8)
    y_val=jnp.arange(8,16)
    print(a_k_h(x_val,y_val))
    print(jax.jit(a_k_h)(x_val,y_val))

if __name__ == "__main__":
  from absl import app
  app.run(main)