from gpgpu import GPGPUContext
import numpy as np

ctx = GPGPUContext()

custom_kernel = """
__kernel void multiply_by_two(__global const float* input,
                             __global float* output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f; // Multiply by 2
}
"""

program = ctx.compile_kernel(custom_kernel)

data = np.random.rand(1000000).astype(np.float32)
input_buf = ctx.to_device(data)
output_buf = ctx.to_device(np.zeros_like(data))

ctx.execute_kernel(
    program.multiply_by_two,
    (data.shape[0],),
    None,
    input_buf, output_buf
)

result = ctx.from_device(output_buf, data.shape, data.dtype)

print(result)