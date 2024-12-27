# GPGPU

A lightweight Python module for GPU computing using PyOpenCL. This module provides a simple, intuitive interface for common GPGPU operations while abstracting away the complexity of OpenCL boilerplate code.

## Features

- Simple, Pythonic interface for GPU computing
- Automatic device selection and management
- Easy memory transfer between host and device
- Pre-defined kernels for common operations
- Support for custom OpenCL kernels
- Seamless integration with NumPy

## Installation

```bash
pip install gpgpu
```

## Quick Start

Here's a simple example of vector addition using the GPU:

```python
import numpy as np
from gpgpu import GPGPUOperations

# Initialize the GPGPU operations
ops = GPGPUOperations()

# Create some test data
a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

# Add vectors on GPU
result = ops.vector_add(a, b)
```

## Matrix Multiplication Example

```python
import numpy as np
from gpgpu import GPGPUOperations

ops = GPGPUOperations()

# Create test matrices
a = np.random.rand(1000, 1000).astype(np.float32)
b = np.random.rand(1000, 1000).astype(np.float32)

# Multiply matrices on GPU
result = ops.matrix_multiply(a, b)
```

## Custom Kernels

You can also create and use custom OpenCL kernels:

```python
from gpgpu import GPGPUContext
import numpy as np

# Initialize context
ctx = GPGPUContext()

# Define custom kernel
custom_kernel = """
__kernel void custom_operation(__global const float* input,
                             __global float* output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}
"""

# Compile kernel
program = ctx.compile_kernel(custom_kernel)

# Prepare data
data = np.random.rand(1000000).astype(np.float32)
input_buf = ctx.to_device(data)
output_buf = ctx.to_device(np.zeros_like(data))

# Execute kernel
ctx.execute_kernel(
    program.custom_operation,
    global_size=(data.shape[0],),
    args=(input_buf, output_buf)
)

# Get result
result = ctx.from_device(output_buf, data.shape, data.dtype)
```

## Requirements

- Python 3.7+
- PyOpenCL
- NumPy

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 