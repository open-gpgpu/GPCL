# GPCL - GPU Computing Library

Easy-to-use GPGPU acceleration for Python. Accelerated using OpenCL, and C++.

## Usage

```python
from gpgpu.core import GPGPUContext

ctx = GPGPUContext(device_type="GPU")  # or "CPU"

result = ctx.vector_add(array1, array2)
```

## Available Operations

Each operation accepts NumPy arrays as input and returns NumPy arrays as output. All operations are executed on the GPU/accelerator device.

### Vector Operations
- `vector_add(a, b)`: Add two vectors element-wise
  - Input: Two 1D arrays of same shape
  - Output: 1D array of same shape
  - Example: `[1,2,3] + [4,5,6] = [5,7,9]`

- `vector_multiply(a, b)`: Multiply two vectors element-wise
  - Input: Two 1D arrays of same shape
  - Output: 1D array of same shape
  - Example: `[1,2,3] * [4,5,6] = [4,10,18]`

- `dot_product(a, b)`: Compute dot product of two vectors
  - Input: Two 1D arrays of same shape
  - Output: Scalar value
  - Example: `[1,2,3] · [4,5,6] = 32`

### Matrix Operations
- `matrix_multiply(a, b)`: Multiply two matrices
  - Input: Two 2D arrays where a.shape[1] == b.shape[0]
  - Output: 2D array of shape (a.shape[0], b.shape[1])
  - Example: 
    ```python
    [[1,2],   [[5,6],    [[19,22],
     [3,4]] *  [7,8]]  =  [43,50]]
    ```

- `matrix_transpose(a)`: Transpose a matrix
  - Input: 2D array
  - Output: 2D array with swapped dimensions
  - Example: 
    ```python
    [[1,2],    [[1,3],
     [3,4]] ->  [2,4]]
    ```

## Arguments

### GPGPUContext Constructor
- `device_type`: "GPU" or "CPU" (default: "GPU")
- `platform_index`: Index of OpenCL platform to use (default: 0)
- `log_level`: Python logging level (optional)

## Examples

### Basic Vector Operations
```python
import numpy as np
from gpgpu.core import GPGPUContext
import logging

# Create context with logging
ctx = GPGPUContext(log_level=logging.INFO)

# Vector addition
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([5, 6, 7, 8], dtype=np.float32)
result = ctx.vector_add(a, b)
print("Vector addition:", result)  # [6, 8, 10, 12]

# Vector multiplication
result = ctx.vector_multiply(a, b)
print("Vector multiplication:", result)  # [5, 12, 21, 32]

# Dot product
result = ctx.dot_product(a, b)
print("Dot product:", result)  # 70
```

### Matrix Operations
```python
# Matrix multiplication
matrix_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
matrix_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
result = ctx.matrix_multiply(matrix_a, matrix_b)
print("Matrix multiplication:")
print(result)
# [[19, 22],
#  [43, 50]]

# Matrix transpose
result = ctx.matrix_transpose(matrix_a)
print("Matrix transpose:")
print(result)
# [[1, 3],
#  [2, 4]]
```

## Performance Tips
1. Use `np.float32` instead of `np.float64` for better performance
2. Keep data on GPU if performing multiple operations
3. Use appropriate array sizes - very small arrays might be faster on CPU
4. Batch operations when possible instead of making many small calls

## Error Handling
Common errors and their solutions:
- `ValueError: Vectors must have the same shape`: Input arrays must match in dimensions
- `RuntimeError: No OpenCL platforms found`: OpenCL drivers not installed or detected
- `RuntimeError: Failed to initialize OpenCL context`: GPU/device not available or driver issues 