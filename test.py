import numpy as np
from gpgpu import GPGPUOperations

ops = GPGPUOperations()

# Create test data
a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

# Add vectors on GPU
result = ops.vector_add(a, b)

print(result)