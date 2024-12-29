import numpy as np
from gpgpu import GPGPUOperations

ops = GPGPUOperations(prefer_cpp=True)

a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

result = ops.vector_add(a, b)

print(result)