import numpy as np
import logging
from gpgpu import GPGPUContext

device = GPGPUContext()

a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

result = device.vector_add(a, b)

print(result)