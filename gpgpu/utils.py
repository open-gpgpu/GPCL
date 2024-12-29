import numpy as np
import pyopencl as cl
from typing import Tuple
from numba import jit, prange
from .core import GPGPUContext
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from . import core_ops
    HAS_CPP_OPS = True
except ImportError:
    HAS_CPP_OPS = False
    logger.warning("C++ extensions not available. Using Python fallbacks.")

def load_kernel(filename: str) -> str:
    """Load OpenCL kernel from file."""
    kernel_path = os.path.join(os.path.dirname(__file__), 'kernels', filename)
    with open(kernel_path, 'r') as f:
        return f.read()

@jit(nopython=True, parallel=True)
def _cpu_vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled CPU fallback for vector addition."""
    result = np.empty_like(a)
    for i in prange(len(a)):
        result[i] = a[i] + b[i]
    return result

@jit(nopython=True, parallel=True)
def _cpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled CPU fallback for matrix multiplication."""
    M, K = a.shape
    K, N = b.shape
    result = np.zeros((M, N), dtype=a.dtype)
    for i in prange(M):
        for j in range(N):
            temp = 0.0
            for k in range(K):
                temp += a[i, k] * b[k, j]
            result[i, j] = temp
    return result

@jit(nopython=True, parallel=True)
def _cpu_vector_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled CPU fallback for element-wise vector multiplication."""
    result = np.empty_like(a)
    for i in prange(len(a)):
        result[i] = a[i] * b[i]
    return result

@jit(nopython=True)
def _cpu_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """JIT-compiled CPU fallback for dot product."""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@jit(nopython=True, parallel=True)
def _cpu_matrix_transpose(a: np.ndarray) -> np.ndarray:
    """JIT-compiled CPU fallback for matrix transpose."""
    rows, cols = a.shape
    result = np.empty((cols, rows), dtype=a.dtype)
    for i in prange(rows):
        for j in range(cols):
            result[j, i] = a[i, j]
    return result

class GPGPUOperations:
    """Common GPGPU operations using pre-defined kernels."""
    
    def __init__(self, context: GPGPUContext = None, prefer_cpp: bool = True):
        """
        Initialize with a GPGPUContext.
        
        Args:
            context (GPGPUContext, optional): GPGPU context to use
            prefer_cpp (bool): Whether to prefer C++ implementations over Numba when GPU is not available
        """
        self.context = context or GPGPUContext()
        self._compile_kernels()
        self.use_gpu = True
        self.prefer_cpp = prefer_cpp and HAS_CPP_OPS

    def _compile_kernels(self):
        """Compile the pre-defined kernels."""
        try:
            self.vector_add_program = self.context.compile_kernel(load_kernel('vector_add.cl'))
            self.matrix_multiply_program = self.context.compile_kernel(load_kernel('matrix_multiply.cl'))
            self.vector_multiply_program = self.context.compile_kernel(load_kernel('vector_multiply.cl'))
            self.dot_product_program = self.context.compile_kernel(load_kernel('dot_product.cl'))
            self.matrix_transpose_program = self.context.compile_kernel(load_kernel('matrix_transpose.cl'))
        except Exception as e:
            logger.warning(f"Failed to compile GPU kernels: {str(e)}. Falling back to CPU execution.")
            self.use_gpu = False
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two vectors."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
            
        if not self.use_gpu:
            if self.prefer_cpp and HAS_CPP_OPS:
                try:
                    return core_ops.vector_add(a, b)
                except Exception as e:
                    logger.warning(f"C++ execution failed: {str(e)}. Falling back to Numba.")
            return _cpu_vector_add(a, b)
            
        try:
            a_buf = self.context.to_device(a)
            b_buf = self.context.to_device(b)
            result_buf = self.context.to_device(np.zeros_like(a))

            self.context.execute_kernel(
                self.vector_add_program.vector_add,
                (a.shape[0],),
                None,
                a_buf, b_buf, result_buf
            )

            return self.context.from_device(result_buf, a.shape, a.dtype)
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Trying C++ or CPU fallback.")
            if self.prefer_cpp and HAS_CPP_OPS:
                try:
                    return core_ops.vector_add(a, b)
                except Exception as e2:
                    logger.warning(f"C++ execution failed: {str(e2)}. Falling back to Numba.")
            return _cpu_vector_add(a, b)
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two matrices."""
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Inputs must be 2D matrices")
        if a.shape[1] != b.shape[0]:
            raise ValueError("Matrix dimensions must match for multiplication")
            
        if not self.use_gpu:
            if self.prefer_cpp and HAS_CPP_OPS:
                try:
                    return core_ops.matrix_multiply(a, b)
                except Exception as e:
                    logger.warning(f"C++ execution failed: {str(e)}. Falling back to Numba.")
            return _cpu_matrix_multiply(a, b)
            
        try:
            M, K = a.shape
            K, N = b.shape
     
            a_buf = self.context.to_device(a)
            b_buf = self.context.to_device(b)
            result_buf = self.context.to_device(np.zeros((M, N), dtype=a.dtype))
            
            self.context.execute_kernel(
                self.matrix_multiply_program.matrix_multiply,
                (M, N),
                None,
                a_buf, b_buf, result_buf, np.int32(M), np.int32(N), np.int32(K)
            )
            
            return self.context.from_device(result_buf, (M, N), a.dtype)
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Trying C++ or CPU fallback.")
            if self.prefer_cpp and HAS_CPP_OPS:
                try:
                    return core_ops.matrix_multiply(a, b)
                except Exception as e2:
                    logger.warning(f"C++ execution failed: {str(e2)}. Falling back to Numba.")
            return _cpu_matrix_multiply(a, b)
    
    def vector_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two vectors element-wise."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
            
        if not self.use_gpu:
            return _cpu_vector_multiply(a, b)
            
        try:
            a_buf = self.context.to_device(a)
            b_buf = self.context.to_device(b)
            result_buf = self.context.to_device(np.zeros_like(a))
            
            self.context.execute_kernel(
                self.vector_multiply_program.vector_multiply,
                (a.shape[0],),
                None,
                a_buf, b_buf, result_buf
            )
            
            return self.context.from_device(result_buf, a.shape, a.dtype)
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Falling back to CPU.")
            return _cpu_vector_multiply(a, b)
    
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the dot product of two vectors."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
            
        if not self.use_gpu:
            return _cpu_dot_product(a, b)
            
        try:
            n = a.shape[0]
            a_buf = self.context.to_device(a)
            b_buf = self.context.to_device(b)
            result_buf = self.context.to_device(np.zeros(1, dtype=np.float32))
            
            self.context.execute_kernel(
                self.dot_product_program.dot_product,
                (1,),
                None,
                a_buf, b_buf, result_buf, np.int32(n)
            )
            
            result = self.context.from_device(result_buf, (1,), np.float32)
            return float(result[0])
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Falling back to CPU.")
            return _cpu_dot_product(a, b)
    
    def matrix_transpose(self, a: np.ndarray) -> np.ndarray:
        """Transpose a matrix."""
        if len(a.shape) != 2:
            raise ValueError("Input must be a 2D matrix")
            
        if not self.use_gpu:
            return _cpu_matrix_transpose(a)
            
        try:
            rows, cols = a.shape
            a_buf = self.context.to_device(a)
            result_buf = self.context.to_device(np.zeros((cols, rows), dtype=a.dtype))
            
            self.context.execute_kernel(
                self.matrix_transpose_program.matrix_transpose,
                (cols, rows),
                None,
                a_buf, result_buf, np.int32(rows), np.int32(cols)
            )
            
            return self.context.from_device(result_buf, (cols, rows), a.dtype)
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Falling back to CPU.")
            return _cpu_matrix_transpose(a) 