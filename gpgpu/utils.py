import numpy as np
import pyopencl as cl
from typing import Tuple
from .core import GPGPUContext

# Common OpenCL kernel sources
VECTOR_ADD_KERNEL = """
__kernel void vector_add(__global const float* a,
                        __global const float* b,
                        __global float* result)
{
    int gid = get_global_id(0);
    result[gid] = a[gid] + b[gid];
}
"""

MATRIX_MULTIPLY_KERNEL = """
__kernel void matrix_multiply(__global const float* a,
                            __global const float* b,
                            __global float* result,
                            const int M, const int N, const int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += a[row * K + i] * b[i * N + col];
    }
    result[row * N + col] = sum;
}
"""

VECTOR_MULTIPLY_KERNEL = """
__kernel void vector_multiply(__global const float* a,
                            __global const float* b,
                            __global float* result)
{
    int gid = get_global_id(0);
    result[gid] = a[gid] * b[gid];
}
"""

DOT_PRODUCT_KERNEL = """
__kernel void dot_product(__global const float* a,
                         __global const float* b,
                         __global float* result,
                         const int n)
{
    int gid = get_global_id(0);
    float temp = 0.0f;
    
    if (gid == 0) {
        for (int i = 0; i < n; i++) {
            temp += a[i] * b[i];
        }
        result[0] = temp;
    }
}
"""

MATRIX_TRANSPOSE_KERNEL = """
__kernel void matrix_transpose(__global const float* input,
                             __global float* output,
                             const int rows,
                             const int cols)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    
    if (gid_x < cols && gid_y < rows) {
        output[gid_x * rows + gid_y] = input[gid_y * cols + gid_x];
    }
}
"""

class GPGPUOperations:
    """Common GPGPU operations using pre-defined kernels."""
    
    def __init__(self, context: GPGPUContext = None):
        """
        Initialize with a GPGPUContext.
        
        Args:
            context (GPGPUContext, optional): GPGPU context to use
        """
        self.context = context or GPGPUContext()
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile the pre-defined kernels."""
        self.vector_add_program = self.context.compile_kernel(VECTOR_ADD_KERNEL)
        self.matrix_multiply_program = self.context.compile_kernel(MATRIX_MULTIPLY_KERNEL)
        self.vector_multiply_program = self.context.compile_kernel(VECTOR_MULTIPLY_KERNEL)
        self.dot_product_program = self.context.compile_kernel(DOT_PRODUCT_KERNEL)
        self.matrix_transpose_program = self.context.compile_kernel(MATRIX_TRANSPOSE_KERNEL)
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Add two vectors on the GPU.
        
        Args:
            a (np.ndarray): First vector
            b (np.ndarray): Second vector
            
        Returns:
            np.ndarray: Result of addition
        """
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
            
        # Transfer data to device
        a_buf = self.context.to_device(a)
        b_buf = self.context.to_device(b)
        result_buf = self.context.to_device(np.zeros_like(a))
        
        # Execute kernel
        self.context.execute_kernel(
            self.vector_add_program.vector_add,
            (a.shape[0],),
            None,
            a_buf, b_buf, result_buf
        )
        
        # Get result
        return self.context.from_device(result_buf, a.shape, a.dtype)
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two matrices on the GPU.
        
        Args:
            a (np.ndarray): First matrix
            b (np.ndarray): Second matrix
            
        Returns:
            np.ndarray: Result of multiplication
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError("Matrix dimensions must match for multiplication")
            
        M, K = a.shape
        K, N = b.shape
        
        # Transfer data to device
        a_buf = self.context.to_device(a)
        b_buf = self.context.to_device(b)
        result_buf = self.context.to_device(np.zeros((M, N), dtype=a.dtype))
        
        # Execute kernel
        self.context.execute_kernel(
            self.matrix_multiply_program.matrix_multiply,
            (M, N),
            None,
            a_buf, b_buf, result_buf, np.int32(M), np.int32(N), np.int32(K)
        )
        
        # Get result
        return self.context.from_device(result_buf, (M, N), a.dtype)
    
    def vector_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two vectors element-wise on the GPU.
        
        Args:
            a (np.ndarray): First vector
            b (np.ndarray): Second vector
            
        Returns:
            np.ndarray: Result of element-wise multiplication
        """
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
            
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
    
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the dot product of two vectors on the GPU.
        
        Args:
            a (np.ndarray): First vector
            b (np.ndarray): Second vector
            
        Returns:
            float: Dot product result
        """
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        
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
    
    def matrix_transpose(self, a: np.ndarray) -> np.ndarray:
        """
        Transpose a matrix on the GPU.
        
        Args:
            a (np.ndarray): Input matrix
            
        Returns:
            np.ndarray: Transposed matrix
        """
        if len(a.shape) != 2:
            raise ValueError("Input must be a 2D matrix")
            
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