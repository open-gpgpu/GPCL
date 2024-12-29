import pyopencl as cl
import numpy as np
from typing import Optional, Union, Tuple, List
import logging
import os

# Set up logger but don't set level - will be configured in class
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def load_kernel(filename: str) -> str:
    """Load OpenCL kernel from file."""
    kernel_path = os.path.join(os.path.dirname(__file__), 'kernels', filename)
    with open(kernel_path, 'r') as f:
        return f.read()

class GPGPUContext:
    """Main context manager for GPGPU operations."""
    
    def __init__(self, device_type: str = "GPU", platform_index: int = 0, log_level: Optional[int] = None):
        """
        Initialize the GPGPU context.
        
        Args:
            device_type (str): Type of device to use ("GPU" or "CPU")
            platform_index (int): Index of the platform to use
            log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                                     If None, logging remains silent.
        """
        if log_level is not None:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(log_level)

        self.device_type = getattr(cl.device_type, device_type.upper())
        self.platform_index = platform_index
        self._setup_context()
        self._compile_kernels()
    
    @classmethod
    def set_log_level(cls, level: int):
        """
        Set the logging level for the GPGPU context.
        
        Args:
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        """
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(handler)
    
    def _setup_context(self):
        """Set up OpenCL context and command queue."""
        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            platform = platforms[self.platform_index]

            devices = platform.get_devices(device_type=self.device_type)
            if not devices:
                logger.warning(f"No {self.device_type} devices found, falling back to CPU")
                devices = platform.get_devices(device_type=cl.device_type.CPU)
            
            self.device = devices[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            logger.info(f"Using device: {self.device.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCL context: {str(e)}")

    def _compile_kernels(self):
        """Compile all pre-defined kernels."""
        try:
            self.kernels = {
                'vector_add': self.compile_kernel(load_kernel('vector_add.cl')).vector_add,
                'matrix_multiply': self.compile_kernel(load_kernel('matrix_multiply.cl')).matrix_multiply,
                'vector_multiply': self.compile_kernel(load_kernel('vector_multiply.cl')).vector_multiply,
                'dot_product': self.compile_kernel(load_kernel('dot_product.cl')).dot_product,
                'matrix_transpose': self.compile_kernel(load_kernel('matrix_transpose.cl')).matrix_transpose
            }
        except Exception as e:
            logger.warning(f"Failed to compile GPU kernels: {str(e)}. Operations will not be available.")
            self.kernels = {}
    
    def compile_kernel(self, source: str) -> cl.Program:
        """
        Compile an OpenCL kernel from source.
        
        Args:
            source (str): OpenCL kernel source code
            
        Returns:
            cl.Program: Compiled OpenCL program
        """
        try:
            program = cl.Program(self.context, source).build()
            return program
        except cl.RuntimeError as e:
            raise RuntimeError(f"Kernel compilation failed: {str(e)}")
    
    def to_device(self, data: np.ndarray, dtype=None) -> cl.Buffer:
        """
        Transfer data to device memory.
        
        Args:
            data (np.ndarray): NumPy array to transfer
            dtype: Optional data type override
            
        Returns:
            cl.Buffer: OpenCL buffer object
        """
        if dtype is None:
            dtype = data.dtype
            
        data = np.asarray(data, dtype=dtype)
        mem_flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        return cl.Buffer(self.context, mem_flags, hostbuf=data)
    
    def from_device(self, buffer: cl.Buffer, shape: Tuple, dtype=np.float32) -> np.ndarray:
        """
        Transfer data from device memory to host.
        
        Args:
            buffer (cl.Buffer): OpenCL buffer to transfer
            shape (tuple): Shape of the output array
            dtype: Data type of the output array
            
        Returns:
            np.ndarray: NumPy array containing the data
        """
        result = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.queue, result, buffer)
        return result
    
    def execute_kernel(self, kernel_name: str, global_size: Tuple, local_size: Optional[Tuple] = None, *args):
        """
        Execute a compiled OpenCL kernel.
        
        Args:
            kernel_name (str): Name of the kernel to execute
            global_size (tuple): Global work size
            local_size (tuple, optional): Local work size
            *args: Kernel arguments
        """
        if kernel_name not in self.kernels:
            raise ValueError(f"Kernel {kernel_name} not found")
        
        kernel = self.kernels[kernel_name]
        kernel(self.queue, global_size, local_size, *args)
        self.queue.finish()

    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two vectors."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        
        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        result_buf = self.to_device(np.zeros_like(a))

        self.execute_kernel('vector_add', (a.shape[0],), None, a_buf, b_buf, result_buf)
        return self.from_device(result_buf, a.shape, a.dtype)

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two matrices."""
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Inputs must be 2D matrices")
        if a.shape[1] != b.shape[0]:
            raise ValueError("Matrix dimensions must match for multiplication")
        
        M, K = a.shape
        K, N = b.shape
        
        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        result_buf = self.to_device(np.zeros((M, N), dtype=a.dtype))
        
        self.execute_kernel('matrix_multiply', (M, N), None,
                          a_buf, b_buf, result_buf, np.int32(M), np.int32(N), np.int32(K))
        
        return self.from_device(result_buf, (M, N), a.dtype)

    def vector_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two vectors element-wise."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        
        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        result_buf = self.to_device(np.zeros_like(a))
        
        self.execute_kernel('vector_multiply', (a.shape[0],), None, a_buf, b_buf, result_buf)
        return self.from_device(result_buf, a.shape, a.dtype)

    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the dot product of two vectors."""
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        
        a_buf = self.to_device(a)
        b_buf = self.to_device(b)
        result_buf = self.to_device(np.zeros(1, dtype=np.float32))
        
        self.execute_kernel('dot_product', (1,), None, a_buf, b_buf, result_buf, np.int32(a.shape[0]))
        result = self.from_device(result_buf, (1,), np.float32)
        return float(result[0])

    def matrix_transpose(self, a: np.ndarray) -> np.ndarray:
        """Transpose a matrix."""
        if len(a.shape) != 2:
            raise ValueError("Input must be a 2D matrix")
        
        rows, cols = a.shape
        a_buf = self.to_device(a)
        result_buf = self.to_device(np.zeros((cols, rows), dtype=a.dtype))
        
        self.execute_kernel('matrix_transpose', (cols, rows), None,
                          a_buf, result_buf, np.int32(rows), np.int32(cols))
        
        return self.from_device(result_buf, (cols, rows), a.dtype) 