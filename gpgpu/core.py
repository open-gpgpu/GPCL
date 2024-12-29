import pyopencl as cl
import numpy as np
from typing import Optional, Union, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPGPUContext:
    """Main context manager for GPGPU operations."""
    
    def __init__(self, device_type: str = "GPU", platform_index: int = 0):
        """
        Initialize the GPGPU context.
        
        Args:
            device_type (str): Type of device to use ("GPU" or "CPU")
            platform_index (int): Index of the platform to use
        """
        self.device_type = getattr(cl.device_type, device_type.upper())
        self.platform_index = platform_index
        self._setup_context()
    
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
    
    def execute_kernel(self, kernel, global_size: Tuple, local_size: Optional[Tuple] = None, *args):
        """
        Execute a compiled OpenCL kernel.
        
        Args:
            kernel: OpenCL kernel to execute
            global_size (tuple): Global work size
            local_size (tuple, optional): Local work size
            *args: Kernel arguments
        """
        kernel(self.queue, global_size, local_size, *args)
        self.queue.finish() 