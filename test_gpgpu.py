import numpy as np
import unittest
import time
import statistics
from gpgpu import GPGPUOperations

def run_benchmark(func, *args, runs=5):
    """Run a benchmark test multiple times and return statistics."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }

class TestGPGPUOperations(unittest.TestCase):
    def setUp(self):
        """Set up GPGPUOperations instance before each test."""
        self.ops = GPGPUOperations()
        np.random.seed(42)  # For reproducible tests
        
    def test_vector_add(self):
        """Test vector addition operation."""
        # Test correctness with small vectors
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = a + b
        result = self.ops.vector_add(a, b)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Performance test with large vectors
        print("\nVector Addition Performance Test:")
        a_large = np.random.rand(1000000).astype(np.float32)
        b_large = np.random.rand(1000000).astype(np.float32)
        stats = run_benchmark(self.ops.vector_add, a_large, b_large)
        print(f"Mean: {stats['mean']:.6f}s ± {stats['stdev']:.6f}s")
        print(f"Range: [{stats['min']:.6f}s, {stats['max']:.6f}s]")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.ops.vector_add(np.array([1.0]), np.array([1.0, 2.0]))
    
    def test_vector_multiply(self):
        """Test element-wise vector multiplication."""
        # Test correctness with small vectors
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = a * b
        result = self.ops.vector_multiply(a, b)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Performance test with large vectors
        print("\nVector Multiplication Performance Test:")
        a_large = np.random.rand(1000000).astype(np.float32)
        b_large = np.random.rand(1000000).astype(np.float32)
        stats = run_benchmark(self.ops.vector_multiply, a_large, b_large)
        print(f"Mean: {stats['mean']:.6f}s ± {stats['stdev']:.6f}s")
        print(f"Range: [{stats['min']:.6f}s, {stats['max']:.6f}s]")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.ops.vector_multiply(np.array([1.0]), np.array([1.0, 2.0]))
    
    def test_dot_product(self):
        """Test vector dot product operation."""
        # Test correctness with small vectors
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = np.dot(a, b)
        result = self.ops.dot_product(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Performance test with large vectors
        print("\nDot Product Performance Test:")
        a_large = np.random.rand(1000000).astype(np.float32)
        b_large = np.random.rand(1000000).astype(np.float32)
        stats = run_benchmark(self.ops.dot_product, a_large, b_large)
        print(f"Mean: {stats['mean']:.6f}s ± {stats['stdev']:.6f}s")
        print(f"Range: [{stats['min']:.6f}s, {stats['max']:.6f}s]")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.ops.dot_product(np.array([1.0]), np.array([1.0, 2.0]))
    
    def test_matrix_multiply(self):
        """Test matrix multiplication operation."""
        # Test correctness with small matrices
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        expected = np.dot(a, b)
        result = self.ops.matrix_multiply(a, b)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Performance test with larger matrices
        print("\nMatrix Multiplication Performance Test:")
        a_large = np.random.rand(500, 500).astype(np.float32)
        b_large = np.random.rand(500, 500).astype(np.float32)
        stats = run_benchmark(self.ops.matrix_multiply, a_large, b_large)
        print(f"Mean: {stats['mean']:.6f}s ± {stats['stdev']:.6f}s")
        print(f"Range: [{stats['min']:.6f}s, {stats['max']:.6f}s]")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.ops.matrix_multiply(np.random.rand(3, 2), np.random.rand(3, 2))
    
    def test_matrix_transpose(self):
        """Test matrix transpose operation."""
        # Test correctness with small matrix
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected = a.T
        result = self.ops.matrix_transpose(a)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Performance test with large matrix
        print("\nMatrix Transpose Performance Test:")
        a_large = np.random.rand(1000, 1000).astype(np.float32)
        stats = run_benchmark(self.ops.matrix_transpose, a_large)
        print(f"Mean: {stats['mean']:.6f}s ± {stats['stdev']:.6f}s")
        print(f"Range: [{stats['min']:.6f}s, {stats['max']:.6f}s]")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.ops.matrix_transpose(np.array([1.0, 2.0, 3.0]))  # 1D array

if __name__ == '__main__':
    unittest.main() 