#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <thread>
#include <vector>

namespace py = pybind11;

template<typename T>
void parallel_vector_add(T* a, T* b, T* result, size_t size, size_t num_threads) {
    std::vector<std::thread> threads;
    size_t chunk_size = size / num_threads;
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
        
        threads.emplace_back([=]() {
            for (size_t j = start; j < end; ++j) {
                result[j] = a[j] + b[j];
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

template<typename T>
void parallel_matrix_multiply(T* a, T* b, T* result, size_t M, size_t N, size_t K, size_t num_threads) {
    std::vector<std::thread> threads;
    size_t chunk_size = M / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_row = t * chunk_size;
        size_t end_row = (t == num_threads - 1) ? M : (t + 1) * chunk_size;
        
        threads.emplace_back([=]() {
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k) {
                        sum += a[i * K + k] * b[k * N + j];
                    }
                    result[i * N + j] = sum;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

py::array_t<float> cpp_vector_add(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    if (buf_a.size != buf_b.size)
        throw std::runtime_error("Input shapes must match");
    
    auto result = py::array_t<float>(buf_a.size);
    auto buf_result = result.request();
    
    size_t num_threads = std::thread::hardware_concurrency();
    parallel_vector_add(
        static_cast<float*>(buf_a.ptr),
        static_cast<float*>(buf_b.ptr),
        static_cast<float*>(buf_result.ptr),
        buf_a.size,
        num_threads
    );
    
    return result;
}

py::array_t<float> cpp_matrix_multiply(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    if (buf_a.ndim != 2 || buf_b.ndim != 2)
        throw std::runtime_error("Inputs must be 2D arrays");
    
    size_t M = buf_a.shape[0];
    size_t K = buf_a.shape[1];
    
    if (buf_b.shape[0] != K)
        throw std::runtime_error("Inner dimensions must match");
    
    size_t N = buf_b.shape[1];
    
    auto result = py::array_t<float>({M, N});
    auto buf_result = result.request();
    
    size_t num_threads = std::thread::hardware_concurrency();
    parallel_matrix_multiply(
        static_cast<float*>(buf_a.ptr),
        static_cast<float*>(buf_b.ptr),
        static_cast<float*>(buf_result.ptr),
        M, N, K,
        num_threads
    );
    
    return result;
}

PYBIND11_MODULE(core_ops, m) {
    m.doc() = "C++ optimized core operations"; 
    m.def("vector_add", &cpp_vector_add, "Optimized vector addition");
    m.def("matrix_multiply", &cpp_matrix_multiply, "Optimized matrix multiplication");
} 