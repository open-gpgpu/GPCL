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