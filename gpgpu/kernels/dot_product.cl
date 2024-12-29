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