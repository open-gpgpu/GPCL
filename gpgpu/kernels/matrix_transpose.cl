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