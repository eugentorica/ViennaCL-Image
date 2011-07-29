
__kernel void lower_triangular_substitute_inplace(
          __global const float * matrix,
          unsigned int matrix_rows,
          unsigned int matrix_cols,
          unsigned int matrix_internal_rows,
          unsigned int matrix_internal_cols,
          __global float * vector)
{
  float temp;
  for (int row = 0; row < matrix_rows; ++row)
  {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_global_id(0) == 0)
      vector[row] /= matrix[row+row*matrix_internal_cols];

    barrier(CLK_GLOBAL_MEM_FENCE);

    temp = vector[row];

    for  (int elim = row + get_global_id(0) + 1; elim < matrix_rows; elim += get_global_size(0))
      vector[elim] -= temp * matrix[elim * matrix_internal_cols + row];
  }
}


