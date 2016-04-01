__kernel void SYMGS(__global double *matrixValues, __global int *mtxIndL,
                    __global char *nonzerosInRow, __global double *matrixDiagonal,
                    __global double *rv, __global double *xv, int offset) {
  __local double temp[32];
  int idx = get_group_id(0);
  int ldx = get_local_id(0);
  int currentNumberOfNonzeros = nonzerosInRow[idx];

  temp[ldx] = 0.0;
  if (ldx < currentNumberOfNonzeros) {
    int curCol = mtxIndL[idx * 27 + ldx];
    temp[ldx] = matrixValues[idx * 27 + ldx] * xv[curCol];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int local_size = 32;
  for (int step = local_size / 2; step > 0; step >>= 1) {
    if (ldx < step) {
      temp[ldx] += temp[ldx + step];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (0 == ldx) {
    double sum = rv[idx];
    sum -= temp[ldx];
    sum += xv[idx + offset] * matrixDiagonal[idx];
    xv[idx + offset] = sum / matrixDiagonal[idx];
  }
}

__kernel void lubys_graph(int c, __global int *row_offset, __global int *col_index,
                          __global int *Colors, __global int *random) {
  int x = get_global_id(0);
  int flag = 1;
  if (Colors[x] == -1) {
    int ir = random[x];
    for (int k = row_offset[x]; k < row_offset[x + 1]; k++) {
      int j = col_index[k];
      int jc = Colors[j];
      if (((jc != -1) && (jc != c)) || (x == j)) {
        continue;
      }
      int jr = random[j];
      if (ir <= jr) {
        flag = 0;
      }
    }
    if (flag) {
      Colors[x] = c;
    }
  }
}
