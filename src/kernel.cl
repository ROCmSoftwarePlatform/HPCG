__kernel void SYMGS(__global double *matrixValues, __global int *mtxIndL,
                    __global char *nonzerosInRow, __global double *matrixDiagonal,
                    __global double *rv, __global double *xv, int offset) {
  int idx = get_global_id(0);
  int currentNumberOfNonzeros = nonzerosInRow[idx];
  double sum = rv[idx];
  for (int j = 0; j < currentNumberOfNonzeros; j++) {
    int curCol = mtxIndL[idx * 27 + j];
    sum -= matrixValues[idx * 27 + j] * xv[curCol];
    //sum -= matrixValues[idx * 27 + j];
  }

  sum += xv[idx + offset] * matrixDiagonal[idx];
  xv[idx + offset] = sum / matrixDiagonal[idx];
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


