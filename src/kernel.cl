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
