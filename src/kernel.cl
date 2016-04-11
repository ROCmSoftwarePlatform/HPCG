#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void SYMGS(__global double *Values, __global int *Col,
                    __global int *RowOff, __global double *matrixDiagonal,
                    __global double *rv, __global double *xv, int offset, int totalRow) {
  __local double temp1[64];
  __local double temp2[64];
  int idx = get_group_id(0);
  int ldx = get_local_id(0);
  int startIndex1 = RowOff[2 * idx + offset];
  int currentNumberOfNonzeros1 = RowOff[2 * idx + offset + 1] - startIndex1;
  int startIndex2 = RowOff[2 * idx + offset + 1];
  int currentNumberOfNonzeros2 = RowOff[2 * idx + offset + 2] - startIndex2;

  temp1[ldx] = 0.0;
  temp2[ldx] = 0.0;

  if (ldx < 32) {
    if (ldx < currentNumberOfNonzeros1) {
      int curCol = Col[startIndex1 + ldx];
      temp1[ldx] = Values[startIndex1 + ldx] * xv[curCol];
    }
  } else {
    if (((ldx - 32) < currentNumberOfNonzeros2) && ((2 * idx) != (totalRow - 1))) {
      int curCol = Col[startIndex2 + ldx - 32];
      temp2[ldx] = Values[startIndex2 + ldx - 32] * xv[curCol];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int local_size = 64;
  for (int step = local_size / 2; step > 0; step >>= 1) {
    if (ldx < step) {
      temp1[ldx] += temp1[ldx + step];
      temp2[ldx] += temp2[ldx + step];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (0 == ldx) {
    double sum1 = rv[2 * idx + offset];
    sum1 -= temp1[ldx];
    sum1 += xv[2 * idx + offset] * matrixDiagonal[(2 * idx + offset) * 27];
    xv[2 * idx + offset] = sum1 / matrixDiagonal[(2 * idx + offset) * 27];
    if (2 * idx != (totalRow - 1)) {
      double sum2 = rv[2 * idx + offset + 1];
      sum2 -= temp2[ldx];
      sum2 += xv[2 * idx + offset + 1] * matrixDiagonal[(2 * idx + offset + 1) * 27];
      xv[2 * idx + offset + 1] = sum2 / matrixDiagonal[(2 * idx + offset + 1) * 27];
    }
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

__kernel void rtzCopy( __global double *rtz,
                       __global double *oldrtz)
{
    //Get our global thread ID
    int id = get_global_id(0);
    //Make sure we do not go out of bounds
    if (!id)
    *oldrtz = *rtz;
}
__kernel void computeBeta( __global double *rtz,
                           __global double *oldrtz,
                           __global double *beta)
{
    //Get our global thread ID
    int id = get_global_id(0);
    //Make sure we do not go out of bounds
    if (!id)
      *beta = *rtz / *oldrtz;
}
__kernel void computeAlpha( __global double *rtz,
                            __global double *pAp,
                            __global double *alpha,
                            __global double *minusAlpha)
{
    //Get our global thread ID
    int id = get_global_id(0);
    //Make sure we do not go out of bounds
    if (!id)
    {
      *alpha = *rtz / *pAp;
      *minusAlpha = -(*alpha);
    }
}
