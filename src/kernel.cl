#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void SYMGS(__global double *Values, __global int *Col,
                    __global int *RowOff, __global double *matrixDiagonal,
                    __global double *rv, __global double *xv, int offset) {
  __local double temp[32];
  int idx = get_group_id(0);
  int ldx = get_local_id(0);
  int startIndex = RowOff[idx + offset];
  int currentNumberOfNonzeros = RowOff[idx + offset + 1] - startIndex;

  temp[ldx] = 0.0;
  if (ldx < currentNumberOfNonzeros) {
    int curCol = Col[startIndex + ldx];
    temp[ldx] = Values[startIndex + ldx] * xv[curCol];
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
    double sum = rv[idx + offset];
    sum -= temp[ldx];
    sum += xv[idx + offset] * matrixDiagonal[(idx + offset) * 27];
    xv[idx + offset] = sum / matrixDiagonal[(idx + offset) * 27];
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
