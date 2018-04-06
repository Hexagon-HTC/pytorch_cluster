#include "THCGraclus.h"

#include "common.cuh"
#include "THCDegree.cuh"
#include "THCColor.cuh"
#include "THCPropose.cuh"
#include "THCResponse.cuh"

void THCTensor_graclus(THCState *state, THCudaLongTensor *self, THCudaLongTensor *row,
                       THCudaLongTensor *col) {
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 3, self, row, col));

  int nNodes = THCudaLongTensor_nElement(state, self);
  THCudaLongTensor_fill(state, self, -1);

  THCudaLongTensor *prop = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_fill(state, prop, -1);

  THCudaLongTensor *degree = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_degree(state, degree, row);

  THCudaLongTensor *cumDegree = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_cumDegree(state, cumDegree, row);

  while(!THCTensor_color(state, self)) {
    THCTensor_propose(state, self, prop, row, col, degree, cumDegree);
    THCTensor_response(state, self, prop, row, col, degree, cumDegree);
  }

  THCudaLongTensor_free(state, prop);
  THCudaLongTensor_free(state, degree);
  THCudaLongTensor_free(state, cumDegree);
}

#include "generic/THCGraclus.cu"
#include "THC/THCGenerateAllTypes.h"
