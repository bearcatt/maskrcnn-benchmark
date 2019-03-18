// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


std::tuple<at::Tensor, at::Tensor> GlobalROIPool_forward(
                                const at::Tensor& input,
                                const int pooled_height,
                                const int pooled_width) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return GlobalROIPool_forward_cuda(input, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor GlobalROIPool_backward(const at::Tensor& grad,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return GlobalROIPool_backward_cuda(
        grad, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}



