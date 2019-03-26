// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void GlobalRoIPoolFForward(
    const int nthreads, const T* bottom_data,
    const int channels, const int height, const int width, 
    const int pooled_height, const int pooled_width, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int roi_height = height;
    int roi_width = width;
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T meanval = is_empty ? 0 : -FLT_MAX;

    const T* offset_bottom_data =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        meanval += offset_bottom_data[h * width + w];
      }
    }
    // T count = static_cast<T>(hend - hstart + 1) * static_cast<T>(wend - wstart + 1);
    top_data[index] = meanval; //  / count;
  }
}

template <typename T>
__global__ void GlobalRoIPoolFBackward(
    const int nthreads, const T* top_diff,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int bottom_offset = (n * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;

    int roi_height = height;
    int roi_width = width;
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);

    // T count = static_cast<T>(hend - hstart + 1) * static_cast<T>(wend - wstart + 1);
    // T grad = static_cast<T>(offset_top_diff[ph * pooled_width + pw]) / count;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        atomicAdd(
            offset_bottom_diff + h * width + w,
            static_cast<T>(offset_top_diff[ph * pooled_width + pw]));
      }
    }
  }
}

at::Tensor GlobalROIPool_forward_cuda(const at::Tensor& input,
                                      const int pooled_height,
                                      const int pooled_width) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

  auto batch_size = input.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({batch_size, channels, pooled_height, pooled_width}, input.options());
  auto output_size = batch_size * pooled_height * pooled_width * channels;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "GlobalROIPool_forward", [&] {
    GlobalRoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

at::Tensor GlobalROIPool_backward_cuda(const at::Tensor& grad,
                                       const int pooled_height,
                                       const int pooled_width,
                                       const int batch_size,
                                       const int channels,
                                       const int height,
                                       const int width) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  // TODO add more checks

  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "GlobalROIPool_backward", [&] {
    GlobalRoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         grad_input.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
