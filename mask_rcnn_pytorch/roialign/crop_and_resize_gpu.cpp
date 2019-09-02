#include "crop_and_resize_gpu.h"
#include <torch/torch.h>
#include "cuda/crop_and_resize_kernel.h"

void crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,      // [y1, x1, y2, x2]
    at::Tensor box_index,  // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor crops) {
  assert(image.is_cuda());
  assert(boxes.is_cuda());
  assert(box_index.is_cuda());
  assert(crops.is_cuda());

  const int batch_size = image.size(0);
  const int depth = image.size(1);
  const int image_height = image.size(2);
  const int image_width = image.size(3);

  const int num_boxes = boxes.size(0);

  // init output space
  crops.resize_({num_boxes, depth, crop_height, crop_width});
  crops.zero_();

  CropAndResizeLaucher(
      image.contiguous().data_ptr<float>(), boxes.contiguous().data_ptr<float>(),
      box_index.contiguous().data_ptr<int>(), num_boxes, batch_size, image_height,
      image_width, crop_height, crop_width, depth, extrapolation_value,
      crops.data_ptr<float>());
}

void crop_and_resize_gpu_backward(
    at::Tensor grads,
    at::Tensor boxes,       // [y1, x1, y2, x2]
    at::Tensor box_index,   // range in [0, batch_size)
    at::Tensor grads_image  // resize to [bsize, c, hc, wc]
) {
  // shape
  const int batch_size = grads_image.size(0);
  const int depth = grads_image.size(1);
  const int image_height = grads_image.size(2);
  const int image_width = grads_image.size(3);

  const int num_boxes = grads.size(0);
  const int crop_height = grads.size(2);
  const int crop_width = grads.size(3);

  // init output space
  grads_image.zero_();

  CropAndResizeBackpropImageLaucher(
      grads.data_ptr<float>(), boxes.data_ptr<float>(), box_index.data_ptr<int>(),
      num_boxes, batch_size, image_height, image_width, crop_height, crop_width,
      depth, grads_image.data_ptr<float>());
}
