#include "brane.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void brane_render(
        torch::Tensor &geom_params,
        torch::Tensor &coords,
        torch::Tensor &colors,
        torch::Tensor &rendered_img,
        const int s, const int h, const int w, 
        const int max_deg_n, const int max_deg_m, const float dmax,
        const bool use_rmax
        ){
      
        CHECK_INPUT(geom_params);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(rendered_img);

        const at::cuda::OptionalCUDAGuard device_guard(device_of(geom_params));

        _brane_render(
            (const float *) geom_params.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (float *) rendered_img.data_ptr(),
            s, h, w, max_deg_n, max_deg_m, dmax, use_rmax);
}

void brane_render_backward(
        torch::Tensor &geom_params,
        torch::Tensor &coords,
        torch::Tensor &colors,
        torch::Tensor &grads,
        torch::Tensor &grads_geom,
        torch::Tensor &grads_coords,
        torch::Tensor &grads_colors,
        const int s, const int h, const int w, 
        const int max_deg_n, const int max_deg_m, const float dmax,
        const bool use_rmax
        ){

        CHECK_INPUT(geom_params); CHECK_INPUT(coords); CHECK_INPUT(colors); CHECK_INPUT(grads);
        CHECK_INPUT(grads_geom); CHECK_INPUT(grads_coords); CHECK_INPUT(grads_colors);

        const at::cuda::OptionalCUDAGuard device_guard(device_of(geom_params));

        _brane_render_backward(
            (const float *) geom_params.data_ptr(),
            (const float *) coords.data_ptr(),
            (const float *) colors.data_ptr(),
            (const float *) grads.data_ptr(),
            (float *) grads_geom.data_ptr(),
            (float *) grads_coords.data_ptr(),
            (float *) grads_colors.data_ptr(),
            s, h, w, max_deg_n, max_deg_m, dmax, use_rmax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("brane_render", &brane_render, "CUDA forward wrapper for Resonant Branes");
        m.def("brane_render_backward", &brane_render_backward, "CUDA backward wrapper for Resonant Branes");
}