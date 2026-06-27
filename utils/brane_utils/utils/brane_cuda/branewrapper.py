import os
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import branecuda

class BraneCUDA(Function):
   
        @staticmethod
        def forward(ctx, geom_params, coords, colors, rendered_img, max_deg_n, max_deg_m, dmax, use_rmax):
            ctx.save_for_backward(geom_params, coords, colors)
            ctx.dmax = dmax
            ctx.max_deg_n = max_deg_n
            ctx.max_deg_m = max_deg_m
            ctx.use_rmax = use_rmax
            
            h, w, c = rendered_img.shape
            s = geom_params.shape[0]
            
            branecuda.brane_render(geom_params, coords, colors, rendered_img, s, h, w, max_deg_n, max_deg_m, dmax, use_rmax)
            return rendered_img

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            geom_params, coords, colors = ctx.saved_tensors
            dmax = ctx.dmax
            
            h, w, c = grad_output.shape
            s = geom_params.shape[0]
            
            grads_geom = torch.zeros_like(geom_params)
            grads_coords = torch.zeros_like(coords)
            grads_colors = torch.zeros_like(colors)
            
            branecuda.brane_render_backward(
                geom_params, coords, colors, grad_output.contiguous(), 
                grads_geom, grads_coords, grads_colors, 
                s, h, w, ctx.max_deg_n, ctx.max_deg_m, dmax, ctx.use_rmax
            )
            return (grads_geom, grads_coords, grads_colors, None, None, None, None, None)

def resonant_brane_render(geom_params, coords, colors, image_size, max_deg_n=2, max_deg_m=2, dmax=100, use_rmax=True):
    geom_params = geom_params.contiguous() # (brane num, 3) -> [sigma_x, sigma_y, theta]
    coords = coords.contiguous()           # (brane num, 2)
    colors = colors.contiguous()           # (brane num, num_modi * 3)
    
    h, w = image_size[:2]
    c = 3 # RGB
    rendered_img = torch.zeros(h, w, c).to(colors.device).to(torch.float32)
    
    return BraneCUDA.apply(geom_params, coords, colors, rendered_img, max_deg_n, max_deg_m, dmax, use_rmax)