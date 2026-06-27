#pragma once

void _brane_render(
        const float *geom_params, 
        const float *coords, 
        const float *colors, 
        float *rendered_img,
        const int s, 
        const int h, 
        const int w, 
        const int max_deg_n, 
        const int max_deg_m, 
        const float dmax,
        const bool use_rmax
);

void _brane_render_backward(
        const float *geom_params, 
        const float *coords, 
        const float *colors, 
        const float *grads,
        float *grads_geom, 
        float *grads_coords, 
        float *grads_colors,
        const int s, 
        const int h, 
        const int w, 
        const int max_deg_n, 
        const int max_deg_m, 
        const float dmax,
        const bool use_rmax
);