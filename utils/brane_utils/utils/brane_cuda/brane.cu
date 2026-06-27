#include <stdio.h>   
#include <cmath>

#define PI 3.1415926536
#define MAX_DEGREE 7

__global__ void _brane_render_cuda(
        const float *geom_params, // [sigma_x, sigma_y, theta]
        const float *coords,      // [x0, y0]
        const float *colors,      // [R_00, G_00, B_00, R_01, G_01, ... R_nm, G_nm, B_nm]
        float *rendered_img,
        const int s,              // number of Branes
        const int h, const int w,
        const int max_deg_n, const int max_deg_m,
        const float dmax,
        const bool use_rmax
    ){

    int curs = blockIdx.x * blockDim.x + threadIdx.x;
    if(curs >= s) return;

    float sigma_x = geom_params[curs * 3 + 0];
    float sigma_y = geom_params[curs * 3 + 1];
    float theta   = geom_params[curs * 3 + 2];
    float x0 = coords[curs * 2 + 0];
    float y0 = coords[curs * 2 + 1];

    float cos_t = cos(theta);
    float sin_t = sin(theta);
    int num_modes = (max_deg_n + 1) * (max_deg_m + 1);

    float aspect_ratio = (float)w / (float)h;  

    //R_max logic
    int n_max_deg = max(max_deg_n, max_deg_m);
    float R_max_sq = 2.0f * n_max_deg + 1.0f;
    float R_cull_sq = R_max_sq + 22.0f;

    for(int hi=0; hi<h; hi++){
        float curh_f = 2.0f * hi / (h - 1) - 1.0f;
        float dy = curh_f - y0;
        if(dy > dmax || dy < -dmax) continue;

        for(int wi=0; wi<w; wi++){
            float curw_f = 2.0f * wi / (w - 1) - 1.0f;
            float dx = (curw_f - x0) * aspect_ratio;  
            // float dx = curw_f - x0;
            if(dx > dmax || dx < -dmax) continue;

            // Canonical Space x', y'
            float x_prime = (dx * cos_t + dy * sin_t) / sigma_x;
            float y_prime = (-dx * sin_t + dy * cos_t) / sigma_y;

            //R_max logic
            float r_sq = x_prime * x_prime + y_prime * y_prime;
            if (use_rmax && r_sq > R_cull_sq) continue;

            // Base Envelope
            float envelope = exp(-0.5f * (x_prime * x_prime + y_prime * y_prime));

            //just to understand when, using only the base envelop, we should discard the brane VS. using R_max.
            // if (use_rmax) {
            //     bool old_cut = (envelope < 1e-5f);
                
            //     if (old_cut) {
            //         if (n_max_deg > 0 && curs % 1000 == 0 && wi % 10 == 0 && hi % 10 == 0) {
            //             printf(">>> SALVATO! Brana %d | Grado: %d | r^2: %.2f | env: %.8f | R_cull_sq: %.8f\n", 
            //                    curs, n_max_deg, r_sq, envelope, R_cull_sq);
            //         }
            //     }
            // }


            // Local Vectors in Registers for Hermite Polynomials
            float Hx[MAX_DEGREE + 1];
            float Hy[MAX_DEGREE + 1];

            Hx[0] = 1.0f;
            if (max_deg_n > 0) Hx[1] = 2.0f * x_prime;
            for (int i = 1; i < max_deg_n; ++i) {
                Hx[i + 1] = 2.0f * x_prime * Hx[i] - 2.0f * i * Hx[i - 1];
            }

            Hy[0] = 1.0f;
            if (max_deg_m > 0) Hy[1] = 2.0f * y_prime;
            for (int j = 1; j < max_deg_m; ++j) {
                Hy[j + 1] = 2.0f * y_prime * Hy[j] - 2.0f * j * Hy[j - 1];
            }

            // Color Accumulation for Vibrant Modes
            float R = 0, G = 0, B = 0;
            int color_offset = curs * num_modes * 3;
            const float norm_factors[6] = {1.0f, 0.707106f, 0.353553f, 0.144337f, 0.051031f, 0.016137f};

            for (int n = 0; n <= max_deg_n; ++n) {
                for (int m = 0; m <= max_deg_m; ++m) {
                    
                    float mode_val = (Hx[n] * norm_factors[n]) * (Hy[m] * norm_factors[m]);

                    int idx = n * (max_deg_m + 1) + m;
                    
                    R += mode_val * colors[color_offset + idx * 3 + 0];
                    G += mode_val * colors[color_offset + idx * 3 + 1];
                    B += mode_val * colors[color_offset + idx * 3 + 2];
                }
            }

            // atomicAdd required: multiple branes can hit the same pixel
            atomicAdd(&rendered_img[(hi * w + wi) * 3 + 0], envelope * R);
            atomicAdd(&rendered_img[(hi * w + wi) * 3 + 1], envelope * G);
            atomicAdd(&rendered_img[(hi * w + wi) * 3 + 2], envelope * B);
        }
    }
}


__global__ void _brane_render_backward_cuda(
        const float *geom_params, const float *coords, const float *colors,
        const float *grads,
        float *grads_geom, float *grads_coords, float *grads_colors,
        const int s, const int h, const int w,
        const int max_deg_n, const int max_deg_m, const float dmax,
        const bool use_rmax
    ){

    int curs = blockIdx.x * blockDim.x + threadIdx.x;
    if(curs >= s) return;

    float sigma_x = geom_params[curs * 3 + 0];
    float sigma_y = geom_params[curs * 3 + 1];
    float theta   = geom_params[curs * 3 + 2];
    float x0 = coords[curs * 2 + 0];
    float y0 = coords[curs * 2 + 1];

    float cos_t = cos(theta);
    float sin_t = sin(theta);
    int num_modes = (max_deg_n + 1) * (max_deg_m + 1);

    // Local accumulators for the geometric gradients of this specific Brane
    float grad_sigma_x_acc = 0.0f, grad_sigma_y_acc = 0.0f;
    float grad_theta_acc = 0.0f, grad_x0_acc = 0.0f, grad_y0_acc = 0.0f;

    float aspect_ratio = (float)w / (float)h;


    //R_max logic
    int n_max_deg = max(max_deg_n, max_deg_m);
    float R_max_sq = 2.0f * n_max_deg + 1.0f;
    float R_cull_sq = R_max_sq + 22.0f;


    for(int hi = 0; hi < h; hi++){
        for(int wi = 0; wi < w; wi++){
            float curw_f = 2.0f * wi / (w - 1) - 1.0f;
            float curh_f = 2.0f * hi / (h - 1) - 1.0f;
            
            float dx = (curw_f - x0) * aspect_ratio; 
            // float dx = curw_f - x0;
            float dy = curh_f - y0;
            if(dx > dmax || dx < -dmax || dy > dmax || dy < -dmax) continue;

            float x_prime = (dx * cos_t + dy * sin_t) / sigma_x;
            float y_prime = (-dx * sin_t + dy * cos_t) / sigma_y;

            //R_max logic
            float r_sq = x_prime * x_prime + y_prime * y_prime;
            if (use_rmax && r_sq > R_cull_sq) continue;

            float envelope = exp(-0.5f * (x_prime * x_prime + y_prime * y_prime));
            // if (use_rmax && envelope < 1e-5f) continue;

            float Hx[MAX_DEGREE + 1]; float Hy[MAX_DEGREE + 1];
            Hx[0] = 1.0f; if (max_deg_n > 0) Hx[1] = 2.0f * x_prime;
            for (int i = 1; i < max_deg_n; ++i) Hx[i + 1] = 2.0f * x_prime * Hx[i] - 2.0f * i * Hx[i - 1];
            Hy[0] = 1.0f; if (max_deg_m > 0) Hy[1] = 2.0f * y_prime;
            for (int j = 1; j < max_deg_m; ++j) Hy[j + 1] = 2.0f * y_prime * Hy[j] - 2.0f * j * Hy[j - 1];

            float R = 0, G = 0, B = 0;
            float dS_dx_r = 0, dS_dx_g = 0, dS_dx_b = 0;
            float dS_dy_r = 0, dS_dy_g = 0, dS_dy_b = 0;
            int color_offset = curs * num_modes * 3;

            float grad_img_r = grads[(hi * w + wi) * 3 + 0];
            float grad_img_g = grads[(hi * w + wi) * 3 + 1];
            float grad_img_b = grads[(hi * w + wi) * 3 + 2];

            const float norm_factors[6] = {1.0f, 0.707106f, 0.353553f, 0.144337f, 0.051031f, 0.016137f};

            for (int n = 0; n <= max_deg_n; ++n) {
                for (int m = 0; m <= max_deg_m; ++m) {

                    // Hermite normalized
                    float Hn_norm = Hx[n] * norm_factors[n];
                    float Hm_norm = Hy[m] * norm_factors[m];
                    
                    // Normalized derivative
                    float Hn_prime_norm = (n > 0) ? sqrtf(2.0f * n) * (Hx[n-1] * norm_factors[n-1]) : 0.0f;
                    float Hm_prime_norm = (m > 0) ? sqrtf(2.0f * m) * (Hy[m-1] * norm_factors[m-1]) : 0.0f;

                    int idx = n * (max_deg_m + 1) + m;
                    float c_r = colors[color_offset + idx * 3 + 0];
                    float c_g = colors[color_offset + idx * 3 + 1];
                    float c_b = colors[color_offset + idx * 3 + 2];

                    float mode_val = Hn_norm * Hm_norm;
                    

                    R += mode_val * c_r;
                    G += mode_val * c_g;
                    B += mode_val * c_b;

                    dS_dx_r += c_r * Hn_prime_norm * Hm_norm;
                    dS_dx_g += c_g * Hn_prime_norm * Hm_norm;
                    dS_dx_b += c_b * Hn_prime_norm * Hm_norm;

                    dS_dy_r += c_r * Hn_norm * Hm_prime_norm;
                    dS_dy_g += c_g * Hn_norm * Hm_prime_norm;
                    dS_dy_b += c_b * Hn_norm * Hm_prime_norm;

                    // Backprop for colors
                    grads_colors[color_offset + idx * 3 + 0] += envelope * mode_val * grad_img_r;
                    grads_colors[color_offset + idx * 3 + 1] += envelope * mode_val * grad_img_g;
                    grads_colors[color_offset + idx * 3 + 2] += envelope * mode_val * grad_img_b;
                }
            }

            // Calculating Gradients for x' and y'
            float dC_dx_prime_r = -x_prime * envelope * R + envelope * dS_dx_r;
            float dC_dx_prime_g = -x_prime * envelope * G + envelope * dS_dx_g;
            float dC_dx_prime_b = -x_prime * envelope * B + envelope * dS_dx_b;
            float grad_x_prime = grad_img_r * dC_dx_prime_r + grad_img_g * dC_dx_prime_g + grad_img_b * dC_dx_prime_b;

            float dC_dy_prime_r = -y_prime * envelope * R + envelope * dS_dy_r;
            float dC_dy_prime_g = -y_prime * envelope * G + envelope * dS_dy_g;
            float dC_dy_prime_b = -y_prime * envelope * B + envelope * dS_dy_b;
            float grad_y_prime = grad_img_r * dC_dy_prime_r + grad_img_g * dC_dy_prime_g + grad_img_b * dC_dy_prime_b;

            
            grad_sigma_x_acc += grad_x_prime * (-x_prime / sigma_x);
            grad_sigma_y_acc += grad_y_prime * (-y_prime / sigma_y);
            grad_theta_acc   += grad_x_prime * (y_prime * sigma_y / sigma_x) + grad_y_prime * (-x_prime * sigma_x / sigma_y);
           
            grad_x0_acc      += grad_x_prime * (-cos_t / sigma_x) * aspect_ratio + grad_y_prime * (sin_t / sigma_y) * aspect_ratio;
            
            grad_y0_acc      += grad_x_prime * (-sin_t / sigma_x) + grad_y_prime * (-cos_t / sigma_y);
        }
    }
    
    
    grads_geom[curs * 3 + 0] = grad_sigma_x_acc;
    grads_geom[curs * 3 + 1] = grad_sigma_y_acc;
    grads_geom[curs * 3 + 2] = grad_theta_acc;
    grads_coords[curs * 2 + 0] = grad_x0_acc;
    grads_coords[curs * 2 + 1] = grad_y0_acc;
}

void _brane_render(const float *geom_params, const float *coords, const float *colors, float *rendered_img, const int s, const int h, const int w, const int max_deg_n, const int max_deg_m, const float dmax, const bool use_rmax) {
    int threads = 64; dim3 grid((s + threads - 1) / threads); dim3 block(threads);
    _brane_render_cuda<<<grid, block>>>(geom_params, coords, colors, rendered_img, s, h, w, max_deg_n, max_deg_m, dmax, use_rmax);
}

void _brane_render_backward(const float *geom_params, const float *coords, const float *colors, const float *grads, float *grads_geom, float *grads_coords, float *grads_colors, const int s, const int h, const int w, const int max_deg_n, const int max_deg_m, const float dmax, const bool use_rmax) {
    int threads = 64; dim3 grid((s + threads - 1) / threads); dim3 block(threads);
    _brane_render_backward_cuda<<<grid, block>>>(geom_params, coords, colors, grads, grads_geom, grads_coords, grads_colors, s, h, w, max_deg_n, max_deg_m, dmax, use_rmax);
}