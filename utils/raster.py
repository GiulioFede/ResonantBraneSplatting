import torch
#import utils.plots as plot_lib 



def rendering_cuda_with_offsets(sigma_x, sigma_y, theta, coords, colours_with_alpha, mode_offsets, sr_size, step_size, max_deg_n, max_deg_m, device, dmax=1):
    from utils.brane_utils.utils.brane_cuda.branewrapper import resonant_brane_render
    
    # Let's define the geometric parameters [sigma_x_scaled, sigma_y_scaled, theta]
    geom_params = torch.cat([
        sigma_x / step_size * 2 / (sr_size[1] - 1), 
        sigma_y / step_size * 2 / (sr_size[0] - 1),  
        theta
    ], dim=-1).contiguous()
    
    # Image coordinates
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    
    
    final_image = resonant_brane_render(
        geom_params=geom_params, 
        coords=coords, 
        colors=colours_with_alpha,
        mode_offsets = mode_offsets, 
        image_size=sr_size, 
        max_deg_n=max_deg_n, 
        max_deg_m=max_deg_m, 
        dmax=dmax
    )
    
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image




def rendering_cuda(sigma_x, sigma_y, theta, coords, colours_with_alpha, sr_size, step_size, max_deg_n, max_deg_m, device, dmax=1, use_rmax=True):
    from utils.brane_utils.utils.brane_cuda.branewrapper import resonant_brane_render
    
    # Let's define the geometric parameters [sigma_x_scaled, sigma_y_scaled, theta]
    geom_params = torch.cat([
        sigma_x / step_size * 2 / (sr_size[0] - 1), 
        sigma_y / step_size * 2 / (sr_size[0] - 1),  
        theta
    ], dim=-1).contiguous()
    
    # Image coordinates
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    
    final_image = resonant_brane_render(
        geom_params=geom_params, 
        coords=coords, 
        colors=colours_with_alpha,
        image_size=sr_size, 
        max_deg_n=max_deg_n, 
        max_deg_m=max_deg_m, 
        dmax=dmax,
        use_rmax = use_rmax
    )
    
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image


def rasterize_image(brane_parameters, # Brane parameters (N, 18) --> [Sx, Sy, Theta, Alpha, Rbase,Gbase,Bbase, c0, c1, ... cn, X, Y]
                    grt, # resolution of the final image
                    scale, # scale factor ex. x2, x5,...
                    default_step_size=1.2,
                    brane_color_type="per-mode",
                    max_deg_n=2, # N_max
                    max_deg_m=2, # M_max
                    dmax=1,
                    damping_list=None,
                    predict_offset_for_each_mode = False,
                    inference = True,
                    sigma_scale = 1.0,
                    use_rmax=True,
                    relax_parameters = None): 
    
    brane_parameters = brane_parameters.float()
    if len(grt)==2:
        grt_resolution = grt
    else:    
        grt_resolution = grt.shape[-2:]
    final_scale = scale
    step_size = default_step_size / final_scale
    
    num_modi = (max_deg_n + 1) * (max_deg_m + 1)


    # We scale up to the maximum allowed size to avoid wave collisions
    if not (relax_parameters is not None and "sigma" in relax_parameters):
        max_sigma_limit = 1.0 / sigma_scale 
        # We train the network to predict sigma values proportional to the local cell
        sigma_x = (max_sigma_limit * 0.99999) * torch.sigmoid(brane_parameters[:, 0:1]) + 1e-6
        sigma_y = (max_sigma_limit * 0.99999) * torch.sigmoid(brane_parameters[:, 1:2]) + 1e-6
    else:
        sigma_x = torch.nn.functional.softplus(brane_parameters[:, 0:1]) + 1e-6
        sigma_y = torch.nn.functional.softplus(brane_parameters[:, 1:2]) + 1e-6
    
    # IMPORTANT: We multiply by pi to allow for full rotations
    theta = 0.999999 * torch.tanh(brane_parameters[:, 2:3]) * torch.pi 
    
    alpha = torch.sigmoid(brane_parameters[:, 3:4])

    mode_offsets = None

    if brane_color_type == "per-mode":
        colors_raw = brane_parameters[:, 4:4+(num_modi * 3)]

        # base color always between 0 and 1
        base_color = torch.sigmoid(colors_raw[:, 0:3])

        if damping_list is not None:
            damping_tensor = torch.tensor(damping_list, device=colors_raw.device).unsqueeze(0)
            mode_colors = torch.tanh(colors_raw[:, 3:]) * damping_tensor
        else:
            # Mode colors must be able to be negative in order to subtract color
            mode_colors = torch.tanh(colors_raw[:, 3:])

        colors = torch.cat([base_color, mode_colors], dim=-1) # (N, num_modes * 3)
        
        # We apply alpha to all colors (broadcasting (N, num_modes*3) * (N, 1))
        colors_with_alpha = colors * alpha 

        if predict_offset_for_each_mode:
            # Offsets are output from the network already perfectly scaled to the NDC [-1, 1] range. No extra steps required!
            mode_offsets = brane_parameters[:, -2-(num_modi * 2) : -2]
        
        coords = brane_parameters[:, -2:] * 2.0 - 1.0

    elif brane_color_type == "constant":
        colors = torch.sigmoid(brane_parameters[:, 4:7]) # (N, 3)
        coeffs = torch.tanh(brane_parameters[:, 7:7+num_modi]) # (N, num_modes)
        
        # We use expansion and multiplication in PyTorch to maintain the universal CUDA kernel
        colors_expanded = colors.unsqueeze(1) * coeffs.unsqueeze(-1) # (N, num_modi, 3)
        
        # Flatten to (N, num_modes * 3) e apply alpha
        colors_with_alpha = colors_expanded.view(-1, num_modi * 3) * alpha
        
        coords = brane_parameters[:, -2:] * 2.0 - 1.0


    '''
        Plotta la coverage of each brane
    '''
    #plot_lib.plot_brane_coverage(grt_resolution, max_deg_n, max_deg_m, sigma_x, sigma_y, step_size, theta, coords, alpha, final_scale, minimum_alpha_to_show=0.05, path_to_save="./da_eliminare/brane_coverage.png")


    if predict_offset_for_each_mode:
        # Call to CUDA rasterizer
        final_image = rendering_cuda_with_offsets(
            sigma_x=sigma_x, 
            sigma_y=sigma_y, 
            theta=theta, 
            coords=coords, 
            colours_with_alpha=colors_with_alpha,
            mode_offsets = mode_offsets, 
            sr_size=grt_resolution, 
            step_size=step_size, 
            max_deg_n=max_deg_n, 
            max_deg_m=max_deg_m, 
            device=sigma_x.device,
            dmax=dmax
        )
    else:
        # Call to cuda rasterizer
        final_image = rendering_cuda(
            sigma_x=sigma_x, 
            sigma_y=sigma_y, 
            theta=theta, 
            coords=coords, 
            colours_with_alpha=colors_with_alpha,
            sr_size=grt_resolution, 
            step_size=step_size, 
            max_deg_n=max_deg_n, 
            max_deg_m=max_deg_m, 
            device=sigma_x.device,
            dmax=dmax,
            use_rmax=use_rmax
        )

    if inference:
        return final_image, None
    else:
        return final_image, alpha


