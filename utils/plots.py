import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import utils.raster as rasterize_lib

def save_inference_grid_with_text(model, 
                                  lr_patch, 
                                  hr_patch, 
                                  scale, 
                                  brane_parameters,
                                  global_step, 
                                  image_dir, 
                                  max_samples, 
                                  dmax):
    
    # 1. Setup
    B = brane_parameters.shape[0]
    samples_to_plot = min(B, max_samples)
    target_res = round(lr_patch.shape[-1] * scale)

    # Adjust figsize: width = 4 * N, height = 4 * 3
    fig, axes = plt.subplots(3, samples_to_plot, figsize=(4 * samples_to_plot, 12))
    
    if samples_to_plot == 1:
        axes = axes.reshape(3, 1)

    h, w = hr_patch.shape[2], hr_patch.shape[3]
    # 2. Loop on samples
    for i in range(samples_to_plot):
        
        brane_p = brane_parameters[i]

        pred_img, _ = rasterize_lib.rasterize_image(brane_p,
                                   hr_patch,
                                   scale,
                                   model.default_step_size,
                                   brane_color_type=model.brane_color_type,
                                   max_deg_n=model.max_degree_n,
                                   max_deg_m=model.max_degree_m,
                                   dmax=dmax,
                                   damping_list=model.damping_list,
                                   predict_offset_for_each_mode=model.predict_offset_for_each_mode,
                                   sigma_scale = 1.0 if not model.scale_sigma_based_on_upk else 2**model.upk,
                                   relax_parameters=model.relax_parameters)
                                   #sigma_scale = 1.0 if model.upk is None else 2**model.upk)
        

        #(C, H, W) -> (H, W, C)
        pred_np = pred_img.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        # Plot Prediction (Row 0)
        ax_pred = axes[0, i]
        ax_pred.imshow(pred_np)
        # >>> Add textt <<<
        ax_pred.set_title(f"Prediction\nN={model.max_degree_n}, M={model.max_degree_m}", fontsize=12, color='darkblue', fontweight='bold')
        ax_pred.axis('off')

        # --- B. LR (Row 1) ---
        # We interpolate LR to the target resolution for display
        lr_up = F.interpolate(lr_patch[i:i+1], size=(target_res, target_res), mode='nearest')
        lr_np = lr_up.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        ax_lr = axes[1, i]
        ax_lr.imshow(lr_np)
        ax_lr.set_title("Low Resolution (NN)", fontsize=10)
        ax_lr.axis('off')

        # --- C. GROUND TRUTH (Row 3) ---
        hr_np = hr_patch[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        ax_hr = axes[2, i]
        ax_hr.imshow(hr_np)
        ax_hr.set_title("Ground Truth", fontsize=10)
        ax_hr.axis('off')

    # 3. Save
    plt.tight_layout()
    filename = os.path.join(image_dir, f"{global_step}_inference_x{scale}.png")
    plt.savefig(filename, dpi=100)
    plt.close(fig) 
    print(f"Saved inference grid to {filename}")



