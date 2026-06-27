
import argparse
import torch
import os
import glob
import cv2
import numpy as np
import torch.nn.functional as F
from utils.raster import rasterize_image
import time
import torch
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
from models.model_versions import model_versions



def preprocess(x, denominator):
    # pad input image to be a multiple of denominator
    _,c,h,w = x.shape
    if h % denominator > 0:
        pad_h = denominator - h % denominator
    else:
        pad_h = 0
    if w % denominator > 0:
        pad_w = denominator - w % denominator
    else:
        pad_w = 0
    x_new = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    return x_new

def postprocess(x, gt_size_h, gt_size_w):
    x_new = x[:, :, :gt_size_h, :gt_size_w]
    return x_new


class Trainer_Lighting(pl.LightningModule):

    def __init__(
        self,
        model_name = None,
    ):
        super().__init__()

        self.model_name = model_name

        self.__init__model_version__()

    
    def __init__model_version__(self):
        
        self.model = model_versions[self.model_name]()



def main(args):

    assert args.path_to_lr_image is not None or args.path_to_image_dataset is not None, "You must provide the path to LR image or to a benchmark folder"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.ckpt_path is not None:
            if args.model_name is None:
                raise ValueError("You must specify also --model_name if you use --ckpt_path.")
            
            ckpt_path = args.ckpt_path
            arch_name = args.model_name
            print(f"Loading custom model '{arch_name}' from: {ckpt_path}")
            
    else:
        print("You must provide a checkpoint path to the model.")

    model = Trainer_Lighting.load_from_checkpoint(
                                strict=False,
                                checkpoint_path=ckpt_path,  
                                model_name=arch_name,       
                                map_location=device)
    
    model = model.model
    model.to(device) 
    model.eval()

    type_of_inference = args.type_of_inference
    dmax = torch.tensor(args.dmax, device=device, dtype=torch.float32)

    if args.path_to_image_dataset:
        # LOAD all LR image path
        lr_image_paths = sorted(glob.glob(os.path.join(args.path_to_image_dataset, "x"+str(args.scale_to_evaluate), "LR", "*.png")))
        dataset_name = args.path_to_image_dataset.split("/")[-1]
    else:
        lr_image_paths = [args.path_to_lr_image]
        dataset_name = args.path_to_lr_image.split("/")[-1].split(".")[-2]
    
    result_dir = os.path.join(args.results_dir, dataset_name, "x"+str(args.scale_to_evaluate))
    os.makedirs(result_dir, exist_ok=True)

    time_cost_list = []
    used_memory_list = []

    print(f"Start inference on {dataset_name} with scale {args.scale_to_evaluate}")

    img_i = 0
    for image_path in lr_image_paths:
        print(image_path)
        image_name = os.path.basename(image_path)
        img_i+=1

        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img_cv[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        gt_size = [math.floor(args.scale_to_evaluate * img.shape[2]), math.floor(args.scale_to_evaluate * img.shape[3])]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()            
            torch.cuda.reset_peak_memory_stats() 
            torch.cuda.synchronize()

        start = time.time()

        with torch.no_grad():
            if type_of_inference == "tiling":
                raise ValueError("Tiling process not yet implemented. Please use parallel instead.")
            else:
                lq_pad = preprocess(img, 12)
                gt_size_pad = torch.tensor([math.floor(args.scale_to_evaluate* lq_pad.shape[2]), math.floor(args.scale_to_evaluate * lq_pad.shape[3])])
                gt_size_pad = gt_size_pad.unsqueeze(0)

                brane_parameters = model(lq_pad, scale=args.scale_to_evaluate, inference=True)[0]
                
                sr_results,_ = rasterize_image(brane_parameters,
                            gt_size_pad[0],
                            args.scale_to_evaluate,
                            model.default_step_size,
                            brane_color_type=model.brane_color_type,
                            max_deg_n=model.max_degree_n,
                            max_deg_m=model.max_degree_m,
                            dmax=dmax,
                            damping_list=model.damping_list,
                            predict_offset_for_each_mode=model.predict_offset_for_each_mode,
                            sigma_scale = 1.0 if not model.scale_sigma_based_on_upk else 2**model.upk,
                            #sigma_scale = 1.0 if model.upk is None else 2**model.upk,
                            use_rmax=True)
                
                sr_results = sr_results.unsqueeze(0)

        torch.cuda.synchronize()
        end = time.time()
        time_cost = end - start
        

        if torch.cuda.is_available():
            used_memory = torch.cuda.max_memory_allocated()
            print(f"{img_i} Image: {image_name} | Time: {time_cost*1000:.4f} ms | GPU Peak: {used_memory /1024 /1024:.2f} MB")
        else:
            used_memory = 0

        time_cost_list.append(time_cost)
        used_memory_list.append(used_memory) 

        output = postprocess(sr_results, gt_size[0], gt_size[1])

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        cv2.imwrite(os.path.join(result_dir, image_name), output)

        del sr_results
        del img
    
    # Final statistics
    if len(time_cost_list) > 2:
        # Time: -> ms
        # Memory: MB
        times_ms = np.array(time_cost_list) * 1000
        mems_mb = np.array(used_memory_list) / (1024 * 1024)

        # --- Time (Skip first two) ---
        valid_times = times_ms[2:] 
        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times)
        
        # --- Memory (Skip first two) ---
        valid_mems = mems_mb[2:]
        avg_mem = np.mean(valid_mems)
        std_mem = np.std(valid_mems)
        min_mem = np.min(valid_mems)
        max_mem = np.max(valid_mems)

        print("\n" + "="*40)
        print(f"FINAL STATISTICS (Skipped first 2 samples)")
        print("="*40)
        
        # Format: Mean ± Std
        print(f"TIME   : {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"MEMORY : {avg_mem:.2f} ± {std_mem:.2f} MB")
        
        # Range: Min and Max
        print(f"MEM Range: [{min_mem:.2f} - {max_mem:.2f}] MB")
        print("="*40)
    else:
        print("Not enough samples to calculate statistics (need > 2).")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image_dataset", type=str) 
    parser.add_argument("--path_to_lr_image", type=str, 
                        help="Use this argument only if you want to do inference on a single image (don't use --path_to_image_dataset))") 
    
    parser.add_argument("--scale_to_evaluate", type=int, choices=[2,3,4,6,8,12,16,24,30], default=6) 
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--type_of_inference", type=str, choices=["parallel"], default="parallel") #TODO "tiling"
    parser.add_argument('--dmax', type = float, default = 0.1,
    help="Determines the rasterization speed for each level. Lower values mean higher speed. "
    )

    
    # parser.add_argument(
    #     "--pretrained_model", 
    #     type=str, 
    #     choices=["RDN_best", "RDN_classic", "EDSR_best"], 
    #     default=None, 
    #     help="Select the pre-trained weights to use. They will be downloaded automatically if missing."
    # )


    #------------------------------
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="./checkpoints/EDSR_rbs_v8.ckpt", 
        help="Path to a specific local checkpoint file. If provided, overrides --pretrained_model."
    )

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="RBS_v8", 
        help="Name of the model architecture (e.g., RBS_v1). Required if --ckpt_path is provided."
    )

    args = parser.parse_args()
    main(args)