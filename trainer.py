from models.rbs import RBS
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import numpy as np
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from utils.raster import rasterize_image
from utils.plots import save_inference_grid_with_text


class Trainer_Lighting(pl.LightningModule):

    def __init__(
        self,
        dataset_name = "DIV2K",
        model_name = None,
        test_and_raster_every_n_steps = 1000,
        image_dir = "",
        max_images_to_plot = 5,
        lr_size=48
    ):
        super().__init__()
        self.save_hyperparameters()

        assert model_name is not None, "You must provide a model name"

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.test_and_raster_every_n_steps = test_and_raster_every_n_steps
        self.image_dir = image_dir
        self.max_images_to_plot = max_images_to_plot
        self.last_step = -1
        self.lr_size = lr_size

        self.__init__model_version__()

    
    def __init__model_version__(self):
        
        self.model = model_versions[self.model_name]()


    def configure_optimizers(self):

        backbone_params = []
        head_params = []

        new_head_components = [
            "mlp_sigma", 
            "mlp_theta", 
            "mlp_alpha", 
            "mlp_mean", 
            "mlp_rgb_modes", 
            "mlp_rgb_base", 
            "mlp_mode_coeffs"
        ]

        for name, param in self.model.named_parameters():
            if any(head_name in name for head_name in new_head_components):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5}, 
            
            {'params': head_params, 'lr': 1e-4}
        ], weight_decay=0, eps=1e-6)


        def lr_lambda(step):
            # warm-up rapido
            if step < 2_000:
                return step / 2_000
            # decay a step fissi
            elif step < 100_000:
                return 1.0
            elif step < 200_000:
                return 0.5
            elif step < 300_000:
                return 0.25
            elif step < 400_000:
                return 0.125
            else:
                return 0.0625


        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   
            }
        }


    def configure_optimizers(self):
        if self.dataset_name == "TOY_CHECKBOARD":
            return self.configure_optimizers_checkboard()
        else:
            return self.configure_optimizers()


    #CHECKBOARD
    def configure_optimizers_checkboard(self):

        encoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "image_encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': 1e-4}, 
            {'params': other_params, 'lr': 1e-4}
        ], weight_decay=0, eps=1e-6)

        def lr_lambda(step):
            # warm-up rapido
            if step < 400:
                return 1.0
            # decay a step fissi
            elif step < 600:
                return 0.75
            elif step < 800:
                return 0.5
            elif step < 900:
                return 0.25
            elif step < 1000:
                return 0.125
            else:
                return 0.0625

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   
            }
        }


    def on_train_epoch_start(self):
        self.model.train()

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        m = self.trainer.callback_metrics

        print(
            f"[Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}] "
            f"global_step={self.global_step} | "
            f"loss={m['loss']:.6f} | "
            f"l1_loss={m['l1_loss']:.6f} | "
            f"lr={m['current_lr']:.2e} | "
            f"scale={m['choosen_scale']}",
            flush=True
        )



    def training_step(self, batch, batch_idx):

        lr_patch, hr_patch, scale = batch

        #(b, N, 18)
        brane_parameters = self.model(lr_patch, scale=scale, grt_image=hr_patch)
    
        total_render_loss = 0
        total_samples = 0

        
        num_of_batches = lr_patch.shape[0]

        for b in range(num_of_batches):
            brane_p = brane_parameters[b]
            grt_image = hr_patch[b]
            
            rasterized_image, _ = rasterize_image(brane_parameters=brane_p, 
                                                    grt=grt_image,
                                                    scale=scale,
                                                    default_step_size=self.model.default_step_size,
                                                    brane_color_type=self.model.brane_color_type,
                                                    max_deg_n=self.model.max_degree_n,
                                                    max_deg_m=self.model.max_degree_m,
                                                    dmax=self.model.dmax,
                                                    damping_list=self.model.damping_list,
                                                    predict_offset_for_each_mode=self.model.predict_offset_for_each_mode,
                                                    inference=False,
                                                    sigma_scale = 1.0 if not self.model.scale_sigma_based_on_upk else 2**self.model.upk,
                                                    relax_parameters = self.model.relax_parameters) 
                                                
            l1_pixel_wise = torch.abs(rasterized_image - grt_image)
            loss_bi = torch.mean(l1_pixel_wise)

            # 3. Rendering Loss
            total_render_loss += loss_bi

            if b == 0 and self.trainer.is_global_zero and self.dataset_name == "TOY_CHECKBOARD":
                frames_dir = os.path.join(self.image_dir, f"video_frames_{self.model_name}")
                os.makedirs(frames_dir, exist_ok=True)

                frame_path = os.path.join(frames_dir, f"frame_{self.global_step:06d}.png")
                save_image(rasterized_image, frame_path)

            total_samples += 1

        # Total Loss
        avg_render_loss = total_render_loss / total_samples
        final_loss = avg_render_loss
        

        log_dict = {}

        log_dict["loss"] = final_loss
        log_dict["choosen_scale"] = scale
        log_dict["l1_loss"] = avg_render_loss
        log_dict["current_lr"] = self.optimizers().param_groups[0]['lr']
        log_dict["global"] = self.global_step

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )




        if self.trainer.is_global_zero:
            if self.global_step % self.test_and_raster_every_n_steps == 0 and self.global_step > 0 and self.global_step!=self.last_step:
                self.last_step = self.global_step
                
                self.raster(lr_patch, hr_patch_ultra_fine=hr_patch, scale=scale)
        
        return final_loss


    @rank_zero_only
    def raster(self, lr_patch, hr_patch_ultra_fine, scale):
        self.model.eval()
        with torch.no_grad():

            max_samples_to_plot = min(self.max_images_to_plot, hr_patch_ultra_fine.shape[0])

            lr_patch = lr_patch[:max_samples_to_plot]
            hr_patch_ultra_fine = hr_patch_ultra_fine[:max_samples_to_plot]


            brane_parameters = self.model(lr_patch, scale=scale, grt_image=hr_patch_ultra_fine, inference=True)
            save_inference_grid_with_text(self.model, 
                                            lr_patch, 
                                            hr_patch_ultra_fine, 
                                            scale, 
                                            brane_parameters,
                                            self.global_step,
                                            self.image_dir,
                                            max_samples_to_plot,
                                            dmax= self.model.dmax)

                    
        self.model.train()

