from dataset.div2k_dataset import DIV2KDataset, ASR_Collate
from dataset.checkboard_dataset import CheckerboardDataset, checkerboard_collate
import argparse
from models.model_versions import model_versions
import torch
import os
from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from trainer import Trainer_Lighting

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    pl.seed_everything(args.global_seed, workers=True)

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    image_dir = f"{experiment_dir}/images"  # Stores images during training

    #model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir,"checkpoints/"),                 # folder where to save
        filename="model_step={step}",           # name with the current step
        every_n_epochs=args.every_n_epochs, 
        save_top_k=1,                          # save the best
        monitor="loss",
        save_last= True,                        #save also the last model
        save_weights_only=False,                # save everithing ? (optimizer, ecc.)
    )




    #Create trainer
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=-1, 
                         min_epochs=1, 
                         max_epochs=args.epochs, 
                         precision=args.precision, 
                         strategy="ddp_find_unused_parameters_true",
                         callbacks=[checkpoint_callback],
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         gradient_clip_val=1.0, 
                         gradient_clip_algorithm="norm",
                         enable_progress_bar=False)



    if args.dataset_name == "TOY_CHECKBOARD":
        dataset = CheckerboardDataset(
                    lr_size=12, 
                    scale=1.0,       
                    iterations=100 
                )
        asr_collate = checkerboard_collate
    
    else:
        if args.dataset_name == "DIV2K":
            dataset = DIV2KDataset(path_to_image_dataset=args.path_to_image_dataset)

        #useful to decide the same random scale for the batch
        asr_collate = ASR_Collate(lr_size=args.start_resolution,
                                    min_scale=1.0,
                                    max_scale=args.max_up_scales)
    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=asr_collate, drop_last=True)


    trainer_light = Trainer_Lighting(
                            dataset_name = args.dataset_name,
                            lr_size=args.start_resolution,
                            model_name=args.model, 
                            test_and_raster_every_n_steps=args.plot_images_every, 
                            image_dir=image_dir,
                            max_images_to_plot = args.max_images_to_plot)




    if trainer.is_global_zero:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(image_dir, exist_ok=True)

    #start training
    trainer.fit(model=trainer_light, train_dataloaders=dataloader, ckpt_path=args.load_checkpoint if args.load_checkpoint!="" else None)









if __name__ == "__main__":
    # Default args here will train DiGT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image_dataset", type=str)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--model", type=str, choices=list(model_versions), default="RBS_v1")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--plot_images-every", type=int, default=5000)
    parser.add_argument("--max_images_to_plot", type=int, default=16)
    parser.add_argument("--use_perceptual", action="store_true")
    parser.add_argument("--start_resolution", type=int, default=48) 
    parser.add_argument("--max_up_scales",     type=int, default=4)

    #type of dataset 
    parser.add_argument("--dataset_name", type=str, choices=["DIV2K", "TOY_CHECKBOARD"], default="DIV2K")

    #light
    parser.add_argument("--precision", type=int, default=16, choices=[16,32,64])
    parser.add_argument("--batch-size", type=int, default=256) 
    parser.add_argument("--every_n_epochs", type=int, default=1000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--global_seed", type=int, default=42)  

    
    args = parser.parse_args()
    main(args)