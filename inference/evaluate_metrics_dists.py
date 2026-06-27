import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import torch
import os
from collections import Counter
import argparse
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
from DISTS_pytorch import DISTS
import argparse
from PIL import Image

def prepare_image(image, resize=False):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def cropborder(imgs, border_size = 0):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [i[:, :, border_size:-border_size, border_size:-border_size] for i in imgs]
    if len(imgs) == 0:
        return imgs[0]
    else:
        return imgs

def main(args):
    device = torch.device(f"cuda:{args.device}")

    dists_model = DISTS().to(device)

    img_list = sorted(glob.glob(osp.join(args.gt, '*')))

    dists_list = []

    if args.scale <= 8:
        crop_border = int(args.scale)
    else:
        crop_border = 8

    for i, img_path in enumerate(img_list):
        
        basename, ext = osp.splitext(osp.basename(img_path))


        try:
            if args.competitor == 'Meta-SR':
                restored_name = f"{basename}_x{float(args.scale):.1f}_SR{ext}"
            else:
                restored_name = basename + ext
                
            restored_path = osp.join(args.restored, restored_name)
            img_sr = prepare_image((Image.open(restored_path).convert("RGB")))

        except:
            print("skip")
            continue


        img_gt = prepare_image((Image.open(img_path).convert("RGB")))

        if crop_border != 0:
            img_sr, img_gt = cropborder([img_sr, img_gt], border_size=crop_border)

        img_sr = img_sr.to(device)
        img_gt = img_gt.to(device)

        dists = (dists_model(img_gt, img_sr)).item()

        print(f'{i+1:3d}: {basename:25}. \tDISTS: {dists:.6f}.    GRT: {img_path}    RESTORED: {osp.join(args.restored, basename + ext)}')

        dists_list.append(dists)

    dists_avg = sum(dists_list)/len(dists_list)
    print(f"Average DISTS is: {dists_avg:.6f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type = str,
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type = str,
                        help='Path to restored images')
    parser.add_argument('--scale', type=float, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--competitor', type=str, default='RBS', help='Name of the competitor', choices=["RBS","Meta-SR"])
    args = parser.parse_args()
    main(args)