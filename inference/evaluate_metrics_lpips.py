import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import argparse
import os
import torch 

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def main(args):
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(args.gt, '*')))
    

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    
    if args.scale <= 8:
        crop_border = int(args.scale)
    else:
        crop_border = 8
    
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        
        try:
            if args.competitor == 'Meta-SR':
                restored_name = f"{basename}_x{float(args.scale):.1f}_SR{ext}"
            else:
                restored_name = basename + ext
                
            restored_path = osp.join(args.restored, restored_name)
            img_restored = cv2.imread(restored_path, cv2.IMREAD_UNCHANGED).astype(
                np.float32) / 255.

        except:
            print("skip")
            continue

        if crop_border != 0:
            img_gt = img_gt[crop_border:-crop_border, crop_border:-crop_border, ...]
            img_restored = img_restored[crop_border:-crop_border, crop_border:-crop_border, ...]


        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.    GRT: {img_path}   RESTORED: {osp.join(args.restored, basename + ext)}')
        lpips_all.append(lpips_val.item())

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type = str,
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type = str,
                        help='Path to restored images')
    parser.add_argument('--scale', type=float, default=2)
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    parser.add_argument('--competitor', type=str, default='RBS', help='Name of the competitor', choices=["RBS", "Meta-SR"])
    args = parser.parse_args()
    main(args)