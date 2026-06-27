import argparse
import cv2
import numpy as np
from os import path as osp
import os
from inference.metrics.metrics import calculate_psnr, calculate_ssim, bgr2ycbcr, scandir


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    
    for i, img_path in enumerate(img_list_gt):
        
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
     

        try:
            # Build the name based on the type of competitor (only Meta-SR give "problems" in using generic code for testing)
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
        

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # crop_border = int(args.scale)
        if args.scale <= 8:
            crop_border = int(args.scale)
        else:
            crop_border = 8
        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}   GRT: {img_path}   RESTORED: {osp.join(args.restored, basename + ext)}')
        
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(args.gt)
    print(args.restored)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, help='Path to restored images')
    parser.add_argument('--scale', type=float, default=2)
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        default=True,
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.', default=False)
    parser.add_argument('--competitor', type=str, default='', help='Name of the competitor', choices=["Meta-SR"])
    args = parser.parse_args()
    main(args)