from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random





import cv2
class DIV2KDataset(Dataset):
    def __init__(self, 
                 path_to_image_dataset=None):
        
        self.path_to_image_dataset = path_to_image_dataset

        self.img_paths = sorted(glob.glob(os.path.join(path_to_image_dataset, "*.png")))




    def __len__(self):
        return len(self.img_paths) 

    def imfromfile(self, path, flag='color', float32=False):
        imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
        img = cv2.imread(path, imread_flags[flag])
        if float32:
            img = img.astype(np.float32) / 255.
        return img


    def __getitem__(self, idx):

        image_path = self.img_paths[idx]

        img_gt = self.imfromfile(path=image_path, float32=True)
    
        return img_gt


import torch
import torch.nn.functional as F
import random
import math

class ASR_Collate:
    def __init__(self, lr_size=48, min_scale=1.1, max_scale=4.0):
        """
        Args:
            window_size: dimension of the window (12).
            min_mult: minimum multiplier (ex. 4 -> 48px).
            max_mult: maximum multiplier (ex. 6 -> 72px).
        """
        self.lr_size = lr_size
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.use_hflip = True
        self.use_rot = True
        self.gt_size_max = math.ceil(self.max_scale * self.lr_size)

    def cubic(self, x):
        """cubic function used for calculate_weights_indices."""
        absx = torch.abs(x)
        absx2 = absx**2
        absx3 = absx**3
        return (1.5 * absx3 - 2.5 * absx2 + 1) * (
            (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) *
                                                                                        (absx <= 2)).type_as(absx))

    def calculate_weights_indices(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        """Calculate weights and indices, used for imresize function.

        Args:
            in_length (int): Input length.
            out_length (int): Output length.
            scale (float): Scale factor.
            kernel_width (int): Kernel width.
            antialisaing (bool): Whether to apply anti-aliasing when downsampling.
        """

        if (scale < 1) and antialiasing:
            kernel_width = kernel_width / scale

        # Output-space coordinates
        x = torch.linspace(1, out_length, out_length)

        # Input-space coordinates. Calculate the inverse mapping such that 0.5
        # in output space maps to 0.5 in input space, and 0.5 + scale in output
        # space maps to 1.5 in input space.
        u = x / scale + 0.5 * (1 - 1 / scale)

        # What is the left-most pixel that can be involved in the computation?
        left = torch.floor(u - kernel_width / 2)

        # What is the maximum number of pixels that can be involved in the
        # computation? 
        # corresponding weights are all zero, it will be eliminated at the end
        # of this function.
        p = math.ceil(kernel_width) + 2

        # The indices of the input pixels involved in computing the k-th output
        # pixel are in row k of the indices matrix.
        indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
            out_length, p)

        # The weights used to compute the k-th output pixel are in row k of the
        # weights matrix.
        distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

        # apply cubic kernel
        if (scale < 1) and antialiasing:
            weights = scale * self.cubic(distance_to_center * scale)
        else:
            weights = self.cubic(distance_to_center)

        # Normalize the weights matrix so that each row sums to 1.
        weights_sum = torch.sum(weights, 1).view(out_length, 1)
        weights = weights / weights_sum.expand(out_length, p)

        # If a column in weights is all zero, get rid of it. only consider the
        # first and last column.
        weights_zero_tmp = torch.sum((weights == 0), 0)
        if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
            indices = indices.narrow(1, 1, p - 2)
            weights = weights.narrow(1, 1, p - 2)
        if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
            indices = indices.narrow(1, 0, p - 2)
            weights = weights.narrow(1, 0, p - 2)
        weights = weights.contiguous()
        indices = indices.contiguous()
        sym_len_s = -indices.min() + 1
        sym_len_e = indices.max() - in_length
        indices = indices + sym_len_s - 1
        return weights, indices, int(sym_len_s), int(sym_len_e)


    @torch.no_grad()
    def imresize_new(self, img, scale_h, scale_w, antialiasing=True):
        ###Note that, now the code only support scale_h == scale_w, if the two params are not the same, maybe some error will occur.
        squeeze_flag = False
        if type(img).__module__ == np.__name__:  # numpy type
            numpy_type = True
            if img.ndim == 2:
                img = img[:, :, None]
                squeeze_flag = True
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        else:
            numpy_type = False
            if img.ndim == 2:
                img = img.unsqueeze(0)
                squeeze_flag = True

        in_c, in_h, in_w = img.size()
        out_h, out_w = round(in_h * scale_h), round(in_w * scale_w)
        kernel_width = 4
        kernel = 'cubic'

        # get weights and indices
        weights_h, indices_h, sym_len_hs, sym_len_he = self.calculate_weights_indices(in_h, out_h, scale_h, kernel, kernel_width,
                                                                                antialiasing)
        weights_w, indices_w, sym_len_ws, sym_len_we = self.calculate_weights_indices(in_w, out_w, scale_w, kernel, kernel_width,
                                                                                antialiasing)
        # process H dimension
        # symmetric copying
        img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
        img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

        sym_patch = img[:, :sym_len_hs, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

        sym_patch = img[:, -sym_len_he:, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

        out_1 = torch.FloatTensor(in_c, out_h, in_w)
        kernel_width = weights_h.size(1)
        for i in range(out_h):
            idx = int(indices_h[i][0])
            for j in range(in_c):
                out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

        # process W dimension
        # symmetric copying
        out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
        out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

        sym_patch = out_1[:, :, :sym_len_ws]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

        sym_patch = out_1[:, :, -sym_len_we:]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

        out_2 = torch.FloatTensor(in_c, out_h, out_w)
        kernel_width = weights_w.size(1)
        for i in range(out_w):
            idx = int(indices_w[i][0])
            for j in range(in_c):
                out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

        if squeeze_flag:
            out_2 = out_2.squeeze(0)
        if numpy_type:
            out_2 = out_2.numpy()
            if not squeeze_flag:
                out_2 = out_2.transpose(1, 2, 0)

        return out_2


    def augment(self, imgs, hflip=True, rotation=True, flows=None, return_status=False):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

        We use vertical flip and transpose for rotation implementation.
        All the images in the list use the same augmentation.

        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Ratotation. Default: True.
            flows (list[ndarray]: Flows to be augmented. If the input is an
                ndarray, it will be transformed to a list.
                Dimension is (h, w, 2). Default: None.
            return_status (bool): Return the status of flip and rotation.
                Default: False.

        Returns:
            list[ndarray] | ndarray: Augmented images and flows. If returned
                results only have one element, just return ndarray.

        """
        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(src=img, flipCode=1, dst = img)
                # print(f"hflip shape is {img.shape}")
            if vflip:  # vertical
                cv2.flip(src=img, flipCode=0, dst = img)
                # print(f"vflip shape is {img.shape}")
            if rot90:
                # print(img.shape)
                img = img.transpose(1, 0, 2)
            return img

        def _augment_flow(flow):
            if hflip:  # horizontal
                cv2.flip(flow, 1, flow)
                flow[:, :, 0] *= -1
            if vflip:  # vertical
                cv2.flip(flow, 0, flow)
                flow[:, :, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0, 2)
                flow = flow[:, :, [1, 0]]
            return flow

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        if flows is not None:
            if not isinstance(flows, list):
                flows = [flows]
            flows = [_augment_flow(flow) for flow in flows]
            if len(flows) == 1:
                flows = flows[0]
            return imgs, flows
        else:
            if return_status:
                return imgs, (hflip, vflip, rot90)
            else:
                return imgs

    def img2tensor(self, imgs, bgr2rgb=True, float32=True):
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

    def __call__(self, batch):

        scale = float(random.uniform(self.min_scale, self.max_scale))

        lr_images = []
        gt_images = []

        for b in range(len(batch)):

            img_gt = batch[b]

            h_img_gt, w_img_gt, _ = img_gt.shape

            lr_size = torch.tensor([self.lr_size, self.lr_size])

            gt_size = torch.tensor([round(scale * lr_size[0].item()), round(scale * lr_size[1].item())])

            # --- PADDING ---
            pad_h = max(0, gt_size[0].item() - h_img_gt)
            pad_w = max(0, gt_size[1].item() - w_img_gt)

            if pad_h > 0 or pad_w > 0:
                img_gt = np.pad(img_gt, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                h_img_gt, w_img_gt, _ = img_gt.shape
            # ------

            start_h_crop_gt = random.randint(0, h_img_gt - gt_size[0])
            start_w_crop_gt = random.randint(0, w_img_gt - gt_size[1])


            crop_gt = img_gt[start_h_crop_gt:start_h_crop_gt + gt_size[0], start_w_crop_gt:start_w_crop_gt+gt_size[1], :]

            scale_modify_h = float(crop_gt.shape[0] / self.lr_size)
            scale_modify_w = float(crop_gt.shape[1] / self.lr_size)
            crop_lr = np.ascontiguousarray(self.imresize_new(img = crop_gt, scale_h = 1 / scale_modify_h, scale_w = 1 / scale_modify_w, antialiasing=True))

            scale_modify = torch.tensor([scale_modify_h, scale_modify_w])

            img_gt, img_lq = self.augment([crop_gt, crop_lr], self.use_hflip, self.use_rot)

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = self.img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

            lr_images.append(img_lq)
            gt_images.append(img_gt)
        
        return stack_tensors(lr_images), stack_tensors(gt_images), scale


def stack_tensors(tensor_list):
    return torch.stack(tensor_list, dim=0)
