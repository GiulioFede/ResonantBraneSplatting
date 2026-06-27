import torch
from torch.utils.data import Dataset

class CheckerboardDataset(Dataset):
    def __init__(self, lr_size=12, scale=4, iterations=1000):
        self.lr_size = lr_size
        self.scale = scale
        self.hr_size = lr_size * int(scale)
        self.iterations = iterations

        self.hr_img = torch.zeros((3, self.hr_size, self.hr_size))
        square_h = self.hr_size // 4
        square_w = self.hr_size // 4

        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    self.hr_img[0, i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w] = 1.0 
                else:
                    self.hr_img[1, i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w] = 1.0 

        self.lr_img = torch.nn.functional.interpolate(
            self.hr_img.unsqueeze(0), 
            size=(self.lr_size, self.lr_size), 
            mode='area'
        ).squeeze(0)

    def __len__(self):
        return self.iterations 

    def __getitem__(self, idx):

        return self.lr_img, self.hr_img



def checkerboard_collate(batch):
    # batch is a list of tuples[(lr_1, hr_1), (lr_2, hr_2), ...]
    lr_images = [item[0] for item in batch]
    hr_images = [item[1] for item in batch]
    
    # (b, c, h, w)
    lr_tensor = torch.stack(lr_images, dim=0)
    hr_tensor = torch.stack(hr_images, dim=0)
    
    scale = 1.0 
    
    return lr_tensor, hr_tensor, scale