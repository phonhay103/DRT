import torch
import torch.cuda as cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model import DeepRecursiveTransformer

class RainData(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.img_transforms = self.build_transform()
        self.datasets = []
        p = Path(dataset_dir)
        for ext in ['png', 'jpg']:
            self.datasets.extend(p.glob(f'*.{ext}'))

    def build_transform(self):
        t = []
        t.append(transforms.ToTensor()) #convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
        return transforms.Compose(t)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        path = self.datasets[idx]
        img = Image.open(path)
        img = self.img_transforms(img)
        return img, path.name

def main(args):
    local_window_dim = args.patch_size * 7
    Path(args.out_path).mkdir(exist_ok=True)
    test_data_loader = DataLoader(RainData(args.in_path), batch_size=1, shuffle=False)
    
    for net_input, img_name in tqdm(test_data_loader):
        ### pad the image to make sure H == W fits the network requirements
        _, _, h_old, w_old = net_input.size()
        h_original = h_old
        w_original = w_old
        multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
        h_pad = (multiplier) * local_window_dim - h_old
        w_pad = (multiplier) * local_window_dim - w_old
        net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
        net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

        ## pad again if h/w or w/h ratio is bigger than 2
        if h_pad > h_old or w_pad > w_old:
            _, _, h_old, w_old = net_input.size()
            multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
            h_pad = (multiplier) * local_window_dim - h_old
            w_pad = (multiplier) * local_window_dim - w_old
            net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
            net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

        ### evaluate | load model with each image
        _, _, new_h, new_w = net_input.size()
        assert new_h == new_w, "Input image should have square dimension"
        eval_net = DeepRecursiveTransformer(args.dim, (new_h, new_w), args.patch_size, args.residual_depth, args.recursive_depth)
        eval_net.load_state_dict(torch.load(args.ckp_path)['state_dict'])
        eval_net.to(args.device)
        eval_net.eval()

        net_input = net_input.cuda()
        with torch.no_grad():
            net_output = eval_net(net_input)

        ### crop the output
        net_output = net_output[:, :, :h_original, :w_original]
        output_data = net_output.cpu().detach().numpy() # B C H W
        output_data = np.transpose(output_data, (0, 2, 3, 1)) # B H W C
        output_data = np.clip(output_data * 255, 0, 255).astype(np.uint8)

        # Save
        img = Image.fromarray(output_data[0])
        img.save(Path(args.out_path, img_name[0]))

        cuda.empty_cache()
        
parser = argparse.ArgumentParser(description='Image Deraining using DRT')
parser.add_argument('--dim', type=int, default=96)
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--residual_depth', type=int, default=3)
parser.add_argument('--recursive_depth', type=int, default=6)
parser.add_argument('--ckp_path', type=str, default='./pretrained/best_model.pt')
parser.add_argument('--device', type=str, default='cuda') # 'cpu'
parser.add_argument('--in_path', type=str, default='datasets/Test-HiNet/Rain100H/input')
parser.add_argument('--out_path', type=str, default='datasets/Test-HiNet/Rain100H/output')

args = parser.parse_args()
main(args)