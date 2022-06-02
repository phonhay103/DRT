import numpy as np
import torch
import os
import cv2

###########################  PSNR  ###################################
def PSNR(img1, img2, max_pixel=255.0):
    """Calculate PSNR
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'
        img2 (ndarray): Images with range [0, 255] with order 'HWC'
    Returns:
        float: psnr result
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
        
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(max_pixel / np.sqrt(mse))
#######################################################################


def mySSIM(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(mySSIM(img1[..., i], img2[..., i]))

    return np.array(ssims).mean()


def batch_SSIM(img1, img2):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        # preprocessing: scale to 255
        im1 = (im1 * 255.0).round().astype(np.uint8)
        im2 = (im2 * 255.0).round().astype(np.uint8)

        ssim = calculate_ssim(im1, im2)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM)


### save model during training
def save_ckp(state, save_dir):
    f_path = save_dir + '/model.pt'
    torch.save(state, f_path)

### load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if('best_error' in checkpoint):
        best_error = checkpoint['best_error']
    else:
        best_error = None
    return model, optimizer, checkpoint['epoch'], checkpoint['error_list'], best_error

### return the base directory
def base_path():
    return os.path.dirname(os.path.abspath(__file__))