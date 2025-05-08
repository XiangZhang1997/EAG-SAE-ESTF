import torch
import os
import numpy as np
import torch.nn.functional as F

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        if key == 'ori_query_img' or key == 'ori_support_imgs':
            continue
        else:
            batch_dict[key] = val.cuda()
    return batch_dict

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    if curr_iter % 50 == 0:
        print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr

def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def double_threshold_iteration(index=1,img=None, h_thresh=0.5, l_thresh=0.3, save=False):
    h, w = img.shape
    img = np.array(img.cpu().detach()*255, dtype=np.uint8) 
    bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8) 
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                    if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                        gbin[i][j] = 255
    if save:
        cv2.imwrite(f"save_picture/bin{index}.png", bin) 
        cv2.imwrite(f"save_picture/gbin{index}.png", gbin) 
    return gbin/255 # (0 or 1)

def process_image_in_patches_overleap(model, batch, patch_size=256,overlap=32):


    # merged_output = process_image_in_patches_overleap_(model, batch, patch_size=160,overlap=20, s="output") 
    merged_output = process_image_in_patches_overleap_(model, batch, patch_size=256,overlap=32, s="output") 
    return merged_output

# def process_image_in_patches_overleap_(model, batch, patch_size=160,overlap=20, s="output"):
def process_image_in_patches_overleap_(model, batch, patch_size=256, overlap=32, s="output"):
    B, C, H, W = batch.shape
    stride = patch_size - overlap
    num_patches_x = (H - overlap) // stride
    num_patches_y = (W - overlap) // stride

    merged_output = torch.zeros((B, 1, H, W), device=batch.device)
    count_map = torch.zeros((B, 1, H, W), device=batch.device) 

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_start = i * stride
            y_start = j * stride
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            x_end = min(x_end, H)
            y_end = min(y_end, W)

            patch = batch[:, :, x_start:x_end, y_start:y_end]
            with torch.no_grad():
                output_dict= model(patch)  
                output_patch = output_dict[s]

            merged_output[:, :, x_start:x_end, y_start:y_end] += output_patch
            count_map[:, :, x_start:x_end, y_start:y_end] += 1

    count_map[count_map == 0] = 1e-8  
    merged_output /= count_map

    return merged_output

def process_image_in_patches(model, batch, patch_size=256, B=1, C=1, H=512, W=512):

    B, C, H, W = batch.shape
    num_patches_x = H // patch_size
    num_patches_y = W // patch_size

    output_patches = []

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_start = i * patch_size
            y_start = j * patch_size
            patch = batch[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]

            with torch.no_grad():
                output_dict = model(patch)
                output_patch = output_dict['output'] 
            output_patches.append(output_patch)

    merged_output = torch.zeros((B, C, H, W), device=batch.device)
    idx = 0
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_start = i * patch_size
            y_start = j * patch_size
            merged_output[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size] = output_patches[idx]
            idx += 1
    return merged_output
