import os
import logging
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from core.utils.load_img import Load_from_Folder, Load_Name_from_Folder, Load_Images
from core.utils import Shrink


# load images and labels
# def load_data(args):
#     # images folder
#     images, name_list = Load_from_Folder(args.data_dir, color="YUV", ct=-1, yuv=args.yuv, size=(args.height, args.width))

#     # load labels if available
#     mos_file = os.path.join(args.data_dir, "mos.csv")
#     if os.path.exists(mos_file):
#         mos_table = pd.read_csv(mos_file)
#         mos_dict = dict()
#         for index, row in mos_table.iterrows():
#             mos_dict[row['image_name']] = row['MOS']
        
#         mos = []
#         new_index = []
#         for i in range(len(name_list)):
#             if name_list[i] in mos_dict:
#                 mos.append(mos_dict[name_list[i]])
#                 new_index.append(i)
        
#         # name_list = [name_list[i] for i in new_index]
#         images = [images[i] for i in new_index]
#         mos = np.array(mos)
#     else:
#         mos = None

#     return images, mos

def load_data(args, batch_size=32):
    # images folder
    name_list = Load_Name_from_Folder(args.data_dir)
    
    # load labels if available
    mos_file = os.path.join(args.data_dir, "mos.csv")
    if os.path.exists(mos_file):
        mos_table = pd.read_csv(mos_file)
        mos_dict = dict()
        for index, row in mos_table.iterrows():
            mos_dict[row['image_name']] = row['MOS']
    else:
        mos_dict = None

    num_batches = (len(name_list) + batch_size - 1) // batch_size
    for i in range(num_batches):
        logging.info(f"Loading batch {i+1}/{num_batches}...")
        batch_name_list = name_list[i*batch_size : (i+1)*batch_size]
        batch_name_list = [os.path.join(args.data_dir, name) for name in batch_name_list]
        
        batch_images = Load_Images(batch_name_list, color='YUV')
        
        if mos_dict is not None:
            batch_mos = []
            for name in batch_name_list:
                name = os.path.basename(name)
                if name in mos_dict:
                    batch_mos.append(mos_dict[name])
            batch_mos = np.array(batch_mos) if len(batch_mos) > 0 else None
        else:
            batch_mos = None
        
        yield batch_images, batch_mos
        

# data augmentation
def augment(images, mos, num_aug):
    aug_images = []
    for i in range(len(images)):
        aug_images.append(crop(images[i], num_aug))

    aug_images = np.concatenate(aug_images, axis=0)

    n = aug_images.shape[0]
    flipped_idx = np.arange(n)
    np.random.shuffle(flipped_idx)
    flipped_idx = flipped_idx[:(n // 2)]
    aug_images[flipped_idx] = flip(aug_images[flipped_idx])

    return aug_images, np.repeat(mos, num_aug)


def crop(img, num_aug, r=384):
    h, w, c = img.shape

    aug = np.zeros((num_aug, r, r, c))
    try:
        h_location = np.random.randint(h - r + 1, size=num_aug)
        w_location = np.random.randint(w - r + 1, size=num_aug)
        for i in range(num_aug):
            aug[i] = img[h_location[i]:(h_location[i] + r), w_location[i]:(w_location[i] + r), :]
    except:
        for i in range(num_aug):
            win = min(h, w)
            aug[i, :win, :win, :] = img[:win, :win, :]

    return aug


def flip(images):
    return np.flip(images, axis=2)


# DCT
def DCT_blocks(X):
    n, h, w = X.shape
    DCT_coeff = dct(dct(X, axis=1, norm="ortho"), axis=2, norm="ortho")
    return DCT_coeff.reshape(-1, h, w)


def zigzag(X):
    output = np.zeros((X.shape[0], 64))
    i_idx = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0,
             1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7]
    j_idx = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7,
             7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7]
    for i in range(64):
        output[:, i] = X[:, i_idx[i], j_idx[i]]
    return output


def jpeg_dct(X, channel='Y'):
    n, h, w, c = X.shape
    new_h, new_w = int(h // 8), int(w // 8)

    X = np.array(X, dtype='float32')

    if channel == 'Y':
        X_block = Shrink(X[:, :, :, 0:1], 8).reshape(-1, 8, 8)
    elif channel == 'U':
        X_block = Shrink(X[:, :, :, 1:2], 8).reshape(-1, 8, 8)
    elif channel == 'V':
        X_block = Shrink(X[:, :, :, 2:3], 8).reshape(-1, 8, 8)
    else:
        X_block = Shrink(X[:, :, :, 0:1], 8).reshape(-1, 8, 8)

    X_block_dct = DCT_blocks(X_block)
    X_block_dct = zigzag(X_block_dct)
    X_block_dct = X_block_dct.reshape(n, new_h, new_w, 64)

    return X_block_dct
