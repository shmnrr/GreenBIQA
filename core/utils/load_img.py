# @yifan
# 2021.01.12

import numpy as np
import cv2
import os
import logging
from skimage.measure import block_reduce

from core.utils.color_space import BGR2RGB, BGR2YUV


# def Load_from_Folder(folder, color='RGB', ct=1, yuv=False, size=None, train=False):
#     name = os.listdir(folder)
#     name.sort()
#     img = []
#     Y, U, V = [], [], []
#     name_list = []
#     h, w = size
    
#     # Calculate the split index based on train flag
#     logging.info(f"Loading {'train' if train else 'test'} data...")
#     split_index = int(len(name) * 0.9) - 1
#     selected_names = name[:split_index] if train else name[split_index:]
#     logging.info(f"Split index: {split_index}")
    
#     for n in selected_names:
#         if yuv:
#             if n.split('.')[-1] != 'yuv':
#                 continue
#             f = open(os.path.join(folder, n), "rb")
#             data_Y = f.read(h * w)
#             data_U = f.read((h//2) * (w//2))
#             data_V = f.read((h//2) * (w//2))
#             data_Y = [int(x) for x in data_Y]
#             data_U = [int(x) for x in data_U]
#             data_V = [int(x) for x in data_V]
#             X = np.zeros((h, w, 3))
#             X[:, :, 0] = np.array(data_Y).reshape(h, w).astype('float32')
#             shrunk_u = np.array(data_U).reshape(h//2, w//2).astype('float32')
#             shrunk_v = np.array(data_V).reshape(h//2, w//2).astype('float32')
#             X[:, :, 1] = cv2.resize(shrunk_u, (w, h), interpolation=cv2.INTER_LINEAR) - 127
#             X[:, :, 2] = cv2.resize(shrunk_v, (w, h), interpolation=cv2.INTER_LINEAR) - 127
#             img.append(X)
#             name_list.append(n)
#             continue
        
#         X = cv2.imread(folder+'/'+n)
#         if X is None:
#             continue
#         name_list.append(n)
#         if color == 'BGR':
#             img.append(X)
#         elif color == 'RGB':
#             img.append(BGR2RGB(X))
#         elif color == 'YUV444' or color == 'YUV':
#             img.append(BGR2YUV(X))
#         elif color == 'YUV420':
#             X = BGR2YUV(X)
#             Y.append(X[:, :, 0])
#             U.append(block_reduce(X[:, :, 1], (2, 2), np.mean))
#             V.append(block_reduce(X[:, :, 2], (2, 2), np.mean))
#         else:
#             logging.info('No such color type!, Color must be BGR, RGB, YUV, YUV444, or YUV420!')
#             break
        
#         ct -= 1
#         if ct == 0:
#             break
        
#     if color == 'BGR' or color == 'RGB' or color == 'YUV444' or color == 'YUV':
#         return img, name_list
#     elif color == 'YUV420':
#         return Y, U, V, name_list


def Load_from_Folder(folder, color='RGB', ct=1, yuv=False, size=None, train=False, batch_size=None):
    name = os.listdir(folder)
    name.sort()
    img = []
    Y, U, V = [], [], []
    name_list = []
    h, w = size
    
    # Calculate the split index based on train flag
    logging.info(f"Loading {'train' if train else 'test'} data...")
    split_index = int(len(name) * 0.9) - 1
    selected_names = name[:split_index] if train else name[split_index:]
    logging.info(f"Split index: {split_index}")
    
    batch_count = 0
    for n in selected_names:
        if yuv:
            if n.split('.')[-1] != 'yuv':
                continue
            f = open(os.path.join(folder, n), "rb")
            data_Y = f.read(h * w)
            data_U = f.read((h//2) * (w//2))
            data_V = f.read((h//2) * (w//2))
            data_Y = [int(x) for x in data_Y]
            data_U = [int(x) for x in data_U]
            data_V = [int(x) for x in data_V]
            X = np.zeros((h, w, 3))
            X[:, :, 0] = np.array(data_Y).reshape(h, w).astype('float32')
            shrunk_u = np.array(data_U).reshape(h//2, w//2).astype('float32')
            shrunk_v = np.array(data_V).reshape(h//2, w//2).astype('float32')
            X[:, :, 1] = cv2.resize(shrunk_u, (w, h), interpolation=cv2.INTER_LINEAR) - 127
            X[:, :, 2] = cv2.resize(shrunk_v, (w, h), interpolation=cv2.INTER_LINEAR) - 127
            img.append(X)
            name_list.append(n)
            continue
        
        X = cv2.imread(folder+'/'+n)
        if X is None:
            continue
        name_list.append(n)
        if color == 'BGR':
            img.append(X)
        elif color == 'RGB':
            img.append(BGR2RGB(X))
        elif color == 'YUV444' or color == 'YUV':
            img.append(BGR2YUV(X))
        elif color == 'YUV420':
            X = BGR2YUV(X)
            Y.append(X[:, :, 0])
            U.append(block_reduce(X[:, :, 1], (2, 2), np.mean))
            V.append(block_reduce(X[:, :, 2], (2, 2), np.mean))
        else:
            logging.info('No such color type!, Color must be BGR, RGB, YUV, YUV444, or YUV420!')
            break
        
        ct -= 1
        if ct == 0:
            break
        
        if batch_size is not None and len(img) % batch_size == 0:
            batch_count += 1
            logging.info(f"Batch {batch_count} with {batch_size} images")
            if color == 'BGR' or color == 'RGB' or color == 'YUV444' or color == 'YUV':
                yield img, name_list
            elif color == 'YUV420':
                yield Y, U, V, name_list
        
    batch_count += 1
    logging.info(f"Final batch {batch_count} with {len(img) % batch_size} images")
    if color == 'BGR' or color == 'RGB' or color == 'YUV444' or color == 'YUV':
        yield img, name_list
    elif color == 'YUV420':
        yield Y, U, V, name_list


def Load_Name_from_Folder(folder):
    name = os.listdir(folder)
    name.sort()
    return name


def Load_Images(name_list, color='RGB'):
    img = []
    Y, U, V = [], [], []
    for n in name_list:
        X = cv2.imread(n)
        if X is None:
            continue
        if color == 'BGR':
            img.append(X)
        elif color == 'RGB':
            img.append(BGR2RGB(X))
        elif color == 'YUV444' or color == 'YUV':
            Y.append(BGR2YUV(X))
        elif color == 'YUV420':
            X = BGR2YUV(X)
            Y.append(X[:, :, 0])
            U.append(block_reduce(X[:, :, 1], (2, 2), np.mean))
            V.append(block_reduce(X[:, :, 2], (2, 2), np.mean))
        else:
            logging.info('No such color type!, Color must be BGR, RGB, YUV, YUV444, or YUV420!')
            break
    if color == 'BGR' or color == 'RGB':
        return img
    elif color == 'YUV444' or color == 'YUV':
        return Y
    elif color == 'YUV420':
        return Y, U, V
