#!/usr/bin/env python
#coding:utf-8
#@Time: 2019/5/1713:58
#@Author: wangximei
#@File: utils.py
#@describtion:

import argparse
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.misc

import scipy.misc
from skimage import transform, filters


def get_blend_map(img, att_map, blur=True, overlap=True):
    # att_map -= att_map.min()
    # if att_map.max() > 0:
    #     att_map /= att_map.max()
    # att_map = transform.resize(att_map, (img.shape[:2]), order=3)
    if blur:
        att_map = filters.gaussian(att_map, 0.02 * max(img.shape[:2]))
        att_map -= att_map.min()
        att_map = att_map / att_map.max()
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = 1 * (1 - att_map ** 0.7).reshape(att_map.shape + (1,)) * img + (att_map ** 0.7).reshape(
            att_map.shape + (1,)) * att_map_v
    return att_map

def visualize_and_save(input, image_path, att, test_epoch, ckpt_dir, channel_index):
    # image_path = "/data/office-home/images/Product/Bike/00001.jpg"
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
    # orig_img = scipy.misc.imread(image_path)
    # orig = scipy.misc.imresize(orig_img, (256, 256), interp='bicubic')
    # start_x = (256 - 224 - 1) // 2
    # start_y = (256 - 224 - 1) // 2
    # orig = orig[start_x:start_x + 224, start_y:start_y + 224, :]
    # print("max(input)", max(max(input)))
    # print("min(input)", min(min(input)))

    # print("input.shape:", input.shape)
    input = input.permute(1,2,0) ## 3*224*224 ==> 224*224*3
    # print("before: ",input[0,0,:])
    mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float().cuda()
    std = torch.tensor([[[0.229, 0.224, 0.225]]]).float().cuda()
    input = input * std + mean
    input = input.cpu().numpy()
    # print("after: ",input[0,0,:])

    atten = scipy.misc.imresize(att, (224, 224), interp='bicubic')
    ax = plt.subplot(1, 3, 1)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.imshow(input)

    ax = plt.subplot(1, 3, 2)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.imshow(atten, alpha=1.0, cmap=plt.cm.Reds)
    # ax.set_xlabel("just for test", fontsize=10, labelpad=5, ha='center')

    ax = plt.subplot(1, 3, 3)
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    ax.imshow(get_blend_map(input, atten))
    # plt.show()
    output_dir = os.path.join(ckpt_dir, "test_epoch_%d" % test_epoch, str(channel_index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = "_".join(image_path.split('/')[-3:]).strip(".jpg")
    output_path = os.path.join(output_dir, image_name + '.pdf')
    fig.savefig(output_path, bbox_inches='tight')