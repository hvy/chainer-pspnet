#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from math import ceil

import numpy as np
from skimage import io

import chainer
import chainer.functions as F
from chainer import serializers
from chainer.datasets import TransformDataset
from chainercv import transforms
from chainercv.links import pspnet
from datasets import cityscapes_labels
from datasets import CityscapesSemanticSegmentationDataset
from datasets import VOCSemanticSegmentationDataset


def inference(model, n_class, base_size, crop_size, img, scales):
    pred = np.zeros((n_class, img.shape[1], img.shape[2]))
    if scales is not None and isinstance(scales, (list, tuple)):
        for i, scale in enumerate(scales):
            pred += scale_process(
                model, img, n_class, base_size, crop_size, scale)
        pred = pred / float(len(scales))
    else:
        pred = scale_process(model, img, n_class, base_size, crop_size, 1.0)
    pred = np.argmax(pred, axis=0)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument(
        '--dataset', type=str, choices=['voc2012', 'cityscapes', 'ade20k'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--stride_rate', type=float, default=2 / 3)
    parser.add_argument('--save_test_image', action='store_true', default=False)
    args = parser.parse_args()

    chainer.config.stride_rate = args.stride_rate
    chainer.config.save_test_image = args.save_test_image

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.color_out_dir is not None:
        if not os.path.exists(args.color_out_dir):
            os.mkdir(args.color_out_dir)

    if args.model == 'VOC':
        n_class = 21
        n_blocks = [3, 4, 23, 3]
        feat_size = 60
        mid_stride = True
        param_fn = 'weights/pspnet101_VOC2012_473_reference.chainer'
        base_size = 512
        crop_size = 473
        dataset = VOCSemanticSegmentationDataset(
            args.voc_data_dir, split='test')
    elif args.model == 'Cityscapes':
        n_class = 19
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        base_size = 2048
        crop_size = 713
        dataset = CityscapesSemanticSegmentationDataset(
            args.cityscapes_img_dir, None, args.split)
    elif args.model == 'ADE20K':
        n_class = 150
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
        base_size = 512
        crop_size = 473

    dataset = TransformDataset(dataset, preprocess)
    print('dataset:', len(dataset))

    chainer.config.train = False
    model = PSPNet(n_class, n_blocks, feat_size, mid_stride=mid_stride)
    serializers.load_npz(param_fn, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu)
        model.to_gpu(args.gpu)

    for i in range(args.start_i, args.end_i + 1):
        img = dataset[i]
        out_fn = os.path.join(
            args.out_dir, os.path.basename(dataset._dataset.img_fns[i]))
        pred = inference(
            model, n_class, base_size, crop_size, img, args.scales)
        assert pred.ndim == 2

        if args.model == 'Cityscapes':
            if args.color_out_dir is not None:
                color_out = np.zeros(
                    (pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            label_out = np.zeros_like(pred)
            for label in cityscapes_labels:
                label_out[np.where(pred == label.trainId)] = label.id
                if args.color_out_dir is not None:
                    color_out[np.where(pred == label.trainId)] = label.color
            pred = label_out

            if args.color_out_dir is not None:
                base_fn = os.path.basename(dataset._dataset.img_fns[i])
                base_fn = os.path.splitext(base_fn)[0]
                color_fn = os.path.join(args.color_out_dir, base_fn)
                color_fn += '_color.png'
                io.imsave(color_fn, color_out)
        io.imsave(out_fn, pred)
