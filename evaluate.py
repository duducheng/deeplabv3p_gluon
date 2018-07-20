from mxnet import gluon
import mxnet as mx
import numpy as np
from tqdm import tqdm
from gluoncv.utils.metrics.voc_segmentation import batch_pix_accuracy, batch_intersection_union

from mylib.deeplabv3p import DeepLabv3p
from mylib.dataset import VOCAugSegmentation


def test_loader(loader, model):
    tbar = tqdm(loader)
    total_inter, total_union, total_correct, total_label = (0,) * 4
    for i, (x, y) in enumerate(tbar):
        x = x.copyto(mx.gpu())
        y = y.copyto(mx.gpu())
        pred = model(x)
        correct, labeled = batch_pix_accuracy(output=pred, target=y)
        inter, union = batch_intersection_union(output=pred, target=y, nclass=21)
        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        pix_acc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        tbar.set_description('pix_acc: %.4f, mIoU: %.4f' % (pix_acc, mIoU))
    return pix_acc, mIoU


if __name__ == '__main__':
    SPLIT = 'val'
    EVALSIZE = 512
    EVALBATCH = 8
    EVALOS = 16
    WEIGHTS = '/home/jiancheng/Downloads/best.params'

    dataset = VOCAugSegmentation(split=SPLIT, crop_size=EVALSIZE, mode='val',
                                 transform=lambda x: x.transpose((2, 0, 1)) / 127.5 - 1.)
    dataloader = gluon.data.DataLoader(dataset, batch_size=EVALBATCH)

    model = DeepLabv3p(OS=EVALOS)
    model.load_params(filename=WEIGHTS, ctx=mx.gpu())

    pix_acc, mIoU = test_loader(dataloader, model)

    print('%s-OS=%s pix_acc: %.4f, mIoU: %.4f' % (SPLIT, EVALOS, pix_acc, mIoU))
