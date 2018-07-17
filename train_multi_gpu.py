raise NotImplementedError

import os
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd

from gluoncv.utils import LRScheduler
from gluoncv.utils.metrics.voc_segmentation import batch_pix_accuracy, batch_intersection_union
from gluoncv.model_zoo.segbase import SoftmaxCrossEntropyLoss

from mylib.deeplabv3p import DeepLabv3p
from mylib.dataset import VOCAugSegmentation


class Trainer(object):
    def __init__(self, flag, batch_size,
                 use_global_stats=True,
                 checkpoint_interval=5,
                 epochs=50,
                 learning_rate=1.e-4,
                 momentum=0.9,
                 weight_decay=1.e-4,
                 train_OS=16,
                 train_split='train_aug',
                 val_split='val',
                 resume=None,
                 test_batch_size=None,
                 data_root=os.path.expanduser('~/.mxnet/datasets/voc'),
                 num_workers=4):

        if test_batch_size is None:
            test_batch_size = batch_size

        self.running_flag = flag
        self.checkpoint_interval = checkpoint_interval

        # dataset and dataloader
        train_dataset = VOCAugSegmentation(root=data_root, split=train_split)
        val_datset = VOCAugSegmentation(root=data_root, split=val_split)
        self.train_data = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, last_batch='rollover',
                                                num_workers=num_workers)
        self.eval_data = gluon.data.DataLoader(val_datset, test_batch_size,
                                               last_batch='keep', num_workers=num_workers)

        # create network
        model = DeepLabv3p(OS=train_OS, classes=21, use_global_stats=use_global_stats)
        self.net = model
        print(model)

        # resume checkpoint if needed
        if resume is not None:
            if os.path.isfile(resume):
                model.load_params(resume, ctx=mx.gpu())
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
        else:
            model.initialize(ctx=mx.gpu())

        # create criterion
        self.criterion = SoftmaxCrossEntropyLoss()

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', baselr=learning_rate, niters=len(self.train_data),
                                        nepochs=epochs)
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',
                                       {'lr_scheduler': self.lr_scheduler,
                                        'wd': weight_decay,
                                        'momentum': momentum,
                                        'multi_precision': True})

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.
        for i, (data, target) in enumerate(tbar):
            data = data.copyto(mx.gpu())
            target = target.copyto(mx.gpu())
            self.lr_scheduler.update(i, epoch)
            with autograd.record(True):
                outputs = self.net(data)
                losses = self.criterion(outputs, target)
                loss = losses.mean()
                mx.nd.waitall()
                loss.backward()
            self.optimizer.step(batch_size=1)  # dummy expression
            train_loss += loss.asscalar()
            tbar.set_description('Epoch %d, training loss %.3f' % (epoch, train_loss / (i + 1)))
            mx.nd.waitall()
            # break

        # save every epoch
        save_checkpoint(self.running_flag, self.net, epoch, self.checkpoint_interval)

    def validation(self, epoch, train=False):
        if train:
            loader = self.train_data
            flag = "train"
        else:
            loader = self.eval_data
            flag = 'val'

        tbar = tqdm(loader)
        total_inter, total_union, total_correct, total_label = (0,) * 4
        for i, (x, y) in enumerate(tbar):
            x = x.copyto(mx.gpu())
            y = y.copyto(mx.gpu())
            pred = self.net(x)
            correct, labeled = batch_pix_accuracy(output=pred, target=y)
            inter, union = batch_intersection_union(output=pred, target=y, nclass=21)
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pix_acc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            tbar.set_description('%s - Epoch %s, pix_acc: %.4f, mIoU: %.4f' % (flag, epoch, pix_acc, mIoU))
            mx.nd.waitall()
            # break


def save_checkpoint(flag, net, epoch, checkpoint_interval):
    """Save Checkpoint"""
    directory = "runs/%s" % flag
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save_params(os.path.join(directory, "lastest.params"))
    if (epoch + 1) % checkpoint_interval == 0:
        net.save_params(os.path.join(directory, 'checkpoint_%s.params' % (epoch + 1)))


if __name__ == "__main__":
    FLAG = 'finetune_train_aug'

    EPOCHS = 50
    BATCH = 6
    TEST_BATCH = 16
    TRAIN_SPLIT = 'train_aug'
    TRAIN_OS = 16
    USE_GLOBAL_STATS = True
    WEIGHTS = '../weights/pascal_train_aug.params'
    LR = 1.e-4

    trainer = Trainer(flag=FLAG,
                      batch_size=BATCH,
                      epochs=EPOCHS,
                      resume=WEIGHTS,
                      learning_rate=LR,
                      train_OS=TRAIN_OS,
                      train_split=TRAIN_SPLIT,
                      test_batch_size=TEST_BATCH,
                      use_global_stats=USE_GLOBAL_STATS,
                      checkpoint_interval=10)
    trainer.validation("INIT")

    for epoch in range(EPOCHS):
        trainer.training(epoch)
        trainer.validation(epoch)
