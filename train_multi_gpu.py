import os
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd

from gluoncv.utils import LRScheduler
from gluoncv.utils.metrics.voc_segmentation import batch_pix_accuracy, batch_intersection_union
from gluoncv.model_zoo.segbase import SoftmaxCrossEntropyLoss, SegEvalModel
from gluoncv.utils.parallel import DataParallelModel, DataParallelCriterion

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
                 ctx=[mx.gpu()],
                 norm_layer = gluon.nn.BatchNorm,
                 num_workers=4):

        if test_batch_size is None:
            test_batch_size = batch_size

        self.running_flag = flag
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size

        # dataset and dataloader
        train_dataset = VOCAugSegmentation(root=data_root, split=train_split)
        val_datset = VOCAugSegmentation(root=data_root, split=val_split)
        self.train_data = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, last_batch='rollover',
                                                num_workers=num_workers)
        self.eval_data = gluon.data.DataLoader(val_datset, test_batch_size,
                                               last_batch='keep', num_workers=num_workers)

        # create network
        model = DeepLabv3p(OS=train_OS, classes=21, use_global_stats=use_global_stats, norm_layer=norm_layer)
        print(model)

        # resume checkpoint if needed
        if resume is not None:
            if os.path.isfile(resume):
                model.load_params(resume, ctx=ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
        else:
            model.initialize(ctx=ctx)

        self.net = DataParallelModel(model, ctx, sync=True)
        self.evaluator = DataParallelModel(SegEvalModel(model), ctx)

        # create criterion
        self.criterion = DataParallelCriterion(SoftmaxCrossEntropyLoss(), ctx, sync=True)

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', baselr=learning_rate, niters=len(self.train_data),
                                        nepochs=epochs)
        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       {'lr_scheduler': self.lr_scheduler,
                                        'wd': weight_decay,
                                        'momentum': momentum,
                                        'multi_precision': True})

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.
        for i, (data, target) in enumerate(tbar):
            self.lr_scheduler.update(i, epoch)
            with autograd.record(train_mode=True):
                outputs = self.net(data)
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            for loss in losses:
                train_loss += loss.asnumpy()[0] / len(losses)
            self.optimizer.step(batch_size=self.batch_size) 
            tbar.set_description('Epoch %d, training loss %.3f' % (epoch, train_loss / (i + 1)))
            mx.nd.waitall()
            # break

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
            outputs = self.evaluator(x, y)
            for (correct, labeled, inter, union) in outputs:
                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description('Epoch %s, validation pixAcc: %.4f, mIoU: %.4f'%\
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()

        return pix_acc, mIoU

    def save_checkpoint(self, epoch, is_best=False):
        save_checkpoint(self.running_flag, self.net.module, epoch, self.checkpoint_interval, is_best)


def save_checkpoint(flag, net, epoch, checkpoint_interval, is_best=False):
    """Save Checkpoint"""
    directory = "runs/%s" % flag
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save_params(os.path.join(directory, "lastest.params"))
    if (epoch + 1) % checkpoint_interval == 0:
        net.save_params(os.path.join(directory, 'checkpoint_%s.params' % (epoch + 1)))
        print("Checkpoint saved.")
    if is_best:
        net.save_params(os.path.join(directory, 'best.params'))
        print("Best model saved.")


if __name__ == "__main__":
    FLAG = 'finetune_train_aug_multi_gpu'

    EPOCHS = 50
    BATCH = 16
    TEST_BATCH = 64
    TRAIN_SPLIT = 'train_aug'
    TRAIN_OS = 16
    USE_GLOBAL_STATS = True
    # DATA_ROOT = os.path.expanduser('~/myDataset/voc')
    DATA_ROOT = os.path.expanduser('~/.mxnet/datasets/voc')
    WEIGHTS = '../weights/pascal_train_aug.params'
    LR = 1.e-4
    CHECKPOINT_INTERVAL = 3
    N_GPUS = 4

    if N_GPUS == 1:
        norm_layer = gluon.nn.BatchNorm
        ctx = [mx.gpu()]
    else:
        norm_layer = gluon.contrib.nn.SyncBatchNorm
        ctx = [mx.gpu(i) for i in range(N_GPUS)]

    trainer = Trainer(flag=FLAG,
                      batch_size=BATCH,
                      epochs=EPOCHS,
                      resume=WEIGHTS,
                      learning_rate=LR,
                      train_OS=TRAIN_OS,
                      train_split=TRAIN_SPLIT,
                      test_batch_size=TEST_BATCH,
                      use_global_stats=USE_GLOBAL_STATS,
                      ctx=ctx, 
                      norm_layer=norm_layer,
                      data_root=DATA_ROOT,
                      checkpoint_interval=CHECKPOINT_INTERVAL)
    
    _, best_mIoU = trainer.validation("INIT")
    # best_mIoU = 0

    for epoch in range(EPOCHS):
        trainer.training(epoch)
        _, mIoU = trainer.validation(epoch)
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            is_best = True
            print("A new best! mIoU = %.4f" % mIoU)
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
