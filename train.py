from mxnet import gluon
import mxnet as mx
import numpy as np
from tqdm import tqdm
from gluoncv.utils.metrics.voc_segmentation import batch_pix_accuracy, batch_intersection_union