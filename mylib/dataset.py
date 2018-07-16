"""Pascal VOC Semantic Segmentation Dataset."""

import os
import random
import scipy.io
import numpy as np
from mxnet import cpu
import mxnet.ndarray as F
from PIL import Image, ImageOps, ImageFilter
from gluoncv.data.segbase import SegmentationDataset


class VOCAugSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset with train_aug (from SBD) augmented dataset."""

    VOC_BASE_DIR = 'VOC2012'
    SBD_BASE_DIR = 'VOCaug/dataset'
    NUM_CLASS = 21

    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/voc'),
                 split='train', mode=None, transform=lambda x: x.transpose((2, 0, 1)) / 127.5 - 1.,
                 base_size=520, crop_size=512):
        super(VOCAugSegmentation, self).__init__(root, split, mode, transform, base_size, crop_size)

        _voc_root = os.path.join(root, self.VOC_BASE_DIR)
        _voc_image_dir = os.path.join(_voc_root, 'JPEGImages')
        _voc_mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _voc_splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')

        _sbd_root = os.path.join(root, self.SBD_BASE_DIR)
        _sbd_image_dir = os.path.join(_sbd_root, 'img')
        _sbd_mask_dir = os.path.join(_sbd_root, 'cls')

        if split == 'train':
            self.mode = mode or "train"
            n_samples = 1464
            _split_f = os.path.join(_voc_splits_dir, 'train.txt')
            self.images, self.masks = self._setup_voc(_split_f, _voc_image_dir, _voc_mask_dir)
        elif split == 'val':
            self.mode = mode or "val"
            n_samples = 1449
            _split_f = os.path.join(_voc_splits_dir, 'val.txt')
            self.images, self.masks = self._setup_voc(_split_f, _voc_image_dir, _voc_mask_dir)
        elif split == "trainval":
            self.mode = mode or "train"
            n_samples = 2913
            _split_f = os.path.join(_voc_splits_dir, 'trainval.txt')
            self.images, self.masks = self._setup_voc(_split_f, _voc_image_dir, _voc_mask_dir)
        elif split == 'train_aug':
            self.mode = mode or "train"
            n_samples = 10582
            _split_f = os.path.join(_voc_splits_dir, 'train.txt')
            self.images, self.masks = self._setup_voc(_split_f, _voc_image_dir, _voc_mask_dir)
            _voc_all_f = os.path.join(_voc_splits_dir, 'trainval.txt')
            _sbd_all_f = os.path.join(_sbd_root, 'trainval.txt')
            images_plus, masks_plus = self._setup_voc_aug(_sbd_all_f, _voc_all_f, _sbd_image_dir, _sbd_mask_dir)
            self.images += images_plus
            self.masks += masks_plus
        else:
            assert split == 'test'  # only available from the PASCAL challenge website after registering.
            self.mode = mode or "test"
            n_samples = 1456
            _split_f = os.path.join(_voc_splits_dir, 'test.txt')
            self.images, self.masks = self._setup_voc(_split_f, _voc_image_dir, _mask_dir=None)

        if split != 'test':
            assert (len(self.images) == len(self.masks))

        assert len(self.images) == n_samples

    def _setup_voc(self, _split_f, _image_dir, _mask_dir):
        images = []
        masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                sample_id = line.rstrip('\n')
                _image = os.path.join(_image_dir, sample_id + ".jpg")
                assert os.path.isfile(_image)
                images.append(_image)
                if _mask_dir is not None:
                    _mask = os.path.join(_mask_dir, sample_id + ".png")
                    assert os.path.isfile(_mask)
                    masks.append(_mask)
        if _mask_dir is not None:
            return images, masks
        return images

    def _setup_voc_aug(self, _sbd_f, _voc_f, _image_dir, _mask_dir):
        voc_all = set()
        with open(_voc_f) as lines:
            for line in lines:
                voc_all.add(line.rstrip('\n'))
        images_plus = []
        masks_plus = []
        with open(_sbd_f) as lines:
            for line in lines:
                sample_id = line.rstrip('\n')
                if sample_id not in voc_all:
                    _image = os.path.join(_image_dir, sample_id + ".jpg")
                    assert os.path.isfile(_image)
                    images_plus.append(_image)
                    _mask = os.path.join(_mask_dir, sample_id + ".mat")
                    assert os.path.isfile(_mask)
                    masks_plus.append(_mask)
        return images_plus, masks_plus

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask_file = self.masks[index]
        if mask_file.endswith(".mat"):
            mask = self._load_mat(mask_file)
        else:
            mask = Image.open(mask_file)

        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            pass

        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return F.array(target, cpu(0))

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size  # the size to be padding
        h, w = img.size
        pad_h = outsize - h
        pad_w = outsize - w
        assert pad_h >= 0
        assert pad_w >= 0
        img = ImageOps.expand(img, border=(0, 0, pad_h, pad_w), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, pad_h, pad_w), fill=0)
        img = self._img_transform(img)
        mask = self._mask_transform(mask)
        mask[-pad_h:, :] = -1  # mask the padding mask with -1
        mask[:, -pad_w:] = -1  # mask the padding mask with -1
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')


if __name__ == '__main__':
    dataset = VOCAugSegmentation(split='trainval')
    print(len(dataset))
    print(dataset[3])
