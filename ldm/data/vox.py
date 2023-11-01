# from .base_dataset import BaseDataset
# from .base_dataset import pil_loader
import numpy as np
import io
import os
import os.path as osp
import torch
import importlib
import random
import cv2
import time
import csv
from collections import namedtuple
from tqdm import tqdm
import pickle
import math
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms

from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageFile
import io
import scipy.io.wavfile as wf
import time
import librosa
# import custom_transforms as tr

class BaseDataset(Dataset, ABC):
    def __init__(self, size):
        super().__init__()
        self.ignore_index = 255
        self.input_size = list(size)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train(self):
        temp = []
        temp.append(Resize(self.input_size))
        # temp.append(RandomHorizontalFlip())
        # temp.append(RandomRotate(15))
        # temp.append(RandomCrop(self.input_size))
        # temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def transform_validation(self):
        temp = []
        temp.append(Resize(self.input_size))
        # temp.append(tr.RandomCrop(self.input_size))
        # temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

class MMDataset(BaseDataset):
    def __init__(self, size=256, split='train', sample_mode='random', **kwargs):
        super().__init__(size)
        self.root_dir_image = '/mnt/data/voxceleb1/VGG_ALL_FRONTAL'
        self.root_dir_audio = '/mnt/data/voxceleb1/wav'
        self.list_dir = '/mnt/data/voxceleb1/data/list_wav'
        self.split = split
        dataset_train = 'vox1_train'
        dataset_val = 'vox1_val'
        dataset_test = 'vox1_test'
        if split == 'train':
            train_list = os.path.join(self.list_dir, dataset_train + '.csv')
            self.transform = self.transform_train()
            _data_list = train_list
        elif split == 'val':
            val_list = os.path.join(self.list_dir, dataset_val + '.csv')
            self.transform = self.transform_validation()
            _data_list = val_list
        elif split == 'test':
            test_list = os.path.join(self.list_dir, dataset_test + '.csv')
            self.transform = self.transform_validation()
            _data_list = test_list
        else:
            raise ValueError

        self.modal = 'both'

        self.rank = 0
        self.split = split

        def get_all_files(dir, ext):
            for e in ext:
                if dir.endswith(e):
                    return [dir]

            file_list = os.listdir(dir)
            ret = []
            for i in file_list:
                ret += get_all_files(osp.join(dir, i), ext)
            return ret

        self.metas = []
        self.id_mapping = dict()
        len_image = 0
        len_audio = 0

        with open(_data_list) as f:
            lines = csv.DictReader(f, delimiter=',')
            for line in lines:
                if line['Set'] == 'none':
                    continue
                face_id = line['VGGFace1_ID']
                audio_id = line['VoxCeleb1_ID']

                line['image_list'] = get_all_files(osp.join(self.root_dir_image, face_id), ['.jpg'])
                line['audio_list'] = get_all_files(osp.join(self.root_dir_audio, audio_id), ['.npy', '.wav'])

                if not face_id in self.id_mapping.keys():
                    self.id_mapping[face_id] = len(self.id_mapping)

                len_image += len(line['image_list'])
                len_audio += len(line['audio_list'])
                self.metas.append(line)

        if sample_mode == 'list' and (split == 'val' or split == 'test'):
            self.triplet_list = []
            with open(osp.join(self.list_dir, 'match_v2f_2_val' + '.txt'), 'r') as f:
                triplet_list = f.read().splitlines()

            filename2index = dict()
            for i in range(len(self.metas)):
                for j in range(len(self.metas[i]['audio_list'])):
                    filename = self.metas[i]['audio_list'][j]
                    filename = filename.replace(self.root_dir_audio + '/', '')
                    filename2index[filename] = (i, j)
                for j in range(len(self.metas[i]['image_list'])):
                    filename = self.metas[i]['image_list'][j]
                    filename = filename.replace(self.root_dir_image + '/', '')
                    filename2index[filename] = (i, j)

            self.triplet_list = []
            for i, item in enumerate(triplet_list):
                self.triplet_list.append([filename2index[ii] for ii in item.split(' ')])

            self.num = len(self.triplet_list)
        else:
            self.num = len(self.metas)
            self.triplet_list = None

        if self.rank == 0:
            print('%s set has %d images, %d audios, %d samples per epoch' % (
            self.split, len_image, len_audio, self.__len__()))

        self.initialized = False

    def __len__(self):
        return self.num

    def __str__(self):
        return 'vox' + '  split=' + str(self.split)

    def load_image(self, metas, image_id):
        image_filename = metas['image_list'][image_id]
        img = pil_loader(filename=image_filename)
        return img

    def load_audio(self, metas, audio_id):
        audio_filename = metas['audio_list'][audio_id]
        aud = pil_loader(filename=audio_filename)
        return aud

    def load(self, metas, image_id=-1, audio_id=-1, with_filename=False):
        if image_id < 0:
            image_id = np.random.choice(range(len(metas['image_list'])))

        if audio_id < 0:
            audio_id = np.random.choice(range(len(metas['audio_list'])))

        ID = metas['VGGFace1_ID']
        sample = {'ID': self.id_mapping[ID]}
        if self.modal == 'image' or self.modal == 'both':
            image = self.load_image(metas, image_id)
            sample['image'] = (image / 127.5 - 1.0).astype(np.float32)
            if with_filename:
                sample['image_filename'] = metas['image_list'][image_id].replace(self.root_dir_image + '/', '')
                sample['gender'] = metas['Gender']
        if self.modal == 'audio' or self.modal == 'both':
            sample['audio'] = self.load_audio(metas, audio_id)
            if with_filename:
                sample['audio_filename'] = metas['audio_list'][audio_id].replace(self.root_dir_audio + '/', '')
                sample['gender'] = metas['Gender']
        # TODO: hack during test
        sample['caption'] = sample['audio']

        return sample

    def getitem(self, idx):
        if self.triplet_list is not None:
            return self.getitem_triple(idx)
        else:
            sample = self.load(self.metas[idx])
        sample = self.transform(sample)
        return sample

    def getitem_triple(self, idx):
        triplet = self.triplet_list[idx]
        sample = self.load(self.metas[triplet[0][0]], audio_id=triplet[0][1])
        sample_p = self.load(self.metas[triplet[1][0]], image_id=triplet[1][1])
        sample_n = self.load(self.metas[triplet[2][0]], image_id=triplet[2][1])
        sample = self.transform(sample)
        sample_p = self.transform(sample_p)
        sample_n = self.transform(sample_n)
        return sample, sample_p, sample_n

    def getitem_all(self, idx):
        samples_audio = []
        samples_image = []
        for j in range(len(self.metas[idx]['image_list'])):
            sample = self.load(self.metas[idx], image_id=j, with_filename=True)
            del sample['audio']
            del sample['audio_filename']
            sample = self.transform(sample)
            samples_image += [sample]

        for j in range(len(self.metas[idx]['audio_list'])):
            sample = self.load(self.metas[idx], audio_id=j, with_filename=True)
            del sample['image']
            del sample['image_filename']
            sample = self.transform(sample)
            samples_audio += [sample]

        return samples_image, samples_audio

    def __getitem__(self, idx):
        return self.getitem(idx)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(filename, label=False):
    ext = os.path.splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:, :, ::-1]  # convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
        return img
    elif ext == '.wav':
        rate, data = wf.read(filename)
        if rate != 16000:
            raise RuntimeError('input wav must be sampled at 16,000 Hz, get %d Hz' % rate)
        if data.ndim > 1:
            # take the left channel
            data = data[:, 0]
        if data.shape[0] < 16000 * 10:
            # make the wav at least 10-second long
            data = np.tile(data, (16000 * 10 + data.shape[0] - 1) // data.shape[0])
        # take the first 10 seconds
        data = np.reshape(data[:16000 * 10], [-1]).astype(np.float32)
        return data
    elif ext == '.npy':
        data = np.load(filename, allow_pickle=True)
        return data.T.reshape((1, 64, -1))[:, :, :1000]
    else:
        raise NotImplementedError('Unsupported file type %s' % ext)


def wav2spec(wav):
    linear_spect = librosa.stft(wav, n_fft=512, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = list(sample.keys())
        for key in key_list:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            if 'image' == key:
                img = sample[key]
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=0).copy()
                else:
                    img = img.transpose((2, 0, 1)).copy()
                sample[key] = torch.from_numpy(img).float()
            elif 'audio' == key:
                aud = sample[key]
                sample[key] = torch.from_numpy(aud).float()
            elif 'label' in key:
                mask = sample[key]
                mask = np.expand_dims(mask, axis=0).copy()
                sample[key] = torch.from_numpy(mask).long()
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            key_list = sample.keys()
            for key in key_list:
                if 'image' not in key:
                    continue
                image = sample[key]
                image_flip = np.flip(image, axis=1)
                sample[key] = image_flip
        return sample


class RandomRotate(object):
    """Randomly rotate image"""

    def __init__(self, angle_r, image_value=127, is_continuous=True):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
        self.IMAGE_VALUE = image_value

    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        PI = 3.141592653
        Hangle = rand_angle * PI / 180
        Hcos = math.cos(Hangle)
        Hsin = math.sin(Hangle)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            image = sample[key]
            imgsize = image.shape
            srcWidth = imgsize[1]
            srcHeight = imgsize[0]
            x = [0, 0, 0, 0]
            y = [0, 0, 0, 0]
            x1 = [0, 0, 0, 0]
            y1 = [0, 0, 0, 0]
            x[0] = -(srcWidth - 1) / 2
            x[1] = -x[0]
            x[2] = -x[0]
            x[3] = x[0]
            y[0] = -(srcHeight - 1) / 2
            y[1] = y[0]
            y[2] = -y[0]
            y[3] = -y[0]
            for i in range(4):
                x1[i] = int(x[i] * Hcos + y[i] * Hsin + 0.5)
                y1[i] = int(-x[i] * Hsin + y[i] * Hcos + 0.5)
            if (abs(y1[2] - y1[0]) > abs(y1[3] - y1[1])):
                Height = abs(y1[2] - y1[0])
                Width = abs(x1[3] - x1[1])
            else:
                Height = abs(y1[3] - y1[1])
                Width = abs(x1[2] - x1[0])
            row, col = image.shape[:2]
            m = cv2.getRotationMatrix2D(center=(col / 2, row / 2), angle=rand_angle, scale=1)
            new_image = cv2.warpAffine(image, m, (Width, Height),
                                       flags=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation,
                                       borderValue=self.IMAGE_VALUE if 'image' in key else self.MASK_VALUE)
            sample[key] = new_image
        return sample


class Resize(object):
    def __init__(self, output_size, is_continuous=False, label_size=None):
        assert isinstance(output_size, (tuple, list))
        if len(output_size) == 1:
            self.output_size = (output_size[0], output_size[0])
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        if not 'image' in sample.keys():
            return sample
        img = sample['image']
        image_shape = list(img.shape)
        if self.output_size[0] / float(image_shape[0]) <= \
                self.output_size[1] / float(image_shape[1]):
            out_h = self.output_size[0]
            out_w = int(self.output_size[0] * image_shape[1] / image_shape[0])
        else:
            out_h = int(self.output_size[1] * image_shape[0] / image_shape[1])
            out_w = self.output_size[1]

        key_list = sample.keys()
        for key in key_list:
            if key != 'image':
                continue
            img = sample[key]
            h, w = img.shape[:2]
            img = cv2.resize(img, dsize=(out_w, out_h),
                             interpolation=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation)
            sample[key] = img

        return sample


class RandomCrop(object):
    def __init__(self, crop_size, image_value=127):
        assert isinstance(crop_size, (tuple, list))
        if len(crop_size) == 1:
            self.crop_size = (crop_size[0], crop_size[0])
        else:
            self.crop_size = crop_size
        self.IMAGE_VALUE = image_value

    def __call__(self, sample):
        rand_pad = random.uniform(0, 1)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            img = sample[key]
            h, w = img.shape[:2]
            new_h, new_w = self.crop_size
            pad_w = new_w - w
            pad_h = new_h - h
            w_begin = max(0, -pad_w)
            h_begin = max(0, -pad_h)
            pad_w = max(0, pad_w)
            pad_h = max(0, pad_h)
            w_begin = int(w_begin * rand_pad)
            h_begin = int(h_begin * rand_pad)
            w_end = w_begin + min(w, new_w)
            h_end = h_begin + min(h, new_h)
            shape = list(img.shape)
            shape[0] = new_h
            shape[1] = new_w
            new_img = np.zeros(shape, dtype=np.float)
            new_img.fill(self.IMAGE_VALUE)
            new_img[pad_h // 2:min(h, new_h) + pad_h // 2, pad_w // 2:min(w, new_w) + pad_w // 2] = img[h_begin:h_end,
                                                                                                    w_begin:w_end]
            sample[key] = new_img
        return sample

