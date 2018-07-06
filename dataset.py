"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import threading

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos

def get_test_frame_index(num_frames, seg, representation, num_segments):
    if representation in ['mv', 'residual']:
        num_frames -= 1

    seg_size = float(num_frames - 1) / num_segments
    v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

    if representation in ['mv', 'residual']:
        v_frame_idx += 1

    return get_gop_pos(v_frame_idx, representation)

def get_train_frame_index(num_frames, seg,representation,num_segments):
    # Compute the range of the segment.
    seg_begin, seg_end = get_seg_range(num_frames, num_segments, seg,
                                             representation=representation)

    # Sample one frame from the segment.
    v_frame_idx = random.randint(seg_begin, seg_end - 1)
    return get_gop_pos(v_frame_idx, representation)

def load_segment(is_train,num_frames,seg,representation,num_segments,video_path,representation_idx,accumulate):
    if is_train:
        gop_index, gop_pos = get_train_frame_index(num_frames, seg, representation, num_segments)
    else:
        gop_index, gop_pos = get_test_frame_index(num_frames, seg, representation, num_segments)

    img = load(video_path, gop_index, gop_pos,
               representation_idx, accumulate)

    if img is None:
        print('Error: loading video %s failed.' % video_path)
        img = np.zeros((256, 256, 2)) if representation == 'mv' else np.zeros((256, 256, 3))
    else:
        if representation == 'mv':
            img = clip_and_scale(img, 20)
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
        elif representation == 'residual':
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

    if representation == 'iframe':
        if is_train:
            img = color_aug(img)

        # BGR to RGB. (PyTorch uses RGB according to doc.)
        img = img[..., ::-1]
    return img

class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))

        print('%d videos loaded.' % len(self._video_list))

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0


        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        threads=[]
        for seg in range(self._num_segments):
            thread=MyThread(func=load_segment,args=(self._is_train,num_frames,seg,self._representation,self._num_segments,video_path,representation_idx,self._accumulate))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
            frames.append(thread.get_result())

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input, label

    def __len__(self):
        return len(self._video_list)
