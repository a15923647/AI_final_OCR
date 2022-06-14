import os
import glob
import pickle
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import numpy as np

class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, 
            root_dir=None, 
            mode=None, 
            paths=None, 
            img_height=32, 
            img_width=100, 
            img_limit=100000, 
            use_mem_buffer=True, 
            start_sample_index=0,
            image_preprocess=lambda x: x, 
            dump_dest="", 
            dump_name_suffix="", 
            save_pkl=True):

        self.start_sample_index = start_sample_index
        self.mode = mode
        self.img_limit = img_limit
        self.use_mem_buffer = use_mem_buffer
        self.image_preprocess = image_preprocess
        self.save_pkl = save_pkl
        
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None
        
        self.buf = [None] * len(paths) if self.use_mem_buffer else None
        self.loaded = 0
        self.img_limit = len(paths)
        self.dump_path = os.path.join(dump_dest, f"Synth90kDataset_{self.mode}_buf_{self.img_limit}{dump_name_suffix}.pkl")
        self.dump_file_exists = os.path.exists(self.dump_path)
        if self.dump_file_exists:
            print(f"load from buf dumped file: {self.dump_path}")
            with open(self.dump_path, 'rb') as fin:
                self.buf = pickle.load(fin)
        
        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines()[self.start_sample_index:self.start_sample_index+self.img_limit]:
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.use_mem_buffer and self.buf[index]:
            return self.buf[index]
        
        self.loaded += 1
        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        
        print(f"{self.loaded} / {self.img_limit} new image, index={index}, path: {path}")
        if self.loaded == self.img_limit and not self.dump_file_exists and self.save_pkl:
            with open(self.dump_path, 'wb') as fout:
                print(f"save buf to {self.dump_path}")
                pickle.dump(self.buf, fout)
            self.dump_file_exists = True
        
        
        image = self.image_preprocess(image)
        
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            if self.use_mem_buffer:
                self.buf[index] = [image, target, target_length]
            return image, target, target_length
        else:
            if self.use_mem_buffer:
                self.buf[index] = [image]
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

def labels2word(inputs, label2char=Synth90kDataset.LABEL2CHAR):
    return "".join([label2char[l] for l in inputs])
