import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
            #              name                        id    trainId   category            catId    hasInstances   ignoreInEval      color
        CityscapesClass(  'black_background'        ,  0 ,      0 , 'Black Background'        , 0       , False       , True     , (    0,   0,   0) ),
        CityscapesClass(  'instrument_shaft'        ,  1 ,      1 , 'Instrument'              , 1       , True        , False    , (    0, 255,   0) ),
        CityscapesClass(  'instrument_clasper'      ,  2 ,      2 , 'Instrument'              , 1       , True        , False    , (    0, 255, 255) ),
        CityscapesClass(  'instrument_wrist'        ,  3 ,      3 , 'Instrument'              , 1       , True        , False    , (  125, 255,  12) ),
        CityscapesClass(  'kidney_parenchyma'       ,  4 ,      4 , 'Kidney Parenchyma'       , 2       , True        , False    , (  255,  55,   0) ),
        CityscapesClass(  'covered_kidney'          ,  5 ,      5 , 'Covered Kidney'          , 3       , True        , False    , (   24,  55, 125) ),
        CityscapesClass(  'thread'                  ,  6 ,      6 , 'Thread'                  , 4       , True        , False    , (  187, 155,  25) ),
        CityscapesClass(  'clamps'                  ,  7 ,      7 , 'Clamps'                  , 5       , True        , False    , (    0, 255, 125) ),
        CityscapesClass(  'suturing_needle'         ,  8 ,      8 , 'Suturing Needle'         , 6       , True        , False    , (  255, 255, 125) ),
        CityscapesClass(  'suction_instrument'      ,  9 ,      9 , 'Instrument'              , 1       , True        , False    , (  123,  15, 175) ),
        CityscapesClass(  'small_intestine'         , 10 ,     10 , 'Small Intestine'         , 7       , True        , False    , (  124, 155,   5) ),
        CityscapesClass(  'ultrasound_probe'        , 11 ,     11 , 'Ultrasound Probe'        , 8       , True        , False    , (   12, 255, 141) ),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(f'{root}/gtFine/images', split)
        self.targets_dir = os.path.join(f'{root}/gtFine/labels', split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_endo_color_mask.png'.format(file_name.split('_endo.png')[0])
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 12
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
