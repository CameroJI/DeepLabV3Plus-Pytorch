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
                  #       name                        id    trainId   category               catId     hasInstances   ignoreInEval   color
        CityscapesClass(  'unlabeled'               ,  0 ,    255 , 'frame'                   , 0       , False       , True     , (  255, 255, 255) ),
        CityscapesClass(  'black_background'        ,  1 ,      0 , 'Black Background'        , 1       , False       , True     , (  127, 127, 127) ),
        CityscapesClass(  'liver'                   ,  2 ,      1 , 'Liver'                   , 2       , True        , False    , (  255, 114, 114) ),
        CityscapesClass(  'gastrointestinal_tract'  ,  3 ,      2 , 'Gastrointestinal Tract'  , 3       , True        , False    , (  231,  70, 156) ),
        CityscapesClass(  'fat'                     ,  4 ,      3 , 'Fat'                     , 4       , True        , False    , (  186, 183,  75) ),
        CityscapesClass(  'grasper'                 ,  5 ,      4 , 'Instrument'              , 5       , True        , False    , (  170, 255,   0) ),
        CityscapesClass(  'connective_tissue'       ,  6 ,      5 , 'Connective Tissue'       , 6       , True        , False    , (  255,  85,   0) ),
        CityscapesClass(  'abdominal_wall'          ,  7 ,      6 , 'Abdominal Wall'          , 0       , True        , False    , (  210, 140, 140) ),
        CityscapesClass(  'blood'                   ,  8 ,      7 , 'Blood'                   , 7       , True        , False    , (  255,   0,   0) ),
        CityscapesClass(  'cystic_duct'             ,  9 ,      8 , 'Cystic Duct'             , 8       , True        , False    , (  255, 255,   0) ),
        CityscapesClass(  'l-hook_electrocautery'   , 10 ,      9 , 'Instrument'              , 5       , True        , False    , (  169, 255, 184) ),
        CityscapesClass(  'hepatic_vein'            , 11 ,     10 , 'Hepatic Vein'            , 11      , True        , False    , (    0,  50, 128) ),
        CityscapesClass(  'gallbladder'             , 12 ,     11 , 'Gallbladder'             , 10      , True        , False    , (  255, 160, 165) ),
        CityscapesClass(  'liver_ligament'          , 13 ,     12 , 'Liver Ligament'          , 12      , True        , False    , (  111,  74,   0) ),
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
        target[target == 255] = 13
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
