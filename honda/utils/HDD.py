import os 
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io, transform


# local host
root = "/Volumes/thesis/ms-thesis/DATASET/HONDA/final/"

# # server
# root = '/home/patel4db/honda/dataset/'

class HDD(Dataset):
    def __init__(self, root=root, phase='train', transform=None):

        self.Images, self.sensors = [], []

        self.root = root
        self.transform = transform

        self.img_folder = self.root + 'camera/'
        self.sensor_folder = self.root + 'sensor/'

        img_folders = sorted(os.listdir(self.root+'camera/'))
        sensor_folders = os.listdir(self.root+'sensor/')

        train_sessions = img_folders[:100]
        valid_sessions  = img_folders[101:]

        phase_session = train_sessions if phase=='train' else valid_sessions
        for sess in phase_session:
            self.session_id = sess
            sess_id = os.path.join(self.img_folder,sess)
            Y = np.load(os.path.join(self.sensor_folder,f'{sess}.npy'))
            idx = 0
            for imgs in sorted(os.listdir(sess_id)):
                img_path = np.array(os.path.join(sess_id, imgs))
                self.Images.append(img_path)
                self.sensors.append(Y[idx])
                idx+=1

        self.data = [(x,y) for x, y in zip(self.Images, self.sensors)]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img_to_read = self.data[index][0].item()
        img = io.imread(img_to_read)
        label = self.data[index][1]

        sample = {'image' : img, 'label' : label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label']


# Transforms
# -----------

'''
Transforming the images for feeding in to the network.
Most Neural Network expect the images of a fixed size.
Therefore, we will need to write some pre-processing code.

- Rescale  :: to scale the image
- Crop     :: to crop the ego-vehicle body and sky
- ToTensor :: to convert the numpy images to torch images (swap channels)

We will write them as callable classes instead of simple functions so that
parameters of the transform need not be passed everytime it's called. 
For this, we just need to implement __call__ method and if required,
__init__ method. We can then use a transform like this

::
    tsfm = Transform(params)
    transformed_sample :: tsfm(sample)
::

'''


class Rescale(object):
    '''
    Rescale the image in a sample to a given size.

    Args :
         | output_size (tuple) :: Desired output size.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.out_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        h, w = img.shape[:2]
        new_h, new_w = self.out_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))

        return {'image' : img, 'label' : label}


class CropImg(object):
    '''
    Crop ego-vehicle and sky in the image of the sample.

    Image size  :: H x W = 720 x 1280
    Cropped img :: H x W = (150 : 650) x 1280
    '''

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        img = img[144:656, :]

        return {'image' : img, 'label' : label}


class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors
    '''

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image :: H x W x C
        # torch image :: C x H x W
        img = img.transpose((2,0,1))

        return {'image' : torch.from_numpy(img), 
                'label' : torch.from_numpy(np.asarray(label))}

class Normalize(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['label']

        return {'image': image.type(torch.FloatTensor)/255,
                'label': mask.type(torch.FloatTensor)/255}


class BuildDataLoader(object):
    def __init__(self, dataset=HDD, scale=Rescale, crop=CropImg, totensor=ToTensor, normalize=Normalize, BS=None):
        self.dataset = dataset
        self.scale   = scale
        self.crop    = crop
        self.totensor = totensor
        self.normalize = Normalize
        self.BS = BS

    def build(self):

        data_transforms = transforms.Compose([
                                    self.crop(),
                                    self.scale((1280//2, 512//2)),
                                    self.totensor(),
                                    self.normalize()
                                            ])


        train_dataset = self.dataset(phase='train', transform=data_transforms)
        val_dataset = self.dataset(phase='val', transform=data_transforms)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.BS, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.BS, shuffle=True)

        return train_loader, val_loader

