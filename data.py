import numpy as np
from utils import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy
import os
import random
import csv
import matplotlib.pyplot as plt
import cv2

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    # maps the name of the file to the bytes of the image
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_data(batch_size, data_train_csv, test_csv_file):
    with open(data_train_csv, encoding="utf-8", newline="\n") as train_csv:
        train = list((row.strip("\n").strip("\r").split(',') for row in (train_csv) if len(row) > 0))
    with open(test_csv_file, encoding="utf-8", newline="\n") as test_csv:
        test = list((row.strip("\n").split(',') for row in (test_csv) if len(row) > 0))

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if False:
        train = train[:10]
        test = test[:10]

    return train, test, shape_rgb, shape_depth

def get_train_test_data(batch_size, datadir='./', data_csv_file='data/nyu2_train.csv', test_csv_file='data/nyu2_test.csv', nyuTest=False, sigmaRGB=0, sigmaD=0):
    data_train_csv = os.path.join(datadir, data_csv_file)
    test_csv_file = os.path.join(datadir, test_csv_file)

    train, test, shape_rgb, shape_depth = get_data(batch_size, data_train_csv=data_train_csv, test_csv_file=test_csv_file)

    for i in range(len(train)):
        train[i][0] = os.path.join(datadir, train[i][0])
        train[i][1] = os.path.join(datadir, train[i][1])

    for i in range(len(test)):
        test[i][0] = os.path.join(datadir, test[i][0])
        test[i][1] = os.path.join(datadir, test[i][1])

    train_generator = NYU_BasicAugmentRGBSequence(train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, sigmaRGB=sigmaRGB, sigmaD=sigmaD)

    if (nyuTest):
        test_generator = NYU_BasicRGBSequence(test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    else:
        test_generator = NYU_BasicAugmentRGBSequence(test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    #train_generator.__getitem__(0, showImage=True)
    test_generator.__getitem__(0)

    return train_generator, test_generator

def get_evaluation_data(eval_csv, datadir="./"):
    eval_csv_file = os.path.join(datadir, eval_csv)
    with open(eval_csv_file, encoding="utf-8", newline="\n") as eval_file:
        eval_rows = list((row.strip("\n").strip("\r").split(',') for row in (eval_file) if len(row) > 0))
    num_images = len(eval_rows)
    shape_rgb = (num_images, 480, 640, 3)
    shape_depth = (num_images, 480, 640)

    # read in images and cat rgb and depth
    rgb = np.zeros(shape_rgb, dtype=np.uint8)
    depth = np.zeros(shape_depth, dtype=np.float32)

    for i in range(num_images):
        sample = eval_rows[i]
        rgb_path = sample[0].strip()
        gt_path = sample[1].strip()

        x = np.clip(np.asarray(Image.open(datadir + rgb_path), dtype=np.uint8).reshape(480, 640, 3), 0, 255)

        y = np.asarray(Image.open(datadir + gt_path), dtype=np.float32).reshape(480, 640).copy().astype(float)* (10.0/255.0)

        rgb[i] = x
        depth[i] = y
    
    return {'rgb': rgb, 'depth': depth, 'crop': None}

def createBorder(x, width, color):

    if (len(x.shape) == 3):
        x[0:width,:,:] = color
        x[:,0:width,:] = color
        x[-width:-1,:,:] = color
        x[:,-width:-1,:] = color
    else:
        x[0:width,:] = color
        x[:,0:width] = color
        x[-width:-1,:] = color
        x[:,-width:-1] = color

    return x

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, sigmaRGB=0, sigmaD=0):
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.sigmaRGB = sigmaRGB
        self.sigmaD = sigmaD

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True, showImage=False):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            # get path of the rgb and ground truth image
            sample = self.dataset[index]
            rgb_path = sample[0].strip()
            gt_path = sample[1].strip()

            x = np.array(Image.open(rgb_path)).reshape(480,640,3)
            y = np.array(Image.open(gt_path)).reshape(480,640,1)

            if (self.sigmaRGB > 0):
                sz = int(np.ceil(6*np.ceil(self.sigmaRGB)) + 1)
                x = cv2.GaussianBlur(x, (sz,sz), self.sigmaRGB)
            if (self.sigmaD > 0):
                sz = int(np.ceil(6*np.ceil(self.sigmaD)) + 1)
                y = np.expand_dims(cv2.GaussianBlur(y, (sz,sz), self.sigmaD), 2)

            if (showImage and i==0):
                plt.subplot(1,2,1)
                plt.imshow(x)    
                plt.title(self.sigmaRGB)
                plt.show(block=False)

                plt.subplot(1,2,2)
                plt.imshow(np.squeeze(y))    
                plt.title(self.sigmaD)
                plt.show(block=False)

                plt.waitforbuttonpress()

            x = createBorder(x, 8, 255)
            y = createBorder(y, 8, 64)

            x = np.clip(x/255,0,1)
            y = np.clip(y/255*self.maxDepth,1,self.maxDepth)

            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, dataset, batch_size,shape_rgb, shape_depth):
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            rgb_path = sample[0].strip()
            gt_path = sample[1].strip()

            x = np.clip(np.asarray(Image.open( rgb_path)).reshape(480,640,3)/255,0,1)
            y = np.asarray(Image.open(gt_path), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0

            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y