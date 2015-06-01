import multiprocessing as mp
import Queue
import threading
import lasagne as nn
import numpy as np
import theano.tensor as T
import time
import os
import sys
import importlib
import cPickle as pickle
from datetime import datetime, timedelta
import string
from itertools import izip, repeat, chain
from collections import defaultdict
import random
from PIL import Image
import numpy
import theano
import copy
import skimage
import logging
import cStringIO
import urllib2
import cv2
from os import error
import glob
import os
import skimage.io
import skimage.transform



# define params and other errata
#import realtime_aug
import nn_plankton
import tmp_dnn
import utils
############################################
### Sander defined methods - Start
############################################

# Conv2DLayer = tmp_dnn.Conv2DDNNLayer
# MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer
#real-time aug based on Sander Dielman's KDSB solution

default_augmentation_params = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': False,
    'allow_stretch': False,
}
sfs = [1.0]
patch_sizes = [(200,200)]
rng_aug = np.random

def one_hot(vec, m=212):
    if m is None:
        m = int(np.max(vec)) + 1
    return np.eye(m)[vec]

def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
def hostname():
    return platform.node()
    
def generate_expid(arch_name):
    return "%s-%s-%s" % (arch_name, hostname(), timestamp())

def log_losses(y, t, eps=1e-15):
    if t.ndim == 1:
        t = one_hot(t)
    y = np.clip(y, eps, 1 - eps)
    losses = -np.sum(t * np.log(y), axis=1)
    return losses

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.mean(losses)

def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

def softmax(x): 
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)

def entropy(x):
    h = -x * np.log(x)
    h[np.invert(np.isfinite(h))] = 0
    return h.sum(1)


def conf_matrix(p, t, num_classes):
    if p.ndim == 1:
        p = one_hot(p, num_classes)
    if t.ndim == 1:
        t = one_hot(t, num_classes)
    return np.dot(p.T, t)

def accuracy_topn(y, t, n=5):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    predictions = np.argsort(y, axis=1)[:, -n:]    
    accs = np.any(predictions == t[:, None], axis=1)
    return np.mean(accs)

def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]
    return current_lr


def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


def log_loss_std(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.std(losses)


def buffered_gen_mp(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = mp.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_process(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
        buffer.close() # unfortunately this does not suffice as a signal: if buffer.get()
        # was called and subsequently the buffer is closed, it will block forever.
 
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
 
    for data in iter(buffer.get, None):
        yield data

def buffered_gen_threaded(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate thread. Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
 
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data

############################################
### Sander defined methods - End
############################################


############################################
### realtime_aug defined methods - start
############################################



def build_augmentation_transform(zoom=(1, 1), rotation=0, shear=0, translation=(0, 0), flip=False): 
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment

def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is
    # fastwarp seems to corrupt image using int8 transforms
    # TODO further research here
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2#.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter

def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't 
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))
    # centering    
    shift_x = int(cols / (2 * downscale_factor) - tcols / 2)
    shift_y = int(rows / (2 * downscale_factor) - trows / 2)
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


class Random_perturbation_transform(object):
    """ decided to put it in class to retain the randomly generated values"""
    def __init__(self, zoom_range, rotation_range, shear_range, translation_range, do_flip=False, allow_stretch=False, rng=np.random):
        self.shift_x = int(rng.uniform(*translation_range))
        self.shift_y = int(rng.uniform(*translation_range))
        self.translation = (self.shift_x, self.shift_y)
        self.rotation = int(rng.uniform(*rotation_range))
        self.shear = int(rng.uniform(*shear_range))
        if do_flip:
            self.flip = (rng.randint(2) > 0) # flip half of the time
        else:
            self.flip = False
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(rng.uniform(*log_zoom_range))
            stretch = np.exp(rng.uniform(*log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(rng.uniform(*log_zoom_range))
            zoom_y = np.exp(rng.uniform(*log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        self.zoom = (zoom_x, zoom_y)

    def random_perturbation_transform(self):
        return build_augmentation_transform(self.zoom, self.rotation, self.shear, self.translation, self.flip)


#not working properly. colors are not meshing well
def perturb_multiscale_new(img, scale_factors, augmentation_params, target_shapes=[(100,100)], rng=np.random):
    """
    scale is a DOWNSCALING factor.
    """

    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    color_channels = [b,g,r]
    output = []

    #local, but globals
    rpt = Random_perturbation_transform(rng=rng, **augmentation_params)
    tform_center, tform_uncenter = build_center_uncenter_transforms(b.shape)

    for channel in color_channels:
        #set operations using b color channel (can use any here)
        tform_augment = rpt.random_perturbation_transform()
        tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)

        for scale, target_shape in zip(scale_factors, target_shapes):
            if isinstance(scale, skimage.transform.ProjectiveTransform):
                tform_rescale = scale
            else:
                tform_rescale = build_rescale_transform(scale, channel.shape, target_shape) # also does centering
            #the reason this fucks up is due 
            output.append(fast_warp(channel, tform_rescale + tform_augment, output_shape=target_shapes[0], mode='constant'))#.astype('float32'))
    

    out = np.zeros((target_shapes[0][0], target_shapes[0][1], 3),dtype='float32')
    out[:,:,0] = output[0]
    out[:,:,1] = output[1]
    out[:,:,2] = output[2]

    return out


############################################
### realtime_aug defined methods - end
############################################


class DataLoader(object):
    """shrinks image size to something more manageable
       meant for test or valid data in current state"""
    # TODO - DRY
    def __init__(self, base_path=os.getcwd()):
        self.base_folder = base_path
        self.dims = (100, 100, 3)

    def estimate_zmuv_batch(self, batch):
        # clump RGB values separately
        self.zmuv_means0 = np.mean([image[:,:,0] for image in batch])
        self.zmuv_means1 = np.mean([image[:,:,1] for image in batch])
        self.zmuv_means2 = np.mean([image[:,:,2] for image in batch])
        self.zmuv_std0 = np.std([image[:,:,0]for image in batch])
        self.zmuv_std1 = np.std([image[:,:,1]for image in batch])
        self.zmuv_std2 = np.std([image[:,:,2]for image in batch])

    #i dont think these are workin
    def apply_zumv(self, image):
        if not self.zmuv_means0:
            raise NameError('zmuv_means0 not defined')
        else:
            image[:,:,0] = image[:,:,0]-self.zmuv_means0
            image[:,:,1] = image[:,:,1]-self.zmuv_means1
            image[:,:,2] = image[:,:,2]-self.zmuv_means2

            image[:,:,0] = image[:,:,0]/self.zmuv_std0
            image[:,:,1] = image[:,:,1]/self.zmuv_std1
            image[:,:,2] = image[:,:,2]/self.zmuv_std2
            return image

    def realtime_augmentation(self, img):
        self.default_augmentation_params = {
            'zoom_range': (1 / 1.1, 1.1),
            'rotation_range': (0, 360),
            'shear_range': (0, 0),
            'translation_range': (-4, 4),
            'do_flip': False,
            'allow_stretch': False,
        }
        self.sfs = [1.0]
        self.patch_sizes = [(100,100)]
        self.rng_aug = np.random
        
        #return realtime_aug.perturb_multiscale_new(img, self.sfs , self.default_augmentation_params,
        #             target_shapes=self.patch_sizes, rng=self.rng_aug)
        
        return perturb_multiscale_new(img, self.sfs , self.default_augmentation_params,
                     target_shapes=self.patch_sizes, rng=self.rng_aug)        


    def get_all_reverb_data(self):
        """function scrapes all data from reverb. all of it. condenses the files"""
        try:
            print "trying to loading pickle object"
            self.categories = pickle.load(open("reverbcategories.pkl"))
            print "object loaded"
            self.mapping = {}
            for k,v in enumerate(self.categories.keys()):
                self.mapping[v] = k
        except:
            raise AttributeError(" user has not downloaded reverb data. or in wrong folder ")

    def create_batch_matrix(self, files):
        # files are tuples
        # better way to do this. Has to be.
        
        # image size is  (192, 192, 3)
        # print 'loading pictures from batch' 

        # picture might not be 3D, or 404 might occur. Need to iteratively do this. 

        # because we're not storing the data on a local machine, but rather calling it
        # via GET, there are going to be issues where you might get a 404 or something
        # so the following represents a way to ensure that even if we get a 404 or a 2D
        # picture, we can randomly sample from the other GET data and use that. The reason
        # why we can do that is because the data will be augmented in realtime so that
        # the network won't ever see the same image twice. 

        pictures = []
        labels = []
        bad_images = []
        print "downloading HTTP images"
        for index in range(len(files)):
            try:
                img = skimage.img_as_float(skimage.io.imread(files[index][1])).astype('float32')
                #print img.shape
                _label = files[index][0]
                if img.shape!=(192,192,3):
                    pictures.append("bad image")
                    labels.append("bad image")
                    bad_images.append(index)
                else:
                    pictures.append(img)
                    labels.append(one_hot(_label))                    
            except:
                bad_images.append(index)
                pictures.append("bad image")
                labels.append("bad image")
                pass
        print "HTTP images loaded to memory"
        # if nothing in bad_images, then we can just go straight to the processing
        if bad_images!=[]:
            # define function to randomly find another category to replace the 
            # bad image
            def sampler(n):
                indecies =[]
                correct=False
                for i in range(n):
                    while correct==False:
                        index = random.choice(range(len(pictures)))
                        if pictures[index]!='bad image' or index not in indecies:
                            correct==True
                            print 'Used sampler()'
                            indecies.append(index)
                            break
                        else:
                            correct==False
                return indecies

            # use function
            indecies = sampler(len(bad_images))

            for index in bad_images:
                choices = copy.deepcopy(indecies)
                pictures[index] = pictures[choices[-1]]
                labels[index] = labels[choices[-1]]
                del choices[-1]

        # stack ymatrix so that it contains 1-hot vectors
        ymatrix = np.array(np.vstack(labels), dtype = 'float32')
        # find ZMUV parameters
        self.estimate_zmuv_batch(pictures)
        # augment then apply zmuv filtering
        pictures = [self.apply_zumv( self.realtime_augmentation(img) ) for img in pictures]
        # load all images into 4D tensor
        xmatrix = np.zeros( (len(files),self.dims[0],self.dims[1],self.dims[2]) ).astype('float32')
        for i in range(len(files)):
            xmatrix[i] = pictures[i]
        # delete from memory
        pictures = []
        labels = []
        xmatrix = xmatrix.reshape((len(files), self.dims[2], self.dims[0], self.dims[1]))
        # reshape just to make sure it is known.
        return xmatrix, ymatrix

    def build_unequal_samples_map(self, total_epochs=10000):
        #control the amount of epochs here.
        if self.categories:
            print "Loading HTTP data"
            from copy import deepcopy

            # valid, store as tuple
            self.valid = []
            keys = np.random.choice(self.categories.keys(),size=512, replace=True)
            for key in keys:
                self.valid.append((self.mapping[key], random.choice(self.categories[key])))

            self.test = []
            keys = np.random.choice(self.categories.keys(),size=1000, replace=True)
            for key in keys:
                self.test.append((self.mapping[key], random.choice(self.categories[key])))        

            self.batchmap = {}
            for batch_num in range(0, total_epochs):
                batch = []
                keys = np.random.choice(self.categories.keys(),size=128, replace=True)
                for key in keys:
                    batch.append((self.mapping[key], random.choice(self.categories[key])))
                random.shuffle(batch)
                self.batchmap[batch_num] = batch
            print "HTTP link-data loaded"
        # else:
            # raise attribute error

    def train_data_generator(self):
        self.build_unequal_samples_map()
        for batch in self.batchmap.values():
            yield self.create_batch_matrix(batch)

    def train_create_gen(self):
        gen = self.data_generator(total_epochs)

        def random_gen():
            for x, y in gen:
                yield x, y

        return buffered_gen_threaded(random_gen())

    def __iter__(self):
        self.build_unequal_samples_map()
        for batch in self.batchmap.values():
            try:
                yield self.create_batch_matrix(batch)
                # first time, if fails, works the second time
                # haven't found bug yet. but it works 
                # ValueError: all the input array dimensions except.. 
                # ...for the concatenation axis must match exactly
            except:
                yield self.create_batch_matrix(batch)

    def random_batch(self, test_or_valid='valid'):
        if test_or_valid=='test' or test_or_valid=='valid':

            if test_or_valid=='test':
                value = self.test
            else:
                value = self.valid

            batch = random.sample(value, 128)
            return self.create_batch_matrix(batch)
        else:
            raise NotImplementedError("""parameter: 'test_or_valid' can only be 'test' or 'valid' """)

    def __repr__(self):
        return "DataLoader"


class Net(object):

    """ 'VGG-ish' style net for guitar classification. 
        Using Sander's layers and initialization"""
    
    def __init__(self, classes = 212):
        self.output_dim = classes
        self.batch_size = 128

        #layers using amazon instance with GRID K520
        self.l_in = nn.layers.InputLayer(shape=(self.batch_size, 3, 100, 100))

        #self.C1 = nn.layers.Conv2DLayer(self.l_in, num_filters=64, filter_size=(3, 3), border_mode="same",
        #     W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        #self.C2 = nn.layers.Conv2DLayer(self.C1, num_filters=64, filter_size=(3, 3), border_mode="same",
        #     W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        #self.M1 = nn.layers.pool.MaxPool2DLayer(self.C2, pool_size=3, stride=2)
        ##self.D1 = nn.layers.DropoutLayer(self.M1, p=0.5)

        #self.C3 = nn.layers.Conv2DLayer(self.M1, num_filters=128, filter_size=(3, 3), border_mode="same",
        #     W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        #self.C4 = nn.layers.Conv2DLayer(self.C3, num_filters=128, filter_size=(3, 3), border_mode="same",
        #     W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        #self.M2 = nn.layers.pool.MaxPool2DLayer(self.C4, pool_size=3, stride=2)
        ##self.D2 = nn.layers.DropoutLayer(self.M2, p=0.5)

        # due to mem contrainst, this went from 64,128,256,512,512 to what it is now

        #self.C5 = nn.layers.Conv2DLayer(self.M2, num_filters=256, filter_size=(3, 3), border_mode="same",
        self.C5 = nn.layers.Conv2DLayer(self.l_in, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C6 = nn.layers.Conv2DLayer(self.C5, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C7 = nn.layers.Conv2DLayer(self.C6, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M3 = nn.layers.pool.MaxPool2DLayer(self.C7, pool_size=3, stride=2)

        self.C8 = nn.layers.Conv2DLayer(self.M3, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C9 = nn.layers.Conv2DLayer(self.C8, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C10 = nn.layers.Conv2DLayer(self.C9, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M4 = nn.layers.pool.MaxPool2DLayer(self.C10, pool_size=3, stride=2)

        self.C11 = nn.layers.Conv2DLayer(self.M4, num_filters=64, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C12 = nn.layers.Conv2DLayer(self.C11, num_filters=64, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C13 = nn.layers.Conv2DLayer(self.C12, num_filters=64, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M5 = nn.layers.pool.MaxPool2DLayer(self.C13, pool_size=3, stride=2)


        self.FC1 = nn.layers.DenseLayer(self.M5, num_units=4096, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.D1 = nn.layers.DropoutLayer(self.FC1, p=0.5)
        self.FC2 = nn.layers.DenseLayer(self.D1, num_units=4096, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.D2 = nn.layers.DropoutLayer(self.FC2, p=0.5)
        self.FC3 = nn.layers.DenseLayer(self.D2, num_units=4096, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.D3 = nn.layers.DropoutLayer(self.FC3, p=0.5)


        self.output_layer = nn.layers.DenseLayer(self.D3, num_units=self.output_dim, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)
        self.objective = nn.objectives.Objective(self.output_layer, 
                                    loss_function=nn.objectives.categorical_crossentropy)


        self.loss = self.objective.get_loss()#self.x, target=self.y)
        self.probabilities = self.output_layer.get_output(self.l_in.input_var, deterministic=True)
        self.pred = T.argmax(self.output_layer.get_output(self.l_in.input_var, deterministic=True), axis=1)

        #get weights
        self.all_params = nn.layers.get_all_params(self.output_layer)
        #set init values to None



    def nesterov_trainer(self):
        batch_x = T.ftensor4('batch_x')
        batch_y = T.fmatrix('batch_y')
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                            theano.Param(batch_y),
                                            ],
                               outputs=self.loss,
                               updates=self.updates,
                               givens={self.l_in.input_var: batch_x, self.objective.target_var: batch_y})
        return train_fn


    def predict_(self, given_set):
        batch_x = T.ftensor4('batch_x')
        prediction = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.pred,
                                givens={self.l_in.input_var: batch_x})
        def predict():
            return prediction(given_set)
        return predict


    def predict_proba_(self, given_set):
        batch_x = T.ftensor4('batch_x')
        pred_prob = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.probs,
                                givens={self.l_in.input_var: batch_x})
        def predict_probf():
            return pred_prob(given_set)
        return predict_probf

    #TODO a lot
    def train(self, learning_schedule = {0: 0.015, 500: 0.0015,  800: 0.00015, 1000: 0.000015}, 
                momentum = 0.9, max_epochs=3000, save_every = 20, save_path = os.getcwd()):

        self.save_every = save_every
        self.metadata_tmp_path = save_path+"/model_params.pkl"
        self.best_params_tmp_path = save_path+"/model_params.pkl"
        self.learning_rate_schedule = learning_schedule
        self.learning_rate = theano.shared(np.float32(self.learning_rate_schedule[0]))
        self.momentum = momentum

        #for trainer
        self.updates = nn.updates.nesterov_momentum(self.loss, self.all_params, self.learning_rate, self.momentum)

        train_fn = self.nesterov_trainer() #nesterov with momentum.

        data_iterator = DataLoader()
        data_iterator.get_all_reverb_data()
        data_iterator.build_unequal_samples_map()
        best_dev_loss = numpy.inf


        #for loading the data onto the gpu
        #create_train_gen = lambda: train_set_iterator.create_gen(max_epochs)

        patience = 1000  
        patience_increase = 2.
        improvement_threshold = 0.995
        done_looping = False
        print '... training the model'
        start_time = time.clock()
        epoch = 0
        timer = None

        #for plotting
        self._costs = []
        self._train_errors = []
        self._dev_errors = []

        while (epoch < max_epochs) and (not done_looping):
            losses_train = []
            losses = []
            avg_costs = []
            timer = time.time()
            for iteration, (x, y) in enumerate(data_iterator):

                if iteration in self.learning_rate_schedule:
                    lr = np.float32(self.learning_rate_schedule[iteration])
                    print "  setting learning rate to %.7f" % lr
                    self.learning_rate.set_value(lr)


                print "  load training data onto GPU"
                avg_cost = train_fn(x, y)
                if np.isnan(avg_cost):
                    raise RuntimeError("NaN DETECTED.")
                
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            
                #for saving the batch
                if ((iteration + 1) % save_every) == 0:
                    print
                    print "  Saving metadata, parameters"

                    with open(self.metadata_tmp_path, 'w') as f:
                        pickle.dump({'losses_train': avg_costs,'param_values': nn.layers.get_all_param_values(self.output_layer)},
                                     f, pickle.HIGHEST_PROTOCOL)

                mean_train_loss = numpy.mean(avg_costs)
                #print "  mean training loss:\t\t%.6f" % mean_train_loss
                #losses_train.append(mean_train_loss)

                #accuracy assessment
                output = one_hot(self.predict_(x)(),m=212)
                train_loss = log_loss(output, y)
                acc = 1 - accuracy(output, y)
                losses.append(train_loss)
                del output
                del x
                del y

                print('  epoch %i took %f seconds' %
                    (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                    (epoch, mean_train_loss))
                print('  epoch %i, training error %f' %
                    (epoch, acc))

                #for plotting
                self._costs.append(mean_train_loss)
                self._train_errors.append(acc)
                
                #valid accuracy
                xd,yd = data_iterator.random_batch('valid')

                valid_output = one_hot(self.predict_(xd)(),m=212)
                valid_acc = 1 - accuracy(valid_output, yd)
                self._dev_errors.append(valid_acc)
                del valid_output
                del xd
                del yd

                if valid_acc < best_dev_loss:
                    best_dev_loss = valid_acc
                    best_params = copy.deepcopy(self.all_params )
                    print('!!!  epoch %i, validation error of best model %f' %
                        (epoch, valid_acc))
                    print
                    print "  Saving best performance parameters"
                    with open(self.best_params_tmp_path, 'w') as f:
                        pickle.dump({'losses_train': avg_costs,'param_values': nn.layers.get_all_param_values(self.output_layer)},
                                     f, pickle.HIGHEST_PROTOCOL)
                    if (valid_acc < best_dev_loss *
                        improvement_threshold):
                        patience = max(patience, iteration * patience_increase)
                    if patience <= iteration:
                        done_looping = True
                        break
                epoch += 1

if __name__ == '__main__':
    net = Net()
    net.train()
    with open('saved_net.pkl', 'w') as f:
        pickle.dump(net,f,pickle.HIGHEST_PROTOCOL)





