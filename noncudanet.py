import multiprocessing as mp
import Queue
import threading
import lasagne as nn
import nn_plankton
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
import tmp_dnn
from collections import defaultdict
import random
from PIL import Image
import numpy
import theano
import utils
import copy
# define params and other errata
maps = {'breedlove': 8,
 'dean': 10,
 'epiphone': 0,
 'esp': 4,
 'fender': 5,
 'g&l': 13,
 'gibson': 17,
 'gretsch': 12,
 'guild': 3,
 'ibanez': 11,
 'jackson': 16,
 'martin': 14,
 'music man': 9,
 'paul reed smith': 7,
 'rickenbacker': 1,
 'schecter': 19,
 'squier': 15,
 'taylor': 2,
 'washburn': 18,
 'yamaha': 6}

############################################
### Sander defined methods - Start
############################################

Conv2DLayer = tmp_dnn.Conv2DDNNLayer
MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer
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

def one_hot(vec, m=20):
    if m is None:
        m = int(np.max(vec)) + 1
    return np.eye(m)[vec]

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



class DataLoader(object):
    """shrinks image size to something more manageable
       meant for test or valid data in current state"""
    # TODO - DRY
    def __init__(self, base_path=os.getcwd(), mapping = maps, train_test_valid='train'):
        self.base_folder = base_path
        self.dims = (100, 100, 3)
        self.test_train_valid = train_test_valid
        try:
            self.files=os.listdir(self.base_folder+'/data/'+self.test_train_valid)
            self.files = [x for x in self.files if x!='.DS_Store']
        except:
            raise OSError("You're not in the homefolder, big dog.")
            os.chdir(self.base_folder)
        self.mapping = mapping

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

    def create_batch_matrix(self, files):
        # better way to do this. Has to be.
        pictures = [Image.open(self.base_folder+
                    '/data/'+self.test_train_valid+'/'+name+'/'+name+'.png') for name in files]
        #downscales the image so that the total size is (300,300,3).
        for img in pictures:
            img.thumbnail((self.dims[0],self.dims[1]), Image.ANTIALIAS)

        # 255 - image  produces strange issues with actual image, and thumbnail is inverting the image. 
        # 256 is used instead. 
        # TODO change eventually
        pictures = [256 - np.asarray(img).astype('float32') for img in pictures]
        self.estimate_zmuv_batch(pictures)
        # changing to the current way, vstack+reshape was causing strange issues
        pictures = [self.apply_zumv(img) for img in pictures]
        # float32 adds time to creation
        xmatrix = np.zeros( (len(files),self.dims[0],self.dims[1],self.dims[2]) ).astype('float32')
        for i in range(len(files)):
            xmatrix[i] = pictures[i]
        del(pictures) #for mem

        #TODO one_hot all ytrain
        ymatrix = np.vstack([one_hot(self.mapping[open(self.base_folder+
                '/data/'+self.test_train_valid+'/'+name+'/label.txt').read()])  for name in files])


        xmatrix = xmatrix.reshape((len(files), self.dims[2], self.dims[0], self.dims[1]))#.astype('float32')
        ymatrix = np.array(ymatrix,dtype = 'float32')
        return xmatrix, ymatrix

    def build_unequal_samples_map(self, total_epochs=5000): # , batch_size = 128
        # need this method because there's something like 7000 fenders, and 
        # next highest is 1000, then 400, then drops. 
        from copy import deepcopy
        #build data maps
        self.sample_map = defaultdict( list )
        data_folder = self.base_folder+'/data/'+self.test_train_valid
        os.chdir(data_folder)
        for f in self.files:
            os.chdir(data_folder)
            os.chdir(f)
            with open('label.txt') as labelf:
                label = labelf.read()
            self.sample_map[label].append(f)
        # sample from maps:
        # 20 classes. 128-batch size. 6-7 classes per sample
        # sample that pulls from other samples
        self.samples = deepcopy(self.sample_map)
        # randomly sample without replacement until nothing 
        # is left, then repeat until you run out of epochs.
        self.batchmap = {}
        for batch_num in range(0, total_epochs):
            batch = []
            nums = []
            # this works because num_gen has 20 nums in it. 
            num_gen = list( chain(*[list(repeat(7,8)),list(repeat(6,12))]))
            random.shuffle(num_gen)
            # (8*7)+(12*6.)=128
            for class_ in self.samples.keys():
                num = num_gen[-1]
                if len(self.samples[class_]) >= num:
                    samples_ = self.samples[class_][-num:]#
                    del self.samples[class_][-num:]
                    batch.append(samples_)
                    del num_gen[-1]
                else:
                    self.samples[class_] = deepcopy(self.sample_map[class_])#
                    random.shuffle(self.samples[class_])
                    samples_ = self.samples[class_][-num:]
                    del self.samples[class_][-num:]
                    batch.append(samples_)
                    del num_gen[-1]
            shufflebatch = list(chain(*batch))
            random.shuffle(shufflebatch)
            self.batchmap[batch_num] = shufflebatch
        os.chdir(self.base_folder)

    def data_generator(self,total_epochs=5000):
        if self.test_train_valid!='train':
            raise NotImplementedError("function not implemented this data type")
        self.build_unequal_samples_map(total_epochs)
        for batch in self.batchmap.values():
            yield self.create_batch_matrix(batch)

    def create_gen(self, total_epochs=5000):
        if self.test_train_valid!='train':
            raise NotImplementedError("function not implemented this data type")
        gen = self.data_generator(total_epochs)

        def random_gen():
            for x, y in gen:
                yield x, y

        return buffered_gen_threaded(random_gen())

    def __iter__(self):
        if self.test_train_valid!='train':
            raise NotImplementedError("function not implemented this data type")
        self.build_unequal_samples_map()
        for batch in self.batchmap.values():
            yield self.create_batch_matrix(batch)

    def random_batch(self):
        if not self.batchmap:
            raise NotImplementedError("need to declare the batchmap first")
        batch = random.choice(self.batchmap.values())
        return self.create_batch_matrix(batch)




class Net(object):

    """ 'VGG-ish' style net for guitar classification. 
        Using Sander's layers and initialization"""
    
    def __init__(self, classes = 20):
        self.output_dim = classes
        self.batch_size = 128

        #layers using amazon instance with GRID K520
        self.l_in = nn.layers.InputLayer(shape=(self.batch_size, 3, 100, 100))

        self.C1 = nn.layers.Conv2DLayer(self.l_in, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C2 = nn.layers.Conv2DLayer(self.C1, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M1 = nn.layers.pool.MaxPool2DLayer(self.C2, pool_size=3, stride=2)
        #self.D1 = nn.layers.DropoutLayer(self.M1, p=0.5)

        self.C3 = nn.layers.Conv2DLayer(self.M1, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C4 = nn.layers.Conv2DLayer(self.C3, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M2 = nn.layers.pool.MaxPool2DLayer(self.C4, pool_size=3, stride=2)
        #self.D2 = nn.layers.DropoutLayer(self.M2, p=0.5)

        self.C5 = nn.layers.Conv2DLayer(self.M2, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C6 = nn.layers.Conv2DLayer(self.C5, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M3 = nn.layers.pool.MaxPool2DLayer(self.C6, pool_size=3, stride=2)
        #self.D3 = nn.layers.DropoutLayer(self.M3, p=0.5)

        self.FC1 = nn.layers.DenseLayer(self.M3, num_units=256, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.D4 = nn.layers.DropoutLayer(self.FC1, p=0.5)
        self.FC2 = nn.layers.DenseLayer(self.D4, num_units=256, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.output_layer = nn.layers.DenseLayer(self.FC2, num_units=self.output_dim, W=nn_plankton.Orthogonal(1.0),
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
    def train(self, learning_schedule = {0: 0.0015, 700: 0.00015,  800: 0.000015}, 
                momentum = 0.9, max_epochs=3000, save_every = 20, save_path = os.getcwd()):

        self.save_every = save_every
        self.metadata_tmp_path = save_path+"/model_params.pkl"
        self.learning_rate_schedule = learning_schedule
        self.learning_rate = theano.shared(np.float32(self.learning_rate_schedule[0]))
        self.momentum = momentum

        #for trainer
        self.updates = nn.updates.nesterov_momentum(self.loss, self.all_params, self.learning_rate, self.momentum)


        train_fn = self.nesterov_trainer() #nesterov with momentum.
        train_set_iterator = DataLoader(os.getcwd(),train_test_valid='train')
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
            for iteration, (x, y) in enumerate(train_set_iterator):

                if iteration in learning_rate_schedule:
                    lr = np.float32(learning_rate_schedule[iteration])
                    print "  setting learning rate to %.7f" % lr
                    learning_rate.set_value(lr)


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
                    print "Saving metadata, parameters"

                    with open(metadata_tmp_path, 'w') as f:
                        pickle.dump({'losses_train': avg_costs,'param_values': nn.layers.get_all_param_values(self.output_layer)},
                                     f, pickle.HIGHEST_PROTOCOL)

                mean_train_loss = numpy.mean(avg_costs)
                #print "  mean training loss:\t\t%.6f" % mean_train_loss
                #losses_train.append(mean_train_loss)

                #accuracy assessment
                output = utils.one_hot(predict_(x)())
                train_loss = utils.log_loss(output, y)
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
                

                #dev_errors = numpy.mean(dev_scoref())
                #valid accuracy
                dev_set_iterator = DataLoader(os.getcwd(), train_test_valid='valid')
                #too many open files
                xd,yd = dev_set_iterator.create_batch_matrix(random.sample(dev_set_iterator.files,128))
                
                valid_output = utils.one_hot(predict_(xd)())
                valid_acc = 1 - utils.accuracy(valid_test, yd)
                self._dev_errors.append(valid_acc)
                del x
                del y 

                if valid_acc < best_dev_loss:
                    best_dev_loss = valid_acc
                    best_params = copy.deepcopy(all_params)
                    print('!!!  epoch %i, validation error of best model %f' %
                        (epoch, valid_acc))
                    print
                    print "Saving best performance parameters"
                    with open(metadata_tmp_path, 'w') as f:
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




