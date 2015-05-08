import multiprocessing as mp
import Queue
import threading
Conv2DLayer = tmp_dnn.Conv2DDNNLayer
MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer
#real-time aug based on Sander Dielman's KDSB solution



maps = {'breedlove': 9,
 'dean': 11,
 'epiphone': 1,
 'esp': 5,
 'fender': 6,
 'g&l': 14,
 'gibson': 18,
 'gretsch': 13,
 'guild': 4,
 'ibanez': 12,
 'jackson': 17,
 'martin': 15,
 'music man': 10,
 'paul reed smith': 8,
 'rickenbacker': 2,
 'schecter': 20,
 'squier': 16,
 'taylor': 3,
 'washburn': 19,
 'yamaha': 7}



#sander realtime gen
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



class ZMUV(object):
    #class to get zero mean unit inv for 
    def __init__(self):
        self.base_folder = os.getcwd()

    def estimate_params(self):
        files = random.sample(self.files,15000)
        xmatrix = np.vstack([np.asarray(Image.open(self.base_folder+'/train/'+name+'/'+name+'.png')) for name in files])
        xmatrix = xmatrix.reshape((15000, 3, 640, 640)) 

        self.zmuv_mean_0 = np.mean([x[0] for x in xmatrix])
        self.zmuv_mean_1 = np.mean([x[1] for x in xmatrix])
        self.zmuv_mean_2 = np.mean([x[2] for x in xmatrix])
        self.zmuv_stds_0 = np.std([x[0] for x in xmatrix])
        self.zmuv_stds_1 = np.std([x[1] for x in xmatrix])
        self.zmuv_stds_2 = np.std([x[2] for x in xmatrix])
        del(xmatrix)


class TestData(object):
    def __init__(self, path, zmuv, maps=maps):
        self.zmuv = zmuv
        self.base_folder = path
        self.files=os.listdir(self.base_folder)[1:]
        self.mapping = maps

    def create_test_matrix(self, files):
        #better way to do this. Has to be.
        xmatrix = np.vstack([np.asarray(Image.open(self.base_folder+'/train/'+name+'/'+name+'.png')) for name in self.files])
        ymatrix = np.hstack([self.mapping[open(self.base_folder+'/'+name+'/label.txt').read()] for name in sef.files])
        xmatrix = xmatrix.reshape((len(self.files), 3, 640, 640)) 
        ymatrix = np.array(ymatrix,dtype = 'int32')
        #TODO: add rotations and zooming.

        #apply ZMUV to batch
        for image in xmatrix:
            image[0] -= self.zmuv.zmuv_mean_0
            image[1] -= self.zmuv.zmuv_mean_1
            image[2] -= self.zmuv.zmuv_mean_2
            image[0] /= self.zmuv.zmuv_stds_0
            image[1] /= self.zmuv.zmuv_stds_1
            image[2] /= self.zmuv.zmuv_stds_2
        return xmatrix,ymatrix


class TrainDatasetMiniBatchIterator(object):
    """  batch iterator """
    def __init__(self, path, total_epochs, zmuv, 
                        train_or_valid='train', 
                        mapping = maps, batch_size=128):
        self.zmuv = zmuv
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.mapping = mapping
        self.base_folder = path+'/'+train_test_valid
        self.files=os.listdir(self.base_folder)[1:]
        self.n_samples = len(self.files)
        self.val = (self.n_samples + self.batch_size - 1) / self.batch_size
        self.batches = {}
        count = 1
        num = 1
        batch_values = []
        for i in self.files[1:]:
            if count > self.val:
                self.batches[num] = batch_values
                count = 1
                num += 1
                batch_values = []
            count += 1
            batch_values.append(i)


    def create_matrix(self, files):
        #add test time augmentation
        xmatrix = np.vstack([np.asarray(Image.open(self.base_folder+'/'+name+'/'+name+'.png')) for name in files])
        ymatrix = np.hstack([self.mapping[open(self.base_folder+'/'+name+'/label.txt').read()] for name in files])
        xmatrix = xmatrix.reshape((self.batch_size, 3, 640, 640)) 
        ymatrix = np.array(ymatrix,dtype = 'int32')
        #TODO: add rotations and zooming.

        #apply ZMUV to batch
        for image in xmatrix:
            image[0] -= self.zmuv.zmuv_mean_0
            image[1] -= self.zmuv.zmuv_mean_1
            image[2] -= self.zmuv.zmuv_mean_2
            image[0] /= self.zmuv.zmuv_stds_0
            image[1] /= self.zmuv.zmuv_stds_1
            image[2] /= self.zmuv.zmuv_stds_2
        return xmatrix,ymatrix

    def __iter__(self):
        for i in xrange(1,self.val):
            yield self.create_matrix(self.batches[i]) 



class TrainDatasetMiniBatchIterator(object):
    """  batch iterator """
    def __init__(self, path, total_epochs, zmuv, 
                        train_or_valid='train', 
                        mapping = maps, batch_size=128):
        self.zmuv = zmuv
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.mapping = mapping
        self.base_folder = path+'/'+train_test_valid
        self.files=os.listdir(self.base_folder)[1:]
        self.n_samples = len(self.files)
        self.val = (self.n_samples + self.batch_size - 1) / self.batch_size
        self.batches = {}
        count = 1
        num = 1
        batch_values = []
        for i in self.files[1:]:
            if count > self.val:
                self.batches[num] = batch_values
                count = 1
                num += 1
                batch_values = []
            count += 1
            batch_values.append(i)

    def create_matrix(self, files):
        #add test time augmentation
        xmatrix = np.vstack([np.asarray(Image.open(self.base_folder+'/'+name+'/'+name+'.png')) for name in files])
        ymatrix = np.hstack([self.mapping[open(self.base_folder+'/'+name+'/label.txt').read()] for name in files])
        xmatrix = xmatrix.reshape((self.batch_size, 3, 640, 640)) 
        ymatrix = np.array(ymatrix,dtype = 'int32')
        #TODO: add rotations and zooming.

        #apply ZMUV to batch
        for image in xmatrix:
            image[0] -= self.zmuv.zmuv_mean_0
            image[1] -= self.zmuv.zmuv_mean_1
            image[2] -= self.zmuv.zmuv_mean_2
            image[0] /= self.zmuv.zmuv_stds_0
            image[1] /= self.zmuv.zmuv_stds_1
            image[2] /= self.zmuv.zmuv_stds_2
        yield xmatrix,ymatrix

    def data_generator(self):
        for i in xrange(1,self.val):
            yield self.create_matrix(self.batches[i]) 

    def create_gen(self):
        gen = data_generator()

        def random_gen():
            for x, y in gen:
                yield x, y

        return buffered_gen_threaded(random_gen())



class Net(object):

    """ VGG style net for guitar classification. Using Sander's layers and initialization"""
    
    def __init__(self, ndim = 3,...):
        self.output_dim = out


        NUM_EPOCHS = 5000
        BATCH_SIZE = 600
        LEARNING_RATE = 0.01
        MOMENTUM = 0.9
        self.batch_size = 128
        self.x = T.fmatrix('x')
        self.y = T.ivector('y')


        self.l_in = lasagne.layers.InputLayer(shape=(self.batch_size, 3, 640, 640))        
        self.C1 = Conv2DLayer(self.l_in, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C2 = Conv2DLayer(self.C1, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M1 = MaxPool2DLayer(self.C3, ds=(3, 3), strides=(2, 2))
        self.D1 = lasagne.layers.DropoutLayer(self.M1, p=0.5)
        self.C3 = Conv2DLayer(self.D1, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C4 = Conv2DLayer(self.C3, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M2 = MaxPool2DLayer(self.C4, ds=(3, 3), strides=(2, 2))
        self.D2 = lasagne.layers.DropoutLayer(self.M2, p=0.5)
        self.C5 = Conv2DLayer(self.D2, num_filters=32, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.C6 = Conv2DLayer(self.C5, num_filters=16, filter_size=(3, 3), border_mode="same",
             W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.M3 = MaxPool2DLayer(self.C6, ds=(3, 3), strides=(2, 2))
        self.D3 = lasagne.layers.DropoutLayer(self.M3, p=0.5)
        self.FC1 = nn.layers.DenseLayer(self.D3, num_units=256, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        self.D4 = lasagne.layers.DropoutLayer(self.FC1, p=0.5)
        self.FC2 = nn.layers.DenseLayer(self.D4, num_units=256, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
        
        self.O = nn.layers.DenseLayer(self.FC2, num_units=self.output_dim, W=nn_plankton.Orthogonal(1.0),
                             b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)
       

        self.objective = lasagne.objectives.Objective(self.O, loss_function=lasagne.objectives.categorical_crossentropy)
        self.mean_cost = self.objective.get_loss(self.x, target=self.y)
        self.probs = self.objective.get_output(self.x, deterministic=True)
        self.pred = T.argmax(self.objective.get_output(self.x, deterministic=True), axis=1)
        self.errors  = T.mean(T.eq(self.pred, self.y), dtype=theano.config.floatX)


    def get_trainer(self):
        batch_x = T.matrix('batch_x')
        batch_y = T.vector('batch_y')
        loss_train = objective.get_loss(batch_x, target=batch_y)
        self.mean_cost = 
        pred = T.argmax(self.objective.get_output(batch_x, deterministic=True), axis=1)
        accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)
        all_params = lasagne.layers.get_all_params(output_layer)
        updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                       theano.Param(batch_y)],
                               outputs=self.mean_cost,
                               updates=updates,
                               givens={self.x: batch_x, self.y: batch_y})
        return train_fn


    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y})

        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


    def predict_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.ypred,
                                givens={self.x: batch_x})
        def predict():
            return pred(given_set)
        return predict


    def predict_proba_(self, given_set):
        batch_x = T.fmatrix('batch_x')
        pred_prob = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.probs,
                                givens={self.x: batch_x})
        def predict_probf():
            return pred_prob(given_set)
        return predict_probf

    #needs a bit of work here
    def fit(max_epochs=300, early_stopping=True, split_ratio=0.1,
            verbose=False, plot=False):

        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        if x_dev == None or y_dev == None:
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                test_size=split_ratio, random_state=42)

        #training function is already adadelta
        train_fn = self.get_trainer()
        zmuv = ZMUV()
        print 'Zero Mean Unit variance'
        zmuv.estimate_params()


        train_set_iterator = TrainDatasetMiniBatchIterator(os.getcwd(), total_epochs ,zmuv=zmuv, train_test_or_valid='train')
        dev_set_iterator = TrainDatasetMiniBatchIterator(os.getcwd(), total_epochs, zmuv=zmuv ,train_test_or_valid='valid')
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf


        #create_train_gen = lambda: self.data_loader.create_random_gen(self.data_loader.images_train, self.data_loader.labels_train)



        patience = 1000  
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant

        done_looping = False
        print '... training the model'
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        timer = None

        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []

        self.stopcost = []
        self.stopval= []

        while (epoch < max_epochs) and (not done_looping):
            zeros = []
            trainzeros= []
            epoch += 1
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for iteration, (x, y) in enumerate(train_set_iterator):
                #call the training function
                avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())

                print('  epoch %i took %f seconds' %
                    (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                    (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                    (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)

            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)

            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                        (epoch, dev_errors))
                if (dev_errors < best_dev_loss *
                    improvement_threshold):
                    patience = max(patience, iteration * patience_increase)
                if patience <= iteration:
                    done_looping = True
                    break
                  
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param





