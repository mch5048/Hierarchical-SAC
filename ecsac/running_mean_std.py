import numpy as np
import pickle
from mpi4py import MPI
import tf_util
import tensorflow as tf

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # mean-std is not always the 2d text!
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')         
        self.count = epsilon
        self.std = np.sqrt(self.var)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(self.var)
        self.count = new_count

    def save_mean_std(self, path):
        _path = path
        mean_std_arr = np.array([self.mean, self.std]) # shape (2,), indexing by [0] and [1]

        with open ( _path, 'wb') as f:             
            pickle.dump(mean_std_arr, f)

    def load_mean_std(self, path):
        # loads mean and stddev derived in training session
        # only used for test session
        _path = path
        with open(_path, 'rb') as f:
            _mean_std_arr = pickle.load(f)
        self.mean = _mean_std_arr[0] 
        self.std = _mean_std_arr[1] 




class RunningMeanStdMPI(object):
    def __init__(self, epsilon=1e-2, shape=(), name=None):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        with tf.variable_scope(name):
            self._sum = tf.get_variable(
                dtype=tf.float64,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="runningsum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float64,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="runningsumsq", trainable=FCalse)
            self._count = tf.get_variable(
                dtype=tf.float64,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

            newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
            newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
            newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
            self.incfiltparams = tf_util.function([newsum, newsumsq, newcount], [],
                                                  updates=[tf.assign_add(self._sum, newsum),
                                                           tf.assign_add(self._sumsq, newsumsq),
                                                           tf.assign_add(self._count, newcount)])

    def update(self, data):
        """
        update the running mean and std

        :param data: (np.ndarray) the data
        """
        data = data.astype('float64')
        data_size = int(np.prod(self.shape))
        totalvec = np.zeros(data_size * 2 + 1, 'float64')
        addvec = np.concatenate([data.sum(axis=0).ravel(), np.square(data).sum(axis=0).ravel(),
                                 np.array([len(data)], dtype='float64')])


        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0: data_size].reshape(self.shape),
                           totalvec[data_size: 2 * data_size].reshape(self.shape), totalvec[2 * data_size])

