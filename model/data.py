import os
import glob
import random
import csv
import itertools
import numpy as np
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)


def file_name(abs_path):
    return os.path.basename(abs_path).replace('.csv', '')


class Dataset(object):
    def __init__(self, path):
        self.path = path
        files = glob.glob(path + '/*.csv')
        self.collections = {file_name(file): file for file in files}

    def rows(self, collection_name, num_epochs=None):
        if collection_name not in self.collections:
            raise ValueError(
                'Collection not found: {}'.format(collection_name)
            )
        epoch = 0
        while True:
            with open(self.collections[collection_name], 'r', newline='') as f:
                r = csv.reader(f)
                for row in r:
                    yield row
            epoch += 1
            if num_epochs and (epoch >= num_epochs):
                raise StopIteration

    def _batch_iter(self, collection_name, batch_size, num_epochs):
        gen = [self.rows(collection_name, num_epochs)] * batch_size
        return itertools.zip_longest(fillvalue=None, *gen)

    def batches(self, collection_name, batch_size, num_epochs=None):
        for batch in self._batch_iter(collection_name, batch_size, num_epochs):
            yield (
                np.array([int(row[0]) for row in batch if row]),
                np.array([[float(x) for x in row[1].split()]
                          for row in batch if row])
            )
