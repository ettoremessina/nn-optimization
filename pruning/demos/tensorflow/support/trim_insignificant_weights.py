import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

import os
import tempfile
import zipfile

class AttemptInfo:
    def __init__(self,
        name,
        tot_num_of_weights, tot_num_of_nonzero_weights, tot_num_of_zero_weights,
        unzippedh5_size, zippedh5_size,
        unzippedlt_size, zippedlt_size,
        y_pred, error_value):
        self.name = name
        self.tot_num_of_weights = tot_num_of_weights
        self.tot_num_of_nonzero_weights = tot_num_of_nonzero_weights
        self.tot_num_of_zero_weights = tot_num_of_zero_weights
        self.unzippedh5_size = unzippedh5_size
        self.zippedh5_size = zippedh5_size
        self.unzippedlt_size = unzippedlt_size
        self.zippedlt_size = zippedlt_size
        self.y_pred = y_pred
        self.error_value = error_value

        self.compressionh5_factor = 0.
        if self.unzippedh5_size != 0:
            self.compressionh5_factor = (100. * (self.unzippedh5_size - self.zippedh5_size)) / self.unzippedh5_size

        self.compressionlt_factor = 0.
        if self.unzippedlt_size != 0:
            self.compressionlt_factor = (100. * (self.unzippedlt_size - self.zippedlt_size)) / self.unzippedlt_size

    def print(self):
        print('Model:', self.name)
        print('  Total number of weights:', self.tot_num_of_weights)
        print('  Total number of non-zero weights:', self.tot_num_of_nonzero_weights)
        print('  Total number of zero weights:', self.tot_num_of_zero_weights)
        print('  Unzipped h5 size: %i bytes' % self.unzippedh5_size)
        print('  Zipped h5 size: %i bytes (compression factor: %.2f%%)' % (self.zippedh5_size, self.compressionh5_factor))
        print('  Unzipped tflite size: %i bytes' % self.unzippedlt_size)
        print('  Zipped tflite size: %i bytes (compression factor: %.2f%%)' % (self.zippedlt_size, self.compressionlt_factor))
        print('  Error (loss) value: %E' % self.error_value)

class AttemptConfig:
    def __init__(self, name, pruning_schedule):
        self.name = name
        self.pruning_schedule = pruning_schedule

def inspect_weigths(name, model, verbose=False):
    nlayer=0
    tot_num_of_weights=0
    tot_num_of_nonzero_weights=0
    if verbose:
        print('Model:', name)
        print('  Layers:')

    for layer in model.layers:
        nlayer += 1
        weight_sets = layer.get_weights()
        if verbose:
            print('    Layer #%i, num of sets: %i'% (nlayer, len(weight_sets)))
        for i in range(0, len(weight_sets)):
            num_of_weights_of_set = np.prod(weight_sets[i].shape)
            tot_num_of_weights += num_of_weights_of_set
            num_of_nonzero_weights_of_set = np.count_nonzero(weight_sets[i])
            tot_num_of_nonzero_weights += num_of_nonzero_weights_of_set
            if verbose:
                print('      Num of weights: %i, of which non-zero: %i and zero: %i'% \
                    (num_of_weights_of_set, num_of_nonzero_weights_of_set, num_of_weights_of_set - num_of_nonzero_weights_of_set))

    tot_num_of_zero_weights = tot_num_of_weights - tot_num_of_nonzero_weights
    if verbose:
        print('  Total number of weights: %i, of which non-zero: %i and zero: %i'%
            (tot_num_of_weights, tot_num_of_nonzero_weights, tot_num_of_zero_weights))
    return tot_num_of_weights, tot_num_of_nonzero_weights, tot_num_of_zero_weights

def retrieve_size_of_model(model):
    _, mfn = tempfile.mkstemp(suffix=".h5")
    _, zfn = tempfile.mkstemp(suffix=".zip")
    try:
        model.save(mfn, include_optimizer=False)
        with zipfile.ZipFile(zfn, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.write(mfn)
        unzipped_size = os.path.getsize(mfn)
        zipped_size = os.path.getsize(zfn)
    finally:
        os.remove(mfn)
        os.remove(zfn)
    return unzipped_size, zipped_size

def retrieve_size_of_lite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    _, lfn = tempfile.mkstemp(suffix=".tflite")
    _, zfn = tempfile.mkstemp(suffix=".zip")
    try:
        with open(lfn, 'wb') as f:
            f.write(tflite_model)
        with zipfile.ZipFile(zfn, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.write(lfn)
        unzipped_size = os.path.getsize(lfn)
        zipped_size = os.path.getsize(zfn)
    finally:
        os.remove(lfn)
        os.remove(zfn)
    return unzipped_size, zipped_size

def build_pruning_model(model_org, pruning_attempt):
    pruning_params = {'pruning_schedule': pruning_attempt }
    pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model_org, **pruning_params)
    return pruning_model

def retrieve_callbacks_for_pruning():
    return [ tfmot.sparsity.keras.UpdatePruningStep() ]

def extract_pruned_model(model_pruning):
    return tfmot.sparsity.keras.strip_pruning(model_pruning)

def print_attempt_infos(attempt_infos):
    print('%-20s %12s %9s %16s'% ('Attempt name', 'Size h5', '(Comp. %)', 'Error (loss)'))
    for ai in attempt_infos:
        print('%-20s %12i (%6.2f%%) %16e'% (ai.name, ai.zippedh5_size, ai.compressionh5_factor, ai.error_value))
