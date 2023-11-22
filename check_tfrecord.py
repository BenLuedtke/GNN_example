import tensorflow as tf
import pandas as pd

#Reading the TFRecord
def read_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "index": tf.io.FixedLenFeature([], tf.int64), 
        "feat1": tf.io.FixedLenFeature([], tf.int64),
        "feat2": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.float32)
    }
    
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    index = example['index']
    feat1 = example['feat1']
    feat2 = example['feat2']
    target = example['target']
    return index, feat1, feat2, target 

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

AUTO = tf.data.experimental.AUTOTUNE
def get_training_dataset(filenames, batch_size=2):
    dataset = load_dataset(filenames, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

training_dataset = get_training_dataset(filenames= ['Record_0.tfrec', 'Record_1.tfrec'])
#training_dataset = training_dataset.unbatch().batch(20)
next(iter(training_dataset))

