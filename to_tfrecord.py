import pandas as pd
import tensorflow as tf
import sys

#Creating fake data for demonstration
X_train = pd.DataFrame({'feat1':[1,2,3], 
                  'feat2':['one','two','three']})
training_y = pd.DataFrame({'target': [3.4, 11.67, 44444.1]})

X_train.to_csv('X_train.csv')
training_y.to_csv('training_y.csv')

#TFRecords boilerplate
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(index, feat1, feat2, target):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'index': _int64_feature(index),
      'feat1': _int64_feature(feat1),
      'feat2': _bytes_feature(feat2),
      'target': _float_feature(target)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

#Loading the data into chunks of size 2.  Change this to 1e5 in your code
CHUNKSIZE = 1e5
train = pd.read_csv('X_train.csv', chunksize=CHUNKSIZE)
y = pd.read_csv('training_y.csv', chunksize=CHUNKSIZE)

file_num = 0
while 1:
    try:
        print(f'{file_num}')
        #Getting the data from the two files 
        df = pd.concat([train.get_chunk(), y.get_chunk()],1)
        
        #Writing the TFRecord
        with tf.io.TFRecordWriter(f'Record_{file_num}.tfrec') as writer:
            for k in range(df.shape[0]):
                row = df.iloc[k,:]
                example = serialize_example(
                    df.index[k],
                    row['feat1'],
                    str.encode(row['feat2']), #Note the str.encode to make tf play nice with strings
                    row['target']) 
                writer.write(example)    
        file_num += 1
    except:
        print(f'ERROR: {sys.exc_info()[0]}')
        break
