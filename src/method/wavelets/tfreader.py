import tensorflow as tf


def _parse_tfrecord_function(example):
    example_fmt = {
        'cA': tf.io.FixedLenFeature([], tf.string),
        'cD': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        }
    parsed = tf.io.parse_single_example(example, example_fmt)
    cA = tf.io.decode_raw(parsed['cA'], tf.float32)
    cD = tf.io.decode_raw(parsed['cD'], tf.float32)

    return [cA, cD], parsed['label']


def make_dataset(filepath, SHUFFLE_BUFFER_SIZE=1024, BATCH_SIZE=32, EPOCHS=5):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.map(lambda x: _parse_tfrecord_function(x))
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset
