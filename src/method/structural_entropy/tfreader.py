import tensorflow as tf


def _parse_tfrecord_function(example):
    example_fmt = {
            'structural_entropy': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
    parsed = tf.io.parse_single_example(example, example_fmt)
    decoded_entropy = tf.io.decode_raw(parsed['structural_entropy'], tf.float32)

    return decoded_entropy, parsed['label']


def make_dataset(filepath, SHUFFLE_BUFFER_SIZE=1024, BATCH_SIZE=32, EPOCHS=5):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.map(lambda x: _parse_tfrecord_function(x))
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset
