import json
import tensorflow as tf
import numpy as np
import math

def initialize_TFRecords(tfrecords_filepath, num_tfrecords=10, filename="training"):
    training_writers = []
    for i in range(num_tfrecords):
        training_writers.append(tf.io.TFRecordWriter(tfrecords_filepath + "{}{}.tfrecords".format(filename,i)))
    return training_writers


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_structural_entropy_example(structural_entropy, label):
    feature={
        'structural_entropy': _bytes_feature(structural_entropy.tostring()),
        'label': _int64_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_wavelet_example(cA, cD, label):
    feature={
        'cA': _bytes_feature(cA.tostring()),
        'cD': _bytes_feature(cD.tostring()),
        'label': _int64_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def load_parameters(parameters_path):
    """
    It loads the network parameters

    Parameters
    ----------
    parameters_path: str
        File containing the parameters of the network
    """
    with open(parameters_path, "r") as param_file:
        params = json.load(param_file)
    return params

def extract_structural_entropy(hex_sequence, chunk_size=256, log=2):
    """
    Split the file into non-overlapping chunks of fixed size. For each chunk, calculate the entropy.

    Parameters
    ----------
    chunk_size: int
        Number of hex values per chunk

    Return
    ------
    structural_entropy: list
        The entropy of every non-overlapping chunk
    """
    structural_entropy = []
    num_chunks = int(len(hex_sequence) / chunk_size)
    for i in range(num_chunks):
        chunk = hex_sequence[i * chunk_size:(i + 1) * chunk_size]
        len_chunk = float(len(chunk))
        counts = np.bincount(chunk)
        probs = counts / len_chunk
        entropy = sum([-p * math.log(p, log) if p > 0 else 0 for p in probs])
        structural_entropy.append(entropy)

    structural_entropy = np.array(structural_entropy, dtype=np.float32)
    return structural_entropy