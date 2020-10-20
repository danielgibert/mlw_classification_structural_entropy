import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../../"))
import sys
import csv
sys.path.append(project_path)
from src.metaphor_engine import MetaPHOR
from src.ml.utils.utils import extract_structural_entropy
from src.ml.utils.utils import serialize_wavelet_example

import numpy as np
import pywt


MAX_HEX_VALUES = 16000000


def microsoft_dataset_to_tfrecords(pe_filepath,
                                   tfrecords_filepath,
                                   labels_filepath,
                                   chunk_size=4096,
                                   log=257
                                   ):
    tfwriter = tf.io.TFRecordWriter(tfrecords_filepath)

    i = 0
    NUM_MAX_SUBSETS = int(MAX_HEX_VALUES/chunk_size)

    # Training TFRecord
    with open(labels_filepath, "r") as labels_file:
        reader = csv.DictReader(labels_file, fieldnames=["Id",
                                                           "Class"])
        reader.__next__()
        for row in reader:
            print("{};{}".format(i, row['Id']))
            metaPHOR = MetaPHOR(pe_filepath + row['Id'] + ".asm")
            bytes_sequence = metaPHOR.get_hexadecimal_data_as_list()
            hex_list = [hex_value if hex_value != '??' else 257 for hex_value in bytes_sequence]

            # Extract structural entropy
            structural_entropy = extract_structural_entropy(hex_list, chunk_size=chunk_size, log=log)
            if len(structural_entropy) <= NUM_MAX_SUBSETS:
                structural_entropy = np.pad(structural_entropy, (0, NUM_MAX_SUBSETS - len(structural_entropy)),
                                            'constant',
                                            constant_values=(0, 0))
            else:
                structural_entropy = structural_entropy[:NUM_MAX_SUBSETS]

            # Haar wavelet decomposition
            (cA, cD) = pywt.dwt(structural_entropy, 'haar')

            serialized_example = serialize_wavelet_example(cA, cD, int(row['Class']) - 1)
            tfwriter.write(serialized_example)

            i += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wavelets-based TFWriter Script')
    parser.add_argument("pe_filepath",
                        type=str,
                        help="Filepath describing the location of the pe files in asm format")
    parser.add_argument("tfrecords_filepath",
                        type=str,
                        help="Where the TFRecord files will be stores")
    parser.add_argument("labels_filepath",
                        type=str,
                        help="CSV filepath containing the ID and class of each PE file in pe_filepath")
    parser.add_argument("--chunk_size",
                        type=int,
                        help="Size of chunks",
                        default=4096)
    parser.add_argument("--log",
                        type=int,
                        help="Size of chunks",
                        default=257)
    args = parser.parse_args()
    microsoft_dataset_to_tfrecords(args.pe_filepath,
                                   args.tfrecords_filepath,
                                   args.chunk_size,
                                   args.log)