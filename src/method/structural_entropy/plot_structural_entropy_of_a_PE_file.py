from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import extract_structural_entropy
import matplotlib.pyplot as plt


def plot_structural_entropy(input_filepath,
                            output_filepath,
                            chunk_size=256,
                            log=257):
    #Extract the structural entropy
    metaPHOR = MetaPHOR(input_filepath)
    bytes_sequence = metaPHOR.get_hexadecimal_data_as_list()
    hex_list = [int(hex_value, 16) if hex_value != '??' else 257 for hex_value in bytes_sequence]

    # Calculate entropy for each chunk
    structural_entropy = extract_structural_entropy(hex_list, chunk_size=chunk_size, log=log)
    plt.plot(structural_entropy)
    plt.ylabel('Entropy')
    plt.xlabel('Chunk')

    plt.savefig(output_filepath)


# Usage example
plot_structural_entropy("../../../data/dataSample/0A32eTdBKayjCWhZqDOQ.asm",
                        "../../../data/structural_entropy_examples/0A32eTdBKayjCWhZqDOQ.png")

