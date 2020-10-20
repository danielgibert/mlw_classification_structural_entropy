from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import extract_structural_entropy
import matplotlib.pyplot as plt
import pywt


def plot_structural_entropy(input_filepath,
                            output_cA,
                            output_cD,
                            chunk_size=256,
                            log=257):
    #Extract the structural entropy
    metaPHOR = MetaPHOR(input_filepath)
    bytes_sequence = metaPHOR.get_hexadecimal_data_as_list()
    hex_list = [int(hex_value, 16) if hex_value != '??' else 257 for hex_value in bytes_sequence]

    # Calculate entropy for each chunk
    structural_entropy = extract_structural_entropy(hex_list, chunk_size=chunk_size, log=log)
    # Extract wavelets approximation and coefficients
    (cA, cD) = pywt.dwt(structural_entropy, 'haar')
    plot1 = plt.figure(1)
    plt.ylabel('Approximation coefficients')
    plt.xlabel('Chunk')
    plt.plot(cA)

    plot2 = plt.figure(2)
    plt.ylabel('Details coefficients')
    plt.xlabel('Chunk')
    plt.plot(cD)

    plot1.savefig(output_cA)
    plot2.savefig(output_cD)





# Usage example
plot_structural_entropy("../../../data/dataSample/0A32eTdBKayjCWhZqDOQ.asm",
                        "../../../data/wavelets_examples/0A32eTdBKayjCWhZqDOQ/0A32eTdBKayjCWhZqDOQ_cA.png",
                        "../../../data/wavelets_examples/0A32eTdBKayjCWhZqDOQ/0A32eTdBKayjCWhZqDOQ_cD.png")

