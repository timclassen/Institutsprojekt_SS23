import numpy as np
import pickle as pkl


def read_bitstream(file_path, debug=False):
    if debug:
        with open(file_path, "rb") as file:
            return pkl.load(file)
    return np.fromfile(file_path, dtype=np.uint8)


def write_bitstream(file_path, bitstream, debug=False):
    with open(file_path, "wb") as file:
        try:
            file.write(bitstream)
        except TypeError:
            pkl.dump(bitstream, file)
