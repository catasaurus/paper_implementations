import tinygrad as tg
import struct
import numpy as np
from pathlib import Path

FILE = Path(__file__)

def load_images(p: str):
    with open(p, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images


def load_labels(p: str):
    with open(p, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    return labels


def main():
    labels = load_labels(
        FILE.parent / "data/mnist/t10k-labels.idx1-ubyte")
    print(labels)


if __name__ == "__main__":
    main()
