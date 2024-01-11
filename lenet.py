import tinygrad as tg
from tinygrad import Tensor, nn
import struct
import numpy as np
from pathlib import Path

FILE = Path(__file__)


def load_images(p: str, labels: list[str]):
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
    # print(labels)

    images = load_images(
        FILE.parent / "data/mnist/t10k-images.idx3-ubyte", labels)


class Lenet:
    def __init__(self):
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.c3 = nn.Conv2d()
        self.c5 = nn.Linear()

    def __call__(self, x):
        x = self.c1(x)
        x = x.avg_pool2d()
        x = self.c3(x)
        x = x.avg_pool2d()
        x = self.c5(x)

if __name__ == "__main__":
    main()
