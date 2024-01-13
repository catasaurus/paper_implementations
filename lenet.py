from tinygrad import Tensor, nn
from pathlib import Path
import gzip
from tinygrad.helpers import fetch
import numpy as np

FILE = Path(__file__)

#
# def load_images(p: str, labels: list[str]):
#     with open(p, "rb") as imgpath:
#         magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
#         images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
#
#     return images
#
#
# def load_labels(p: str):
#     with open(p, "rb") as lbpath:
#         magic, n = struct.unpack(">II", lbpath.read(8))
#         labels = np.fromfile(lbpath, dtype=np.uint8)
#
#     return labels
#

# took this function from tinygrad extra


def fetch_mnist(tensors=False):
    def parse(file): return np.frombuffer(
        gzip.open(file).read(), dtype=np.uint8).copy()
    # http://yann.lecun.com/exdb/mnist/ lacks https
    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz")
                    )[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:]
    X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz")
                   )[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:]
    if tensors:
        return Tensor(X_train).reshape(-1, 1, 28, 28), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
    else:
        return X_train, Y_train, X_test, Y_test


def main():
    # labels = load_labels(
    #     FILE.parent / "data/mnist/t10k-labels.idx1-ubyte")
    # # print(labels)
    #
    # images = load_images(
    #     FILE.parent / "data/mnist/t10k-images.idx3-ubyte", labels)
    #
    # tinygrad Tensor conversion happens here as load_images uses numpy methods on labels
    # images = Tensor(images)
    # labels = Tensor(labels)

    X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=True)

    model = Lenet()
    optim = nn.optim.Adam(nn.state.get_parameters(model))

    for i in range(70):
        samples = Tensor.randint(512, high=X_train.shape[0])
        optim.zero_grad()
        loss = model(X_train[samples]).sparse_categorical_crossentropy(
            Y_train[samples]).backward()
        optim.step()
        print(i, loss.item())


class Lenet:
    def __init__(self):
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.c3 = nn.Conv2d(6, 16, 5)
        self.c5 = nn.Conv2d(16, 120, 5)
        self.f6 = nn.Linear(120, 84)

    def __call__(self, x: Tensor):
        x = self.c1(x)
        x = x.tanh()
        x = x.avg_pool2d()
        x = self.c3(x)
        x = x.tanh()
        x = x.avg_pool2d()
        x = self.c5(x)
        x = x.tanh()
        x = x.flatten(1)
        x = self.f6(x)
        x = x.log_softmax()

        return x


if __name__ == "__main__":
    main()
