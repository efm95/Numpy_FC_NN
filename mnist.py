import os
from urllib import request
import gzip
import numpy as np
from typing import Tuple

import framework as lib


class MNIST:
    FILES = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    URL = "http://yann.lecun.com/exdb/mnist/"

    @staticmethod
    def gzload(file, offset):
        with gzip.open(file, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    def __init__(self, set, cache="./cache"):
        os.makedirs(cache, exist_ok=True)

        for name in self.FILES:
            path = os.path.join(cache, name)
            if not os.path.isfile(path):
                print("Downloading " + name)
                request.urlretrieve(self.URL + name, path)

        if set=="test":
            f_offset = 2
        elif set=="train":
            f_offset = 0
        else:
            assert False, "Invalid set: "+set

        self.images = self.gzload(os.path.join(cache, self.FILES[f_offset]), 16).reshape(-1,28*28).astype(np.float)/255.0
        self.labels = self.gzload(os.path.join(cache, self.FILES[f_offset+1]), 8)

    def __len__(self) -> int:
        return self.images.shape[0]


train_validation_set = MNIST("train")
test_set = MNIST("test")

n_train = int(0.7 * len(train_validation_set))
print("MNIST:")
print("   Train set size:", n_train)
print("   Validation set size:", len(train_validation_set) - n_train)
print("   Test set size", len(test_set))

np.random.seed(12345)
batch_size = 64

loss = lib.CrossEntropy()
learning_rate = 0.1

model = lib.Sequential([
    lib.Linear(28*28, 20),
    lib.Tanh(),
    lib.Linear(20, 10),
    lib.Softmax()
])

indices = np.random.permutation(len(train_validation_set))

train_indices = indices[:n_train]
validation_indices = indices[n_train:]


def verify(images: np.ndarray, targets: np.ndarray) -> Tuple[int, int]:
    y = model.forward(images)
    i = np.argmax(y,axis=1)

    num_ok = np.sum(i==targets)
    total_num = len(i)
    return num_ok, total_num


def test() -> float:
    accu = 0.0
    count = 0

    for i in range(0, len(test_set), batch_size):
        images = test_set.images[i:i + batch_size]
        labels = test_set.labels[i:i + batch_size]

        num_ok, total_num = verify(images,labels)

        accu += num_ok
        count += total_num

    return accu / count * 100.0


def validate() -> float:
    accu = 0.0
    count = 0

    for i in range(0, len(validation_indices), batch_size):
        images = train_validation_set.images[validation_indices[i: i+batch_size]]
        labels = train_validation_set.labels[validation_indices[i: i+batch_size]]

        ## Implement. Use the verify() function to verify your data.
        num_ok, total_num = verify(images, labels)

        accu += num_ok
        count += total_num
        
    return accu/count * 100.0

best_validation_accuracy = 0
best_epoch = -1

for epoch in range(1000):
    loss_tmp = 0
    for i in range(0,len(train_indices),batch_size):
        images = train_validation_set.images[train_indices[i: i+batch_size]]
        labels = train_validation_set.labels[train_indices[i: i+batch_size]]

        loss_tmp += lib.train_one_step(model,loss,learning_rate,images,labels)*len(labels)


    loss_value = loss_tmp/len(train_indices)
    validation_accuracy = validate()

    print("Epoch %d: loss: %f, validation accuracy: %.2f%%" % (epoch, loss_value, validation_accuracy))

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_epoch = epoch
        print("\tThe epoch %d has the current best accuracy equal to %.2f%%. We wait %d interations for having it." % (
            best_epoch, best_validation_accuracy, epoch - best_epoch))
    else: 
        if epoch-best_epoch>10:
            print("\tCurrent accuracy didn't change after %d iterations" %(epoch-best_epoch))
            break


print("Test set performance: %.2f%%" % test())



