import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda, Variable, optimizers, Chain
from chainer.training import extensions

xp = cuda.cupy

separated = 1000
dataset = chainer.datasets.image_dataset.LabeledImageDataset(
            pairs='./train_master.txt', root='train_resized'
        )


def transform(in_data):
    img, label = in_data
    img, label = xp.array(img), xp.array(label)
    img /= 255
    return img, label


dataset = chainer.datasets.TransformDataset(dataset, transform)
train, test = dataset[:separated], dataset[separated:]


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1=L.Convolution2D(3, 256, 11, pad=5),
            bn1=L.BatchNormalization(512),
            conv2=L.Convolution2D(256, 512, 3, pad=1),
            bn2=L.BatchNormalization(512),
            conv3=L.Convolution2D(512, 512, 3, pad=1),
            bn3=L.BatchNormalization(512),
            conv4=L.Convolution2D(512, 512, 3, pad=1),
            bn4=L.BatchNormalization(512),
            conv5=L.Convolution2D(512, 512, 3, pad=1),
            bn5=L.BatchNormalization(512),
            conv6=L.Convolution2D(512, 512, 3, pad=1),
            fc1=L.Linear(None, 512),
            fc2=L.Linear(512, 24),
        )

    def __call__(self, x, train=True):
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu, 2)
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv2(h), test=not train)), 2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv3(h), test=not train)), 2)
        h = F.max_pooling_2d(F.relu(self.bn3(self.conv4(h), test=not train)), 2)
        h = F.max_pooling_2d(F.relu(self.bn4(self.conv5(h), test=not train)), 2)
        h = F.max_pooling_2d(F.relu(self.bn5(self.conv6(h), test=not train)), 2)
        h = F.dropout(h)
        h = F.dropout(F.relu(self.fc1(h)), train=train)
        return self.fc2(h)


cuda.get_device(1).use()
model = L.Classifier(Model())
model.to_gpu()
optimizer = optimizers.Adam(alpha=0.0001)
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, 64)
test_iter = chainer.iterators.SerialIterator(test, 64, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=1)
trainer = training.Trainer(updater, (400, 'epoch'), out="logs")
trainer.extend(extensions.Evaluator(test_iter, model, device=1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
