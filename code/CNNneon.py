from neon.backends import gen_backend
from neon.layers import Conv, Affine, Pooling
from neon.initializers import Uniform
from neon.transforms.activation import Rectlin, Softmax
from neon.models import Model
from neon.initializers import Kaiming
from neon.optimizers import Adadelta
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator
import numpy as np
from Readfile import DataSet, readfile
be = gen_backend(backend='cpu', batch_size=30)
data = readfile('PreImage', 'label.csv')
X = data.train_data
Y = data.train_label-1
X_test = data.test_data
Y_test = data.test_label-1
train_set = ArrayIterator(X=X, y=Y, nclass=11, lshape=(1,200,200))
test_set = ArrayIterator(X_test, None, nclass=11, lshape=(1,200,200))
init_uni = Uniform(low=-0.1, high=0.1)
layers = [Conv(fshape=(4,4,16), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Conv(fshape=(4,4,32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Conv(fshape=(4,4,32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Affine(nout=500, init=init_uni, activation=Rectlin()),
          Affine(nout=11, init=init_uni, activation=Softmax())]

model = Model(layers)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
optimizer = GradientDescentMomentum(learning_rate=0.005,
                                    momentum_coef=0.9)
callbacks = Callbacks(model, train_set)

model.fit(dataset=train_set, cost=cost, optimizer=optimizer,  num_epochs=40, callbacks=callbacks)
model.save_params('model.pkl')
# out = model.get_outputs(test_set)
# row = len(Y_test)
# result = np.zeros((row,1))
# i=0
# while i<row:
# 	result[i] = out[i].argmax()
# 	i=i+1
# np.save('result.npy', result)