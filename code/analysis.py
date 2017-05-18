import numpy as np 
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
from Readfile import DataSet, readfile
be = gen_backend(backend='cpu', batch_size=30)
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

model.load_params('model.pkl')
data = readfile('PreImage', 'label.csv')
X_test = data.test_data
test_set = ArrayIterator(X_test, None, nclass=11, lshape=(1,200,200))
true = data.test_label
out = model.get_outputs(test_set)
row = len(X_test)
pred = np.zeros((row,1))
i=0
while i<row:
	pred[i] = out[i].argmax()
	i=i+1
pred=pred+1
loss = abs(true - pred)
print(loss)
count = 0
for i in range(len(loss)):
	if loss[i] != 0:
		count=count+1
print(count)
print(1-float(count)/110)
