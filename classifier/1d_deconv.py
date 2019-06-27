import keras.backend as K
from keras import Sequential
from keras.engine.topology import Layer
from keras.layers import Lambda, Conv2DTranspose


class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        print("build", input_shape)
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x, axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:, 0]))
        self._model.summary()
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)
