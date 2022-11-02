'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Khalid A. Alobaid
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA
 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.
'''

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU




def cnn_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


class CNN_Model(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def cnn_model(self):
        input = Input(shape=self.image_shape)

        model = Conv2D(filters=64, kernel_size=11, strides=1, padding="same")(input)
        model = LeakyReLU(alpha=0.2)(model)

        model = cnn_block(model, 64, 11, 2)
        model = cnn_block(model, 128, 11, 1)
        model = cnn_block(model, 128, 11, 2)
        model = cnn_block(model, 256, 11, 1)
        model = cnn_block(model, 256, 11, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)

        model = Dense(1)(model)
        model = Activation('linear')(model)

        model = Model(inputs=input, outputs=model)

        return model
