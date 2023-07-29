from keras import *
from keras.layers import *
import tensorflow as tf
kernel_regularizer = regularizers.l2(1e-5)
bias_regularizer = regularizers.l2(1e-5)
kernel_regularizer = None
bias_regularizer = None

def conv_lstm(input1, input2, channel=256):
    lstm_input1 = Reshape((1, input1.shape.as_list()[1], input1.shape.as_list()[2], input1.shape.as_list()[3]))(input1)
    lstm_input2 = Reshape((1, input2.shape.as_list()[1], input2.shape.as_list()[2], input1.shape.as_list()[3]))(input2)

    lstm_input = custom_concat(axis=1)([lstm_input1, lstm_input2])
    x = ConvLSTM2D(channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(lstm_input)
    return x

def conv_2(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(conv_)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)   
    return conv_

def conv_2_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer) 

def conv_2_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4)) 

def conv_1(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_1_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer) 

def conv_1_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def dilate_conv(inputs, filter_num, dilation_rate):
    conv_ = Conv2D(filter_num, kernel_size=(3,3), dilation_rate=dilation_rate, padding='same', kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

class custom_concat(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(custom_concat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.built = True
        super(custom_concat, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.res = tf.concat(x, self.axis)

        return self.res

    def compute_output_shape(self, input_shape):
        # return (input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:]
        # print((input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:])
        input_shapes = input_shape
        output_shape = list(input_shapes[0])

        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]

        return tuple(output_shape)


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)       
        self.upsampling = upsampling
        
    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        #return tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
        #                                           int(inputs.shape[2] * self.upsampling[1])))
        return tf.image.resize(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))



def concat_pool(conv, pool, filter_num, strides=(2, 2)):
    conv_downsample = Conv2D(filter_num, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(conv)
    conv_downsample = BatchNormalization()(conv_downsample)
    conv_downsample = Activation('relu')(conv_downsample)
    concat_pool_ = Concatenate()([conv_downsample, pool])
    return concat_pool_
######################################
from keras.optimizers import Adam
import keras.backend as K
#from custom_layer import *


def CLCI_Net(input_shape=(224, 176, 1), num_class=1):
    # The row and col of input should be resized or cropped to an integer multiple of 16.
    inputs = Input(shape=input_shape)

    conv1 = conv_2_init(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    concat_pool11 = concat_pool(conv1, pool1, 32, strides=(2, 2))
    fusion1 = conv_1_init(concat_pool11, 64 * 4, kernel_size=(1, 1))

    conv2 = conv_2_init(fusion1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    concat_pool12 = concat_pool(conv1, pool2, 64, strides=(4, 4))
    concat_pool22 = concat_pool(conv2, concat_pool12, 64, strides=(2, 2))
    fusion2 = conv_1_init(concat_pool22, 128 * 4, kernel_size=(1, 1))

    conv3 = conv_2_init(fusion2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    concat_pool13 = concat_pool(conv1, pool3, 128, strides=(8, 8))
    concat_pool23 = concat_pool(conv2, concat_pool13, 128, strides=(4, 4))
    concat_pool33 = concat_pool(conv3, concat_pool23, 128, strides=(2, 2))
    fusion3 = conv_1_init(concat_pool33, 256 * 4, kernel_size=(1, 1))

    conv4 = conv_2_init(fusion3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    concat_pool14 = concat_pool(conv1, pool4, 256, strides=(16, 16))
    concat_pool24 = concat_pool(conv2, concat_pool14, 256, strides=(8, 8))
    concat_pool34 = concat_pool(conv3, concat_pool24, 256, strides=(4, 4))
    concat_pool44 = concat_pool(conv4, concat_pool34, 256, strides=(2, 2))
    fusion4 = conv_1_init(concat_pool44, 512 * 4, kernel_size=(1, 1))

    conv5 = conv_2_init(fusion4, 512)
    conv5 = Dropout(0.5)(conv5)

    clf_aspp = CLF_ASPP(conv5, conv1, conv2, conv3, conv4, input_shape)

    up_conv1 = UpSampling2D(size=(2, 2))(clf_aspp)
    up_conv1 = conv_1_init(up_conv1, 256, kernel_size=(2, 2))
    skip_conv4 = conv_1_init(conv4, 256, kernel_size=(1, 1))
    context_inference1 = conv_lstm(up_conv1, skip_conv4, channel=256)
    conv6 = conv_2_init(context_inference1, 256)

    up_conv2 = UpSampling2D(size=(2, 2))(conv6)
    up_conv2 = conv_1_init(up_conv2, 128, kernel_size=(2, 2))
    skip_conv3 = conv_1_init(conv3, 128, kernel_size=(1, 1))
    context_inference2 = conv_lstm(up_conv2, skip_conv3, channel=128)
    conv7 = conv_2_init(context_inference2, 128)

    up_conv3 = UpSampling2D(size=(2, 2))(conv7)
    up_conv3 = conv_1_init(up_conv3, 64, kernel_size=(2, 2))
    skip_conv2 = conv_1_init(conv2, 64, kernel_size=(1, 1))
    context_inference3 = conv_lstm(up_conv3, skip_conv2, channel=64)
    conv8 = conv_2_init(context_inference3, 64)

    up_conv4 = UpSampling2D(size=(2, 2))(conv8)
    up_conv4 = conv_1_init(up_conv4, 32, kernel_size=(2, 2))
    skip_conv1 = conv_1_init(conv1, 32, kernel_size=(1, 1))
    context_inference4 = conv_lstm(up_conv4, skip_conv1, channel=32)
    conv9 = conv_2_init(context_inference4, 32)


    if num_class == 1:
        conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)
    else:
        conv10 = Conv2D(num_class, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def CLF_ASPP(conv5, conv1, conv2, conv3, conv4, input_shape):

    b0 = conv_1_init(conv5, 256, (1, 1))
    b1 = dilate_conv(conv5, 256, dilation_rate=(2, 2))
    b2 = dilate_conv(conv5, 256, dilation_rate=(4, 4))
    b3 = dilate_conv(conv5, 256, dilation_rate=(6, 6))

    out_shape0 = input_shape[0] // pow(2, 4)
    out_shape1 = input_shape[1] // pow(2, 4)
    b4 = AveragePooling2D(pool_size=(out_shape0, out_shape1))(conv5)
    b4 = conv_1_init(b4, 256, (1, 1))
    b4 = BilinearUpsampling((out_shape0, out_shape1))(b4)

    clf1 = conv_1_init(conv1, 256, strides=(16, 16))
    clf2 = conv_1_init(conv2, 256, strides=(8, 8))
    clf3 = conv_1_init(conv3, 256, strides=(4, 4))
    clf4 = conv_1_init(conv4, 256, strides=(2, 2))

    outs = Concatenate()([clf1, clf2, clf3, clf4, b0, b1, b2, b3, b4])

    outs = conv_1_init(outs, 256 * 4, (1, 1))
    outs = Dropout(0.5)(outs)

    return outs