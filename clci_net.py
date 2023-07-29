from keras import *
from keras.layers import *
import tensorflow as tf

kernel_regularizer = regularizers.l2(1e-5)
bias_regularizer = regularizers.l2(1e-5)
# Regularization parameters (these are currently set to None, but you can modify them if needed)
kernel_regularizer = None
bias_regularizer = None

def conv_lstm(input1, input2, channel=256):
    """
    Creates a ConvLSTM2D layer by combining two input tensors.

    This function reshapes the input tensors to add a time dimension and then concatenates them along the time
    dimension. It then applies a ConvLSTM2D layer with specified channel size, kernel size, strides, and
    optional regularization.

    Args:
        input1 (tf.Tensor): The first input tensor to the ConvLSTM layer.
        input2 (tf.Tensor): The second input tensor to the ConvLSTM layer.
        channel (int): The number of output channels (filters) of the ConvLSTM layer. Default is 256.

    Returns:
        tf.Tensor: The output tensor of the ConvLSTM layer.

    """
    
    # Reshape inputs to add the time dimension for ConvLSTM
    lstm_input1 = Reshape((1, input1.shape.as_list()[1], input1.shape.as_list()[2], input1.shape.as_list()[3]))(input1)
    lstm_input2 = Reshape((1, input2.shape.as_list()[1], input2.shape.as_list()[2], input1.shape.as_list()[3]))(input2)

    # Concatenate the two inputs along the time dimension
    lstm_input = custom_concat(axis=1)([lstm_input1, lstm_input2])
    
    # Create ConvLSTM2D layer
    x = ConvLSTM2D(channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(lstm_input)
    return x

def conv_2(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    """
    Defines a 2D Convolution block with optional regularization.

    This function creates a 2D convolutional block consisting of two Conv2D layers with batch normalization
    and ReLU activation functions. It allows for specifying the number of filters (channels), kernel size,
    strides, kernel initializer, and an optional regularization.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional block.
        filter_num (int): The number of filters (channels) in the convolutional layers.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel, respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).
        kernel_initializer (str): The initializer for the convolutional kernels. Options include 'glorot_uniform'
                                  and 'he_normal'. Default is 'glorot_uniform'.
        kernel_regularizer (tf.keras.regularizers.Regularizer or None): An optional regularization applied
                                                                       to the convolutional kernel. Default is None.

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(conv_)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)   
    return conv_

def conv_2_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    """
    Defines a 2D Convolution block with 'he_normal' kernel initializer.

    This function creates a 2D convolutional layer with a specified number of filters (channels),
    kernel size, strides, and 'he_normal' kernel initializer.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel, respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer) 

def conv_2_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    """
    Defines a 2D Convolution block 

    This function creates a 2D convolutional layer with a specified number of filters (channels),
    kernel size, strides, 'he_normal' kernel initializer, and L2 regularization with a weight decay
    of 5e-4.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel, respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4)) 

def conv_1(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    """
    Defines a 2D Convolution block

    This function creates a 2D convolutional layer with a specified number of filters (channels),
    kernel size, strides, kernel initializer, and optional regularization.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel, respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).
        kernel_initializer (str): The kernel initializer for the convolutional layer. It can be either
                                   'glorot_uniform' or 'he_normal' or any other valid initializer
                                   available in Keras. Default is 'glorot_uniform'.
        kernel_regularizer (tf.keras.regularizers.Regularizer or None): Optional regularization to be
                                                                       applied to the kernel weights of
                                                                       the convolutional layer.
                                                                       Default is None.

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_1_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    """
    Defines a 2D Convolution block

    This function creates a 2D convolutional layer with a specified number of filters (channels),
    kernel size, strides, and 'he_normal' kernel initializer. Additionally, it allows for optional
    regularization by providing the `kernel_regularizer` parameter, which can be set to None if no
    regularization is desired.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer) 

def conv_1_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    """
    Creates a 2D Convolution block

    This function defines a 2D convolutional layer with a specified number of filters (channels),
    kernel size, strides, 'he_normal' kernel initializer, and fixed L2 regularization with a
    regularization strength of 5e-4.

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        kernel_size (tuple): The size of the convolutional kernel, specified as a tuple of two
                             integers representing the height and width of the kernel respectively.
                             Default is (3, 3).
        strides (tuple): The strides of the convolution along the height and width dimensions,
                         specified as a tuple of two integers. Default is (1, 1).

    Returns:
        tf.Tensor: The output tensor of the convolutional block.

    """
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def dilate_conv(inputs, filter_num, dilation_rate):
    """
    Creates a dilated Conv2D layer.

    This function defines a 2D convolutional layer with a specified number of filters and
    dilation rate. 

    Args:
        inputs (tf.Tensor): The input tensor to the convolutional layer.
        filter_num (int): The number of filters (channels) in the convolutional layer.
        dilation_rate (tuple): The dilation rate for the convolutional layer, specified
                               as a tuple of two integers representing the dilation rate
                               along the height and width dimensions respectively.

    Returns:
        tf.Tensor: The output tensor of the dilated convolutional layer.

    """
    conv_ = Conv2D(filter_num, kernel_size=(3,3), dilation_rate=dilation_rate, padding='same', kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

class custom_concat(Layer):
    """
    Custom Keras layer to perform concatenation along a specified axis.

    This layer takes multiple input tensors and concatenates them along the specified axis.
    The concatenation is performed element-wise along the axis, preserving the dimensions
    of all other axes.

    Args:
        axis (int): The axis along which to concatenate the inputs. The default value is -1,
                    which corresponds to the last axis.

    Attributes:
        axis (int): The axis along which the concatenation is performed.

    """
    def __init__(self, axis=-1, **kwargs):
        """
        Initializes the custom_concat layer.

        Args:
            axis (int): The axis along which to concatenate the inputs.

        """
        super(custom_concat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """
        Builds the custom_concat layer.

        Args:
            input_shape (tuple): The shape of the input tensor(s).

        """
        # Create a trainable weight variable for this layer.
        self.built = True
        super(custom_concat, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        """
        Performs the concatenation operation on the input tensors.

        Args:
            x (list): A list of input tensors to be concatenated.

        Returns:
            tf.Tensor: The concatenated tensor.

        """
        self.res = tf.concat(x, self.axis)

        return self.res

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the custom_concat layer.

        Args:
            input_shape (tuple): The shape of the input tensor(s).

        Returns:
            tuple: The shape of the concatenated output tensor.

        """
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
        """
        Initializes the BilinearUpsampling layer.

        Args:
            upsampling: A tuple specifying the upsampling factor along the height and width dimensions.
                       Default is (2, 2), i.e., upsampling by a factor of 2 in both height and width.
            **kwargs: Additional keyword arguments to pass to the base class (Layer).

        """
        super(BilinearUpsampling, self).__init__(**kwargs)       
        self.upsampling = upsampling
        
    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer based on the input shape.

        Args:
            input_shape: A tuple representing the input shape (batch_size, height, width, channels).

        Returns:
            Tuple representing the output shape (batch_size, new_height, new_width, channels)
            after applying the upsampling factor.

        """
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        """
        Performs the upsampling operation using bilinear interpolation.

        Args:
            inputs: The input tensor (batch_size, height, width, channels).

        Returns:
            The upscaled tensor obtained using bilinear interpolation.

        """
        return tf.image.resize(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

def concat_pool(conv, pool, filter_num, strides=(2, 2)):
    """
    Concatenates a Convolutional layer with a Pooling layer.

    Args:
        conv: Input Convolutional layer.
        pool: Input Pooling layer.
        filter_num: Number of filters for the Convolutional layer.
        strides: Strides for the Convolutional layer. Default is (2, 2).

    Returns:
        A concatenated layer obtained by concatenating the Convolutional layer and the Pooling layer.
    """
    conv_downsample = Conv2D(filter_num, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(conv)
    conv_downsample = BatchNormalization()(conv_downsample)
    conv_downsample = Activation('relu')(conv_downsample)
    concat_pool_ = Concatenate()([conv_downsample, pool])
    return concat_pool_

######################################
# Importing required libraries for the model
from keras.optimizers import Adam
import keras.backend as K

def CLCI_Net(input_shape=(224, 176, 1), num_class=1):
    """
    Creates the CLCI_Net model for semantic segmentation.

    Args:
        input_shape: Tuple representing the shape of the input tensor (height, width, channels).
                     The row and column of the input should be resized or cropped to an integer multiple of 16.
        num_class: Number of classes for segmentation. For binary segmentation, num_class should be set to 1.

    Returns:
        Model: Keras model representing the CLCI_Net for semantic segmentation.

    """
    
    inputs = Input(shape=input_shape)

    # Encoder blocks
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

    # Decoder blocks
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

    # Final convolutional layer with sigmoid or softmax activation based on the number of classes
    if num_class == 1:
        conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)
    else:
        conv10 = Conv2D(num_class, (1, 1), activation='softmax')(conv9)

    # Create and return the model
    model = Model(inputs=inputs, outputs=conv10)

    return model

def CLF_ASPP(conv5, conv1, conv2, conv3, conv4, input_shape):
    """
    Creates the ASPP (Atrous Spatial Pyramid Pooling) block.

    Args:
        conv5: Convolutional layer from the 5th encoder stage.
        conv1: Convolutional layer from the 1st encoder stage.
        conv2: Convolutional layer from the 2nd encoder stage.
        conv3: Convolutional layer from the 3rd encoder stage.
        conv4: Convolutional layer from the 4th encoder stage.
        input_shape: Shape of the input tensor (batch_size, height, width, channels).

    Returns:
        Output tensor after the ASPP block.

    """

    # Branches with different dilation rates
    b0 = conv_1_init(conv5, 256, (1, 1))
    b1 = dilate_conv(conv5, 256, dilation_rate=(2, 2))
    b2 = dilate_conv(conv5, 256, dilation_rate=(4, 4))
    b3 = dilate_conv(conv5, 256, dilation_rate=(6, 6))

    # Branch for global pooling
    out_shape0 = input_shape[0] // pow(2, 4)
    out_shape1 = input_shape[1] // pow(2, 4)
    b4 = AveragePooling2D(pool_size=(out_shape0, out_shape1))(conv5)
    b4 = conv_1_init(b4, 256, (1, 1))
    b4 = BilinearUpsampling((out_shape0, out_shape1))(b4)

    # Convolutional layers for downsampled feature maps from different encoder stages
    clf1 = conv_1_init(conv1, 256, strides=(16, 16))
    clf2 = conv_1_init(conv2, 256, strides=(8, 8))
    clf3 = conv_1_init(conv3, 256, strides=(4, 4))
    clf4 = conv_1_init(conv4, 256, strides=(2, 2))

    # Concatenate all the branches
    outs = Concatenate()([clf1, clf2, clf3, clf4, b0, b1, b2, b3, b4])

    # Additional convolutional layer for feature aggregation
    outs = conv_1_init(outs, 256 * 4, (1, 1))
    # Dropout regularization to prevent overfitting
    outs = Dropout(0.5)(outs)

    return outs