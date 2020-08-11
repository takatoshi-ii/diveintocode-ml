"""Miscellaneous utility functions."""

from datetime import datetime
import colorsys
import os
import sys
from functools import reduce
from functools import wraps

import math
import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine.base_layer import Layer
from PIL import Image
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation, add, Lambda, concatenate, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adadelta, Adagrad
from keras.regularizers import l2
from keras.utils import multi_gpu_model
import random as rd
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from clr_callback import CyclicLR #https://github.com/bckenstler/CLR

np.set_printoptions(precision=3, suppress=True)
MAX_VERTICES = 0
ANGLE_STEP  = 15
NUM_ANGLES3  = 0
NUM_ANGLES  = 0

grid_size_multiplier = 4
anchor_mask =[0]
anchors_per_level = len(anchor_mask)


dropped_boxes = 0
used_boxes = 1


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw = image.shape[1]
    ih = image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    cvi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cvi = cv.resize(cvi, (nw, nh), interpolation=cv.INTER_CUBIC)
    dx = int((w - nw) // 2)
    dy = int((h - nh) // 2)
    new_image = np.zeros((h, w, 3), dtype='uint8')
    new_image[...] = 128
    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = cvi
    else:
        new_image = cvi[-dy:-dy + h, -dx:-dx + w, :]

    return new_image.astype('float32') / 255.0


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

'''
def get_random_data(annotation_line, input_shape, random=True, max_boxes=80, hue_alter=10, sat_alter=30, val_alter=30, proc_img=True):
    # load data

    line = annotation_line.split()
    image = cv.imread(line[0])  # the color conversion is later. it is not necessary to realize bgr->rgb->hsv->rgb
    iw = image.shape[1]
    ih = image.shape[0]
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            # image = image.resize((nw, nh), Image.BICUBIC)
            image = cv.cvtColor(cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC), cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.
        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box), 0:5] = box[:, 0:5]
        return image_data, box_data

    # resize image
    random_scale = rd.uniform(.6, 1.0)
    scale = min(w / iw, h / ih)
    nw = int(iw * scale * random_scale)
    nh = int(ih * scale * random_scale)

    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            nw = int(nw * rd.uniform(.8, 1.0))
        else:
            nh = int(nh * rd.uniform(.8, 1.0))

    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)

    #CLAHE adjust
    if np.random.rand() < 0.1:
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))
        image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # place image
    nwiw = nw / iw
    nhih = nh / ih
    dx = rd.randint(0, max(w - nw, 0))
    dy = rd.randint(0, max(h - nh, 0))

    new_image = np.full((h, w, 3), 128, dtype='uint8')

    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = image[0:nh, 0:nw, :]
    else:
        new_image = image[-dy:-dy + h, -dx:-dx + w, :]

    # flip image or not
    flip = rd.random() < .5
    if flip:  new_image = cv.flip(new_image, 1)

    # distort image
    hsv = np.int32(cv.cvtColor(new_image, cv.COLOR_BGR2HSV))

    # linear hsv distortion
    hsv[..., 0] += rd.randint(-hue_alter, hue_alter)
    hsv[..., 1] += rd.randint(-sat_alter, sat_alter)
    hsv[..., 2] += rd.randint(-val_alter, val_alter)

    # additional non-linear distortion
    if np.random.rand() < 0.5:
        hsv[..., 1] = hsv[..., 1] * rd.uniform(.7, 1.3)
        hsv[..., 2] = hsv[..., 2] * rd.uniform(.7, 1.3)

    hsv[..., 0][hsv[..., 0] > 179] = 179
    hsv[..., 0][hsv[..., 0] < 0] = 0
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 1][hsv[..., 1] < 0] = 0
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv[..., 2][hsv[..., 2] < 0] = 0

    image_data = cv.cvtColor(np.uint8(hsv), cv.COLOR_HSV2RGB).astype('float32') / 255.0

    if np.random.rand() < 0.1:
        image_data = np.clip(image_data + np.random.rand() * image_data.std() * np.random.random(image_data.shape), 0, 1)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))

    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nwiw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nhih + dy

        if flip: box[:, [0, 2]] = (w - 1) - box[:, [2, 0]]

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] >= w] = w - 1
        box[:, 3][box[:, 3] >= h] = h - 1
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        if len(box) > max_boxes: box = box[:max_boxes]

        box_data[:len(box), 0:5] = box[:, 0:5]

    return image_data, box_data
'''


def get_random_data(annotation_line, input_shape, random=True, max_boxes=80, jitter=.05, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()

    image = cv.cvtColor(cv.imread(line[0]), cv.COLOR_BGR2RGB)
    #image = Image.open(line[0])
    #iw, ih = image.size
    iw = image.shape[1]
    ih = image.shape[0]
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    if np.random.rand() < .6: #jackpot! we return the data in original not augmented form
        random = False

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            # image = image.resize((nw, nh), Image.BICUBIC)
            image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)
            image = Image.fromarray(image)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data


    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.8, 1.2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    #image = image.resize((nw, nh), Image.BICUBIC)
    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_CUBIC)


    image = Image.fromarray(image)

    nwiw = nw/iw
    nhih = nh/ih

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    #flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    if np.random.rand() < .05:
        image_data = np.clip(image_data + np.random.rand() * image_data.std() * np.random.random(image_data.shape), 0, 1)


    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nwiw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nhih + dy
        if flip: box[:, [0, 2]] = (w-1) - box[:, [2, 0]]

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] >= w] = w-1
        box[:, 3][box[:, 3] >= h] = h-1
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]

    return image_data, box_data




"""YOLO_v3 Model Defined in Keras."""


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


def resblock_body_top(x, top, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    xx = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(top)
    x = Concatenate()([x, xx])
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se_resnet.py
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = LeakyReLU(alpha=0.1)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())





def app_block(x, filters):
    filt = filters // 8
    y1 = Conv2D(filt, (3, 3), padding='same', use_bias=False, strides=(1, 1), dilation_rate=(2, 2),
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = Conv2D(filt, (3, 3), padding='same', use_bias=False, strides=(1, 1), dilation_rate=(4, 4),
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = Conv2D(filt, (3, 3), padding='same', use_bias=False, strides=(1, 1), dilation_rate=(8, 8),
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y4 = Conv2D(filt, (3, 3), padding='same', use_bias=False, strides=(1, 1), dilation_rate=(16, 16),
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)

    y1 = Mish()(BatchNormalization()(y1))
    y2 = Mish()(BatchNormalization()(y2))
    y3 = Mish()(BatchNormalization()(y3))
    y4 = Mish()(BatchNormalization()(y4))


    y = Add()([y1, y2, y3, y4])

    y = Concatenate()([x, y])
    return y



def resblock_body_v4(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode

    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(x)


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    base = 5  # orig base = 8
    main_top = x
    x = DarknetConv2D_BN_Leaky(base * 2, (3, 3))(x)

    top = AveragePooling2D((2,2))(main_top)
    x = resblock_body_top(x, top, base * 6, 1)

    top = AveragePooling2D((4, 4))(main_top)
    x = resblock_body_top(x, top, base * 14, 2)
    x = app_block(x, base * 14)
    tiny = x

    top = AveragePooling2D((8, 8))(main_top)
    x = resblock_body_top(x, top, base * 28, 8)
    x = app_block(x, base * 28)
    small = x
    x = resblock_body(x, base * 50, 8)
    x = app_block(x, base * 50)
    medium = x
    x = resblock_body(x, base * 100, 8)
    x = app_block(x, base * 100)
    big = x

    return tiny, small, medium, big


def darknet_body_v4(x):
    base = 4
    x = DarknetConv2D_BN_Mish(base*4, (3,3))(x)
    x = resblock_body_v4(x, base * 8, 1, False)

    x = resblock_body_v4(x, base * 16, 2)
    x = app_block(x, base * 16)
    tiny = x

    x = resblock_body_v4(x, base * 32, 8)
    x = app_block(x, base * 32)
    small = x

    x = resblock_body_v4(x, base * 64, 8)
    x = app_block(x, base * 64)
    medium = x

    x = resblock_body_v4(x, base * 128, 8)
    x = app_block(x, base * 128)
    big = x

    return tiny, small, medium, big



def make_last_layers(x, num_filters, out_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    tiny, small, medium, big = darknet_body_v4(inputs)

    base = 4
    tiny   = DarknetConv2D_BN_Leaky(base*32, (1, 1))(tiny)
    small  = DarknetConv2D_BN_Leaky(base*32, (1, 1))(small)
    medium = DarknetConv2D_BN_Leaky(base*32, (1, 1))(medium)
    big    = DarknetConv2D_BN_Leaky(base*32, (1, 1))(big)

    all = Add()([medium, UpSampling2D(2,interpolation='bilinear')(big)])
    all = Add()([small, UpSampling2D(2,interpolation='bilinear')(all)])
    all = Add()([tiny, UpSampling2D(2,interpolation='bilinear')(all)])



    num_filters = base*32

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(all)

    all = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5 + NUM_ANGLES3), (1, 1)))(x)

    return Model(inputs, all)



def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = anchors_per_level
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(tf.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1], name='yolo_head/tile/reshape/grid_y'),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(tf.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1], name='yolo_head/tile/reshape/grid_x'),
                    [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1, name='yolo_head/concatenate/grid')
    grid = K.cast(grid, K.dtype(feats))
    global _var
    _var = [grid_shape, feats, anchors_tensor]
    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + NUM_ANGLES3], name='yolo_head/reshape/feats')

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))

    box_confidence      = K.sigmoid(feats[..., 4:5])
    box_class_probs     = K.sigmoid(feats[..., 5:5 + num_classes])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes




def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=80,
              score_threshold=.5,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    input_shape = K.shape(yolo_outputs)[1:3] * grid_size_multiplier
    boxes, box_scores = yolo_boxes_and_scores(yolo_outputs, anchors[anchor_mask], num_classes, input_shape, image_shape)


    mask = box_scores >= score_threshold
    box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
        vstup je to nase kratke
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # tady jsou uz stredy
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[:,:, 5:NUM_ANGLES3 + 5:3] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask), 5 + num_classes + NUM_ANGLES3),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    # pro vsechny boxy

    global dropped_boxes
    global used_boxes

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            if n in anchor_mask:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[0][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[0][0]).astype('int32')
                k = anchor_mask.index(n)
                c = true_boxes[b, t, 4].astype('int32')

                #if y_true[0][b, j, i, k, 4] == 1:
                #    dropped_boxes += 1
                #else:
                #    used_boxes += 1

                y_true[0][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[0][b, j, i, k, 4] = 1
                y_true[0][b, j, i, k, 5 + c] = 1
                y_true[0][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES3] = true_boxes[b, t, 5: 5 + NUM_ANGLES3]
    #print(' ', dropped_boxes, used_boxes, dropped_boxes / used_boxes * 100.0)
    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_diou(b1, b2):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    # calculate param v and alpha to extend to CIoU
    #v = 4*K.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    #alpha = v / (1.0 - iou + v)
    #diou = diou - alpha*v

    diou = K.expand_dims(diou, -1)
    return diou


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    #sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss

def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = 1
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * grid_size_multiplier, K.dtype(y_true[0]))
    loss = 0

    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for layer in range(num_layers):
        object_mask = y_true[layer][..., 4:5]
        true_class_probs = y_true[layer][..., 5:5 + num_classes]
        true_class_probs = _smooth_labels(true_class_probs, 0.1)

        grid, raw_pred, pred_xy, pred_wh= yolo_head(yolo_outputs[layer], anchors[anchor_mask], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]
        box_loss_scale *= 2

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                                                                                                             from_logits=True) * ignore_mask
        confidence_loss *= box_loss_scale

        raw_true_box = y_true[layer][..., 0:4]
        diou = box_diou(pred_box, raw_true_box)
        diou_loss = object_mask * box_loss_scale * (1 - diou)
        diou_loss = K.sum(diou_loss) / mf

        confidence_loss = K.sum(confidence_loss) / mf

        loss += (2*diou_loss + confidence_loss) / (K.sum(object_mask)/mf + 1)
    return loss


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'yolo_classes.txt',
        "score": 0.5,
        "iou": 0.5,
        "model_image_size": (448,864),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        #self.sess = K.get_session()
        #self.sess = tf.compat.v1.keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
            print('loading model with architecture')
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), anchors_per_level, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
            print('loading weights for architecture defined in the source code')
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5 + NUM_ANGLES3), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))


        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                     len(self.class_names), self.input_image_shape,
                                                     score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            image_data = np.expand_dims(letterbox_image(image, tuple(reversed(self.model_image_size))), 0)
        else:
            print('THE functionality is not implemented!')

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })


        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":

    """
    Retrain the YOLO model for your own dataset.
    """


    def _main():
        phase = 1

        annotation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_pedestrians.txt'
        validation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating_pedestrians.txt'
        log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model4\phase1\log/'
        #classes_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes_pedest_only.txt'
        #anchors_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors_full_res.txt'
        classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes_pedest_only.txt'
        anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors_full_res.txt'
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)
        #input_shape = (960, 1952) #yeah, we go full res!
        input_shape = (480, 960) #yeah, we go full res!


        os.makedirs(log_dir, exist_ok=True)

        if phase == 1:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
        else:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path='training/ep011-loss7.768-val_loss5.317.h5')


        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=False, period=1, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, delta=0.03)

        clr = CyclicLR(base_lr=0.01, max_lr=1.0, step_size=500, mode='triangular2')

        with open(annotation_path) as f:
            lines = f.readlines()

        with open(validation_path) as f:
            lines_val = f.readlines()

        num_val = int(len(lines_val))
        num_train = len(lines)



        batch_size = 1 # 2, 3, 10
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))



        model.compile(optimizer=Adadelta(1.0), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        epochs = 50
        
        data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, False),
        #steps_per_epoch=max(1, num_train // batch_size),
        steps_per_epoch=1000
        validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes, False)
        validation_steps=max(1, num_val // batch_size)
        epochs=epochs
        initial_epoch=0
        callbacks=[checkpoint, clr]
        print("★★★★★input_shape = ", input_shape)
        
        history = model.fit_generator(data_generator_wrapper (lines, batch_size, input_shape, anchors, num_classes, False),
                                      steps_per_epoch=1000,
                                      validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes, False),
                                      validation_steps=max(1, num_val // batch_size),
                                      epochs=epochs,
                                      initial_epoch=0,
                                      callbacks=[checkpoint, clr])



    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='model_data/yolo_weights.h5'):
        """create the training model"""
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)
        y_true = Input(shape=(h // grid_size_multiplier, w // grid_size_multiplier, anchors_per_level, num_classes + 5 + NUM_ANGLES3))

        model_body = yolo_body(image_input, anchors_per_level, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [model_body.output, y_true])
        model = Model([model_body.input, y_true], model_loss)

        # print(model.summary())
        return model


    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, is_random):
        """data generator for fit_generator"""
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0: np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=is_random)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)


    def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)



    if __name__ == '__main__':
        _main()
