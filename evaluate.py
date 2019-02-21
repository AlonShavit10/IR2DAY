disply_img=1 # disply images in the browser (in addition to save them to the drive)
ir_path="data/pics/full_test"
#ir_path="data/pics/night3"

vis_ref_folder="data/pics/vis_ref" # the pic to be used as style code
max_pics_to_evaluate=9999999
eval_name='total evaluation'

use_random_style=False # if ture- use a random style (don't extract from ref day image)
random_syle_factor=1.0 # factor for random STD
pass_over_2=False # for faster eval- skip every second image

import os

try:
    import os
    os.chdir('project_dir')
except:
    print ('Already mounted! dirctory was not changed')

###  Import packages
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K
from keras.utils import conv_utils
from keras import initializers, regularizers, constraints
from keras.utils.generic_utils import get_custom_objects
import numpy as np
import time
import glob
from PIL import Image
from IPython.display import clear_output
from IPython.display import display
import cv2
from skimage.measure import compare_ssim as ssim
from skimage import data, img_as_float
import matplotlib.pyplot as plt
%matplotlib inline

! pip install Augmentor
import Augmentor

from tensorflow.python.client import device_lib


def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)

class GroupNormalization(Layer):
    def __init__(self, axis=-1,
                 gamma_init='one', beta_init='zero',
                 gamma_regularizer=None, beta_regularizer=None,
                 epsilon=1e-6, 
                 group=32,
                 data_format=None,
                 **kwargs): 
        super(GroupNormalization, self).__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group
        self.data_format = K.common.normalize_data_format(data_format)
        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]       
        if self.data_format == 'channels_last':
            channel_axis = -1
            shape[channel_axis] = input_shape[channel_axis]
        elif self.data_format == 'channels_first':
            channel_axis = 1
            shape[channel_axis] = input_shape[channel_axis]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4 and len(input_shape) != 2:
            raise ValueError('Inputs should have rank ' +
                             str(4) + " or " + str(2) +
                             '; Received input shape:', str(input_shape))

        if len(input_shape) == 4:
            if self.data_format == 'channels_last':
                batch_size, h, w, c = input_shape
                if batch_size is None:
                    batch_size = -1
                
                if c < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(c) +
                                     '; Group size: ' + str(self.group)
                                    )

                x = K.reshape(inputs, (batch_size, h, w, self.group, c // self.group))
                mean = K.mean(x, axis=[1, 2, 4], keepdims=True)
                std = K.sqrt(K.var(x, axis=[1, 2, 4], keepdims=True) + self.epsilon)
                x = (x - mean) / std

                x = K.reshape(x, (batch_size, h, w, c))
                return self.gamma * x + self.beta
            elif self.data_format == 'channels_first':
                batch_size, c, h, w = input_shape
                if batch_size is None:
                    batch_size = -1
                
                if c < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(c) +
                                     '; Group size: ' + str(self.group)
                                    )

                x = K.reshape(inputs, (batch_size, self.group, c // self.group, h, w))
                mean = K.mean(x, axis=[2, 3, 4], keepdims=True)
                std = K.sqrt(K.var(x, axis=[2, 3, 4], keepdims=True) + self.epsilon)
                x = (x - mean) / std

                x = K.reshape(x, (batch_size, c, h, w))
                return self.gamma * x + self.beta
                
        elif len(input_shape) == 2:
            reduction_axes = list(range(0, len(input_shape)))
            del reduction_axes[0]
            batch_size, _ = input_shape
            if batch_size is None:
                batch_size = -1
                
            mean = K.mean(inputs, keepdims=True)
            std = K.sqrt(K.var(inputs, keepdims=True) + self.epsilon)
            x = (inputs  - mean) / std
            
            return self.gamma * x + self.beta
            

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'group': self.group
                 }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'InstanceNormalization': InstanceNormalization})
device_lib.list_local_devices()

### Config

K.set_learning_phase(1)

channel_axis=-1
channel_first = False



# Architecture configs
IMAGE_SHAPE = (128, 128, 3)
nc_in = 3 # number of input channels of generators
nc_D_inp = 3 # number of input channels of discriminators
n_dim_style = 8
n_resblocks = 4 # number of residual blocks in decoder and content encoder
n_adain = 2*n_resblocks
n_dim_adain = 256
nc_base = 64 # Number of channels of the first conv2d of encoder
n_downscale_content = 2 # Number of content encoder dowscaling
n_dowscale_style = 4 # Number of style encoder dowscaling
use_groupnorm = False # else use_layer norm in upscaling blocks
w_l2 = 1e-4 # L2 weight regularization

# Optimization configs
use_lsgan = True
use_mixup = True
mixup_alpha = 0.2
batchSize = 1 # was 4
conv_init_dis = RandomNormal(0, 0.02) # initializer of dicriminators' conv layers
conv_init = 'he_normal' # initializer of generators' conv layers
lrD = 0.00001 # Discriminator learning rate
lrG = 0.00001 # Generator learning rate
opt_decay = 0 # Learning rate decay over each update.
TOTAL_ITERS = 300000 # Max training iterations

# Loss weights for generators
w_D = 1 # Adversarial loss (Gan)
lambda_s=1; # Latent codes- style
lambda_c=1; # Latent codes- content
luma_w_ir=7;
chrome_w_ir=3;
luma_w_vis=7;
chrome_w_vis=4;

# 5. Define models

##################################################################################
# Basic Blocks
##################################################################################
def conv_block(input_tensor, f, k=3, strides=2, use_norm=False):
    x = input_tensor
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=k, strides=strides, kernel_initializer=conv_init,
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=(not use_norm), padding="valid")(x)
    if use_norm:
        x = InstanceNormalization(epsilon=1e-5)(x)
    x = Activation("relu")(x)
    return x

def conv_block_d(input_tensor, f, use_norm=False):
    x = input_tensor
    x = ReflectPadding2D(x, 2)
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init_dis,
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=(not use_norm), padding="valid")(x)
    if use_norm:
        x = InstanceNormalization(epsilon=1e-5)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def upscale_nn(inp, f, use_norm=False):
    x = inp
    x = UpSampling2D()(x)
    x = ReflectPadding2D(x, 2)
    x = Conv2D(f, kernel_size=5, kernel_initializer=conv_init, 
               kernel_regularizer=regularizers.l2(w_l2), 
               use_bias=(not use_norm), padding='valid')(x)
    if use_norm:
        if use_groupnorm:
            x = GroupNormalization(group=32)(x)
        else:
            x = GroupNormalization(group=f)(x) # group=f equivalant to layer norm
    x = Activation('relu')(x)
    return x
  
def ReflectPadding2D(x, pad=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
    return x

##################################################################################
# Discriminators
##################################################################################
def Discriminator(nc_in, input_size=IMAGE_SHAPE[0]):
    inp = Input(shape=(input_size, input_size, nc_in))
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
    x = ReflectPadding2D(x, 2)
    out = Conv2D(1, kernel_size=5, kernel_initializer=conv_init_dis, 
                 kernel_regularizer=regularizers.l2(w_l2),
                 use_bias=False, padding="valid")(x)  
    if not use_lsgan:
        x = Activation('sigmoid')(x) 
    return Model(inputs=[inp], outputs=out)

def Discriminator_MS(nc_in, input_size=IMAGE_SHAPE[0]):
    # Multi-scale discriminator architecture
    inp = Input(shape=(input_size, input_size, nc_in))
    
    def conv2d_blocks(inp, nc_base=64, n_layers=3):
        x = inp
        dim = nc_base
        for _ in range(n_layers):
            x = conv_block_d(x, dim, False)
            dim *= 2
        x = Conv2D(1, kernel_size=1, kernel_initializer=conv_init_dis,
                   kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=True, padding="valid")(x)
        if not use_lsgan:
            x = Activation('sigmoid')(x)
        return x
    
    x0 = conv2d_blocks(inp)    
    ds1 = AveragePooling2D(pool_size=(3, 3), strides=2)(inp)
    x1 = conv2d_blocks(ds1)
    ds2 = AveragePooling2D(pool_size=(3, 3), strides=2)(ds1)
    x2 = conv2d_blocks(ds2)
    return Model(inputs=[inp], outputs=[x0, x1, x2])

##################################################################################
# Encoder - style
##################################################################################    
def Encoder_style_MUNIT(nc_in=3, input_size=IMAGE_SHAPE[0], n_dim_adain=256, n_dim_style=n_dim_style, nc_base=nc_base, n_dowscale_style=n_dowscale_style):
    # Style encoder architecture 
    inp = Input(shape=(input_size, input_size, nc_in))
    x = ReflectPadding2D(inp, 3)
    x = Conv2D(64, kernel_size=7, kernel_initializer=conv_init, 
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=True, padding="valid")(x)
    x = Activation('relu')(x)   
    
    dim = 1
    for i in range(n_dowscale_style):
        dim = 4 if dim >= 4 else dim*2
        x = conv_block(x, dim*nc_base)
    x = GlobalAveragePooling2D()(x)    
    style_code = Dense(n_dim_style, kernel_regularizer=regularizers.l2(w_l2))(x) # Style code
    return Model(inp, style_code)
##################################################################################
# Encoder - content
##################################################################################   
def Encoder_content_MUNIT(nc_in=3, input_size=IMAGE_SHAPE[0], n_downscale_content=n_downscale_content, nc_base=nc_base):
    # Content encoder architecture 
    def res_block_content(input_tensor, f):
        x = input_tensor
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, 
                   kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = InstanceNormalization(epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, 
                   kernel_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = InstanceNormalization(epsilon=1e-5)(x)
        x = add([x, input_tensor])
        return x      
    
    inp = Input(shape=(input_size, input_size, nc_in))
    x = ReflectPadding2D(inp, 3)
    x = Conv2D(64, kernel_size=7, kernel_initializer=conv_init, 
               kernel_regularizer=regularizers.l2(w_l2),
               use_bias=False, padding="valid")(x)
    x = InstanceNormalization()(x) #
    x = Activation('relu')(x) # 
    
    dim = 1
    ds = 2**n_downscale_content
    for i in range(n_downscale_content):
        dim = 4 if dim >= 4 else dim*2
        x = conv_block(x, dim*nc_base, use_norm=True)
    for i in range(n_resblocks):
        x = res_block_content(x, dim*nc_base)
    content_code = x # Content code
    return Model(inp, content_code)

##################################################################################
# Decoder
##################################################################################  
def MLP_MUNIT(n_dim_style=n_dim_style, n_dim_adain=n_dim_adain, n_blk=3, n_adain=2*n_resblocks):
    # MLP for AdaIN parameters
    inp_style_code = Input(shape=(n_dim_style,))
    
    adain_params = Dense(n_dim_adain, kernel_regularizer=regularizers.l2(w_l2), activation='relu')(inp_style_code)
    for i in range(n_blk - 2):
        adain_params = Dense(n_dim_adain, kernel_regularizer=regularizers.l2(w_l2), activation='relu')(adain_params)
    adain_params = Dense(2*n_adain*n_dim_adain, kernel_regularizer=regularizers.l2(w_l2))(adain_params) # No output activation 
    return Model(inp_style_code, [adain_params])
  
def Decoder_MUNIT(nc_in=256, input_size=IMAGE_SHAPE[0]//(2**n_downscale_content), n_dim_adain=256, n_resblocks=n_resblocks):
    def op_adain(inp):
        x = inp[0]
        mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
        adain_bias = inp[1]
        adain_bias = K.reshape(adain_bias, (-1, 1, 1, n_dim_adain))
        adain_weight = inp[2]
        adain_weight = K.reshape(adain_weight, (-1, 1, 1, n_dim_adain))        
        out = tf.nn.batch_normalization(x, mean, var, adain_bias, adain_weight, variance_epsilon=1e-7)
        return out
      
    def AdaptiveInstanceNorm2d(inp, adain_params, idx_adain):
        assert inp.shape[-1] == n_dim_adain
        x = inp
        idx_head = idx_adain*2*n_dim_adain
        adain_weight = Lambda(lambda x: x[:, idx_head:idx_head+n_dim_adain])(adain_params)
        adain_bias = Lambda(lambda x: x[:, idx_head+n_dim_adain:idx_head+2*n_dim_adain])(adain_params)
        out = Lambda(op_adain)([x, adain_bias, adain_weight])
        return out
      
    def res_block_adain(inp, f, adain_params, idx_adain):
        x = inp
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, 
                   kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = Lambda(lambda x: AdaptiveInstanceNorm2d(x[0], x[1], idx_adain))([x, adain_params])     
        x = Activation('relu')(x)
        x = ReflectPadding2D(x)
        x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, 
                   kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2),
                   use_bias=False, padding="valid")(x)
        x = Lambda(lambda x: AdaptiveInstanceNorm2d(x[0], x[1], idx_adain+1))([x, adain_params])    
        
        x = add([x, inp])
        return x  
    
    inp_style = Input((n_dim_style,))
    style_code = inp_style
    mlp = MLP_MUNIT()
    adain_params = mlp(style_code)
    
    inp_content = Input(shape=(input_size, input_size, nc_in))
    content_code = inp_content
    x = inp_content
    
    for i in range(n_resblocks):
        x = res_block_adain(x, nc_in, adain_params, 2*i) 
        
    dim = 1
    for i in range(n_downscale_content):
        dim = dim if nc_in//dim <= 64 else dim*2
        x = upscale_nn(x, nc_in//dim, use_norm=True)
    x = ReflectPadding2D(x, 3)
    out = Conv2D(3, kernel_size=7, kernel_initializer=conv_init, 
                 kernel_regularizer=regularizers.l2(w_l2), 
                 padding='valid', activation="tanh")(x)
    return Model([inp_style, inp_content], [out, style_code, content_code])

from keras.backend.common import normalize_data_format

encoder_style_A = Encoder_style_MUNIT()
encoder_content_A = Encoder_content_MUNIT()
encoder_style_B = Encoder_style_MUNIT()
encoder_content_B = Encoder_content_MUNIT()
decoder_A = Decoder_MUNIT()
decoder_B = Decoder_MUNIT()

x = Input(shape=IMAGE_SHAPE) # dummy input tensor
netGA = Model(x, decoder_A([encoder_style_A(x), encoder_content_A(x)]))
netGB = Model(x, decoder_B([encoder_style_B(x), encoder_content_B(x)]))

netDA = Discriminator_MS(nc_D_inp)
netDB = Discriminator_MS(nc_D_inp)

## improved
encoder_style_A_improved = Encoder_style_MUNIT()
encoder_content_A_improved = Encoder_content_MUNIT()
encoder_style_B_improved = Encoder_style_MUNIT()
encoder_content_B_improved = Encoder_content_MUNIT()
decoder_A_improved = Decoder_MUNIT()
decoder_B_improved = Decoder_MUNIT()

x_improved = Input(shape=IMAGE_SHAPE) # dummy input tensor
netGA_improved = Model(x_improved, decoder_A_improved([encoder_style_A_improved(x_improved), encoder_content_A_improved(x_improved)]))
netGB_improved = Model(x_improved, decoder_B_improved([encoder_style_B_improved(x_improved), encoder_content_B_improved(x_improved)]))

netDA_improved = Discriminator_MS(nc_D_inp)
netDB_improved = Discriminator_MS(nc_D_inp)

from keras.models import model_from_json
#download ref weights
with open('TIR2LAb_ref/TIR2Lab_model.json') as f:
        json_string = f.readline()
transformer = model_from_json(json_string)
transformer.load_weights('TIR2LAb_ref/TIR2Lab_weights.h5')
import keras as k
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color
import cv2
import matplotlib.pyplot as plt
batchSize=1
def postprocess_tir2lab_results(batch, validationPath):		
	rescaled_batch = lab_rescale(batch)
	for idx, image in enumerate(rescaled_batch):
		image = color.lab2rgb(image.astype('float64'))
		image = image*255
		im = Image.fromarray(np.squeeze(image).astype('uint8'))
		im.save(validationPath + '/I000' + str(idx) + '.png')
		
def lab_rescale(im):
    im[:,:,:,0] = im[:,:,:,0]*100.0
    im[:,:,:,1] = im[:,:,:,1]*185.0 - 87.0
    im[:,:,:,2] = im[:,:,:,2]*203.0 - 108.0
    return im

##
def getCompare (img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256)) 
    img=cv2.copyMakeBorder(img,0,0,32,32,cv2.BORDER_CONSTANT)
    img2=img/256

    
    
    img3 = img2.reshape(256, 320, 1)
    img3 = np.reshape(img2,(1,256,320,1))
    
    
    
    output=transformer.predict(img3)
    
    my_try=output.reshape(256, 320,3)
    
    
    output_2=output
    output_2[:,:,:,0] = output_2[:,:,:,0]*100.0
    output_2[:,:,:,1] = output_2[:,:,:,1]*185.0 - 87.0
    output_2[:,:,:,2] = output_2[:,:,:,2]*203.0 - 108.0
    
    my_try=output_2.reshape(256, 320,3)
    
    output3=color.lab2rgb(my_try.astype('float64')) # output2
    
    source=output3
    trim=source[0:255,32:320-33]
    trim_resize=np.float32(cv2.resize(trim,(128,128)))
    final_output=trim_resize.reshape(1,128,128,3)
    
    return final_output


print("Loading Improved...")
try:
    encoder_style_A_improved.load_weights("data/weights/encoder_style_A.h5")
    encoder_style_B_improved.load_weights("data/weights/encoder_style_B.h5")
    encoder_content_A_improved.load_weights("data/weights/encoder_content_A.h5")
    encoder_content_B_improved.load_weights("data/weights/encoder_content_B.h5")
    decoder_A_improved.load_weights("data/weights/decoder_A.h5")
    decoder_B_improved.load_weights("data/weights/decoder_B.h5")
    netGA_improved.load_weights("data/weights/netGA.h5") # was disabled 
    netGB_improved.load_weights("data/weights/netGB.h5") # was disabled  
    netDA_improved.load_weights("data/weights/netDA.h5") 
    netDB_improved.load_weights("data/weights/netDB.h5") 
    print ("Model Improved weights files are successfully loaded")
except:
    print ("Error occurs during Improved weights files loading.")
    pass
    

    

print("Loading Regular...")
try:
    encoder_style_A.load_weights("data/weights_original/encoder_style_A.h5")
    encoder_style_B.load_weights("data/weights_original/encoder_style_B.h5")
    encoder_content_A.load_weights("data/weights_original/encoder_content_A.h5")
    encoder_content_B.load_weights("data/weights_original/encoder_content_B.h5")
    decoder_A.load_weights("data/weights_original/decoder_A.h5")
    decoder_B.load_weights("data/weights_original/decoder_B.h5")
    netGA.load_weights("data/weights_original/netGA.h5") # was disabled 
    netGB.load_weights("data/weights_original/netGB.h5") # was disabled  
    netDA.load_weights("data/weights_original/netDA.h5") 
    netDB.load_weights("models_4_layers/netDB.h5") 
    print ("Model Original weights files are successfully loaded")
except:
    print ("Error occurs during original weights files loading.")
    pass


	
	




def model_paths(netEnc_content, netEnc_style, netDec):
    fn_content_code = K.function([netEnc_content.inputs[0]], [netEnc_content.outputs[0]])
    fn_style_code = K.function([netEnc_style.inputs[0]], [netEnc_style.outputs[0]])
    fn_decoder_rgb = K.function(netDec.inputs, [netDec.outputs[0]])
    
    fake_output = netDec.outputs[0]   
    fn_decoder_out = K.function(netDec.inputs, [fake_output])
    return fn_content_code, fn_style_code, fn_decoder_out

path_content_code_A, path_style_code_A, path_decoder_A = model_paths(encoder_content_A, encoder_style_A, decoder_A)
path_content_code_B, path_style_code_B, path_decoder_B = model_paths(encoder_content_B, encoder_style_B, decoder_B)

path_content_code_A_improved, path_style_code_A_improved, path_decoder_A_improved = model_paths(encoder_content_A_improved, encoder_style_A_improved, decoder_A_improved)
path_content_code_B_improved, path_style_code_B_improved, path_decoder_B_improved = model_paths(encoder_content_B_improved, encoder_style_B_improved, decoder_B_improved)



### Utils for Visualization

def translation(src_imgs, style_image, fn_content_code_src, fn_style_code_tar, fn_decoder_rgb_tar, rand_style=use_random_style):
    # Cross domain translation function
    # This funciton is for visualization purpose
    """
    Inputs:
        src_img: source domain images, shape=(input_batch_size, h, w, c)
        style_image: target style images,  shape=(input_batch_size, h, w, c)
        fn_content_code_src: Source domain K.function of content encoder
        fn_style_code_tar: Target domain K.function of style encoder
        fn_decoder_rgb_tar: Target domain K.function of decoder
        rand_style: sample style codes from normal distribution if set True.
    Outputs:
        fake_rgb: output tensor of decoder having chennels [R, G, B], shape=(input_batch_size, h, w, c)
    """
    batch_size = src_imgs.shape[0]
    content_code = fn_content_code_src([src_imgs])[0]
    if rand_style:
        style_code = np.random.normal(size=(batch_size, n_dim_style))*random_syle_factor
        #style_code = np.random.uniform(size=(batch_size, n_dim_style))  
    elif style_image is None:
        style_code = fn_style_code_tar([src_imgs])[0]
    else:
        style_code = fn_style_code_tar([style_image])[0]
    
    fake_rgb = fn_decoder_rgb_tar([style_code, content_code])[0]
    return fake_rgb

def showG_improved(test_A, test_B,ref_img,name,save_folder,ground_truth):
    sample_imgs_pBtA = []
    sample_imgs_pAtB = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i+1], None, path_content_code_A, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B, rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA.append(im)
        
    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i+1], None, path_content_code_B, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_B.shape[0])]))   
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A, rand_style=True)[0]
            for i in range(test_B.shape[0])])) 
        sample_imgs_pAtB.append(im)
        
##########improved
    sample_imgs_pBtA_improved = []
    sample_imgs_pAtB_improved = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i+1], None, path_content_code_A_improved, path_style_code_A_improved, path_decoder_A_improved)[0] 
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A_improved, path_style_code_B_improved, path_decoder_B_improved)[0] 
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A_improved, path_style_code_B_improved, path_decoder_B_improved, rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA_improved.append(im)
        
    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i+1], None, path_content_code_B_improved, path_style_code_B_improved, path_decoder_B_improved)[0] 
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B_improved, path_style_code_A_improved, path_decoder_A_improved)[0] 
        for i in range(test_B.shape[0])]))   
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B_improved, path_style_code_A_improved, path_decoder_A_improved, rand_style=True)[0]
            for i in range(test_B.shape[0])])) 
        sample_imgs_pAtB_improved.append(im)
        
    figure_A = np.concatenate([
        np.squeeze(test_A),
        imgs_pAtA,
        imgs_pBtA,
        sample_imgs_pBtA[0],
        sample_imgs_pBtA[1],
        sample_imgs_pBtA[2],
        sample_imgs_pBtA[3],
        sample_imgs_pBtA[4]
        ], axis=-2 )
    figure_A = figure_A.reshape(batchSize*IMAGE_SHAPE[0], 8*IMAGE_SHAPE[1], 3)
    figure_A = np.clip((figure_A + 1) * 255 / 2, 0, 255).astype('uint8')
    figure_B = np.concatenate([
        np.squeeze(test_B),
        sample_imgs_pAtB[0],
        sample_imgs_pAtB_improved[0],
        np.squeeze(ref_img),
        np.squeeze(ground_truth)
        ], axis=-2 )
    figure_B = figure_B.reshape(batchSize*IMAGE_SHAPE[0], 5*IMAGE_SHAPE[1], 3)
    figure_B = np.clip((figure_B + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = np.concatenate([figure_B], axis=1)
    img_to_disply=Image.fromarray(figure)
    #name=str(counter)
    #img_to_disply.save(save_folder+"/"+name+".jpg")
    img_to_disply.save(save_folder+"/"+name)
    if (disply_img==1):
      display(img_to_disply)
    
def showG(test_A, test_B):
    sample_imgs_pBtA = []
    sample_imgs_pAtB = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i+1], None, path_content_code_A, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B, rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA.append(im)
        
    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i+1], None, path_content_code_B, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_B.shape[0])]))   
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A, rand_style=True)[0]
            for i in range(test_B.shape[0])])) 
        sample_imgs_pAtB.append(im)
        
    figure_A = np.concatenate([
        np.squeeze(test_A),
        imgs_pAtA,
        imgs_pBtA,
        sample_imgs_pBtA[0],
        sample_imgs_pBtA[1],
        sample_imgs_pBtA[2],
        sample_imgs_pBtA[3],
        sample_imgs_pBtA[4]
        ], axis=-2 )
    figure_A = figure_A.reshape(batchSize*IMAGE_SHAPE[0], 8*IMAGE_SHAPE[1], 3)
    figure_A = np.clip((figure_A + 1) * 255 / 2, 0, 255).astype('uint8')
    figure_B = np.concatenate([
        np.squeeze(test_B),
        sample_imgs_pAtB[0]
        ], axis=-2 )
    figure_B = figure_B.reshape(batchSize*IMAGE_SHAPE[0], 2*IMAGE_SHAPE[1], 3)
    figure_B = np.clip((figure_B + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = np.concatenate([figure_B], axis=1)
    img_to_disply=Image.fromarray(figure)
    name="alon"
    img_to_disply.save("lwir2day_data/collections/lwir3_128/output/"+name+".jpg")
    display(img_to_disply)    

import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage import data, img_as_float
from scipy import spatial


#########################################################TEST##################

def getLABimg (in_img):
    #in_img=((in_img+1)/2)*255
    in_img=in_img*255
    in_img=in_img.astype(np.uint8)
    
    b,g,r = cv2.split(in_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    
    # LAB
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    return lab

# assuems to input image is RGB, between -1 to 1
def getSsim (img1,img2):
    img1=getLABimg (img1)
    img2=getLABimg (img2)
    l1,a1,b1 = cv2.split(img1)       
    l2,a2,b2 = cv2.split(img2)
    img1 = img_as_float(l1)
    img2 = img_as_float(l2)   
    ssim_output = ssim(img1, img2,
                      data_range=img2.max() - img2.min())
    return ssim_output



'''
iter_num starts from 0 
old_score_tuple= the old ***avg*** score
new_score_tuple= the new score
current_iter_num=the current iter (starts from 0)

return- the updated avg score
''' 
# iter_num starts from 0    
def update_score(old_score_tuple,new_score_tuple,current_iter_num):
    if current_iter_num==0:
        output=new_score_tuple
    else:
        output=tuple(np.ndarray.tolist((((np.multiply(old_score_tuple,(current_iter_num))+new_score_tuple)/(current_iter_num+1)))))
    return output

#0=ssim_vals_luma 1=ssim_vals_3_ch 2=cosine_similarity 3=matchedSquereDiss 4=l2_diss 5=l1_diss 6=ncc
def get_score_from_img(images):
    
    ssim_vals_luma=[]
    for img in images:
        ssim_vals_luma.append(np.round(getSsim(images[0],img),3))
        
    ssim_vals_3_ch=[] 
    for img in images:
        ssim_vals_3_ch.append(np.round(ssim(images[0],img,multichannel = True),3))
        
    #https://stackoverflow.com/questions/42292685/calculate-similarity-of-picture-and-its-sketch
    cosine_similarity=[] 
    for img in images:
        grey_img_ref=(cv2.cvtColor(np.float32(images[0]), cv2.COLOR_BGR2GRAY).flatten())
        grey_imf_comp=cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY).flatten()
        cosine_similarity.append(np.round(1-spatial.distance.cosine(grey_img_ref,grey_imf_comp),3))
    
    #matchedSquereDiss
    matchedSquereDiss=[]
    for img in images:
        res = cv2.matchTemplate(np.float32(images[0]), np.float32(img), cv2.TM_SQDIFF_NORMED)
        matchedSquereDiss.append(np.round(1-res[0,0],3))
        
        
    # L2 disstance
    l2_diss=[] 
    for img in images:
        grey_img_ref=(cv2.cvtColor(np.float32(images[0]), cv2.COLOR_BGR2GRAY))
        grey_imf_comp=cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        res=np.sqrt(np.sum(np.power((grey_imf_comp-grey_img_ref),2)))/np.size(grey_img_ref)
        l2_diss.append(np.round(1-res,3))
        
    
    # L1 disstance
    l1_diss=[] 
    for img in images:
        grey_img_ref=(cv2.cvtColor(np.float32(images[0]), cv2.COLOR_BGR2GRAY))
        grey_imf_comp=cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        res=np.sum(np.abs((grey_imf_comp-grey_img_ref)))/np.size(grey_img_ref)
        l1_diss.append(np.round(1-res,3))
        
    #  normalized cross-correlation disstanve
        # L1 disstance
    ncc_diss=[] 
    for img in images:
        grey_img_ref=(cv2.cvtColor(np.float32(images[0]), cv2.COLOR_BGR2GRAY))
        grey_imf_comp=cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        res=np.sum((grey_img_ref-np.mean(grey_img_ref))*(grey_imf_comp-np.mean(grey_imf_comp)))/((np.size(grey_img_ref)-1)*np.std(grey_imf_comp)*np.std(grey_img_ref))
        ncc_diss.append(np.round(res,3))
    
    #return {'ssim_luma':ssim_vals_luma,'ssim':ssim_vals_3_ch,:'cosine':cosine_similarity,'matchedSqr':matchedSquereDiss,'l2':l2_diss,'l1':l1_diss,'ncc':ncc_diss}
    return (ssim_vals_luma,ssim_vals_3_ch,cosine_similarity,matchedSquereDiss,l2_diss,l1_diss,ncc_diss)

def showG_improved_score(test_A, test_B,ref_img,name,save_folder,ground_truth,current_iter_num):
    sample_imgs_pBtA = []
    sample_imgs_pAtB = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i+1], None, path_content_code_A, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A, path_style_code_B, path_decoder_B, rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA.append(im)
        
    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i+1], None, path_content_code_B, path_style_code_B, path_decoder_B)[0] 
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A)[0] 
        for i in range(test_B.shape[0])]))   
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B, path_style_code_A, path_decoder_A, rand_style=True)[0]
            for i in range(test_B.shape[0])])) 
        sample_imgs_pAtB.append(im)
        
##########improved
    sample_imgs_pBtA_improved = []
    sample_imgs_pAtB_improved = []
    imgs_pAtA = np.squeeze(np.array([
        translation(test_A[i:i+1], None, path_content_code_A_improved, path_style_code_A_improved, path_decoder_A_improved)[0] 
        for i in range(test_A.shape[0])]))
    imgs_pBtA = np.squeeze(np.array([
        translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A_improved, path_style_code_B_improved, path_decoder_B_improved)[0] 
        for i in range(test_A.shape[0])]))
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_A[i:i+1], test_B[i:i+1], path_content_code_A_improved, path_style_code_B_improved, path_decoder_B_improved, rand_style=True)[0]
            for i in range(test_A.shape[0])]))
        sample_imgs_pBtA_improved.append(im)
        
    imgs_pBtB = np.squeeze(np.array([
        translation(test_B[i:i+1], None, path_content_code_B_improved, path_style_code_B_improved, path_decoder_B_improved)[0] 
        for i in range(test_B.shape[0])]))
    imgs_pAtB = np.squeeze(np.array([
        translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B_improved, path_style_code_A_improved, path_decoder_A_improved)[0] 
        for i in range(test_B.shape[0])]))   
    for i in range(5):
        im = np.squeeze(np.array([
            translation(test_B[i:i+1], test_A[i:i+1], path_content_code_B_improved, path_style_code_A_improved, path_decoder_A_improved, rand_style=True)[0]
            for i in range(test_B.shape[0])])) 
        sample_imgs_pAtB_improved.append(im)
        
    figure_A = np.concatenate([
        np.squeeze(test_A),
        imgs_pAtA,
        imgs_pBtA,
        sample_imgs_pBtA[0],
        sample_imgs_pBtA[1],
        sample_imgs_pBtA[2],
        sample_imgs_pBtA[3],
        sample_imgs_pBtA[4]
        ], axis=-2 )
    figure_A = figure_A.reshape(batchSize*IMAGE_SHAPE[0], 8*IMAGE_SHAPE[1], 3)
    figure_A = np.clip((figure_A + 1) * 255 / 2, 0, 255).astype('uint8')
    figure_B = np.concatenate([
        np.squeeze(test_B),
        sample_imgs_pAtB[0],
        sample_imgs_pAtB_improved[0],
        np.squeeze(ref_img),
        np.squeeze(ground_truth)
        ], axis=-2 )
    figure_B = figure_B.reshape(batchSize*IMAGE_SHAPE[0], 5*IMAGE_SHAPE[1], 3)
    figure_B = np.clip((figure_B + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = np.concatenate([figure_B], axis=1)
    img_to_disply=Image.fromarray(figure)
    #name=str(counter)
    #img_to_disply.save(save_folder+"/"+name+".jpg")
    img_to_disply.save(save_folder+"/"+name)
    if (disply_img==1):
      display(img_to_disply)
    ###calculate score
    input_file_all=save_folder+"/"+name
    img_all=cv2.imread(input_file_all)/255
    ir=img_all[0:128, 0:128]
    my=img_all[0:128, 128:256]
    my_improved=img_all[0:128, 128+128:256+128]
    article_ref=img_all[0:128, 128+256:256+256]
    ground_truth=img_all[0:128, 128+384:256+384]
    
    images=[]
    images.append(ground_truth) #index 0 is ground truth
    images.append(article_ref) #index 1 is article ref
    images.append(my) #index 2 is my
    images.append(my_improved) #index 3 is my_improved
    

import os, os.path
import numpy as np
import datetime
now = datetime.datetime.now()
time_stamp=(now.strftime("%d-%m-%Y_%H-%M-%S_"))
save_folder_path="evaluate_output/"+eval_name+'_'+time_stamp
if not os.path.exists(save_folder_path):
  os.makedirs(save_folder_path)


pA = Augmentor.Pipeline(vis_ref_folder)
num_of_files= (len([name for name in os.listdir(ir_path) if os.path.isfile(os.path.join(ir_path, name))]))
gA = pA.keras_generator(batch_size=1)
imgs_A, _ = next(gA)

vis_A, _ = gA.send(batchSize) 
vis_A = vis_A*2 - 1 # transform [0, 1] to [-1, 1]


current_score=0
updated_score=0

count=0
file_list=os.listdir(ir_path)

#for filename in range(0, len(file_list), 2):
for filename in os.listdir(ir_path):
  if (count % 2 != 0) & (pass_over_2==True) :
    count=count+1
    continue
  if filename.endswith(".jpg"):   
    name=filename
    full_path=os.path.abspath(ir_path+"/"+name)
    img = cv2.imread(full_path) 
    
    img_orig=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig=np.float32(img_orig/255)*2-1

    
    img=np.float32(img/255)
    vis_B_both_channels=np.reshape(img,(1,128,256,3))
    vis_B=vis_B_both_channels[:,0:128, 0:128,:] # trim the IR
    vis_B_re=vis_B.reshape(128, 128,3)
    vis_B_re_int=np.uint8(vis_B_re*255)
    compared_img=getCompare (vis_B_re_int)
    compared_img = compared_img*2 - 1 # transform [0, 1] to [-1, 1]
    vis_B = vis_B*2 - 1 # transform [0, 1] to [-1, 1]
    ground_truth=img_orig[0:128, 128:256,:]
    showG_improved(vis_A, vis_B,compared_img,name,save_folder_path,ground_truth)
    
    
    ### calculate score
    input_file_all=save_folder_path+"/"+name
    img_all=cv2.imread(input_file_all)/255
    ir=img_all[0:128, 0:128]
    my=img_all[0:128, 128:256]
    my_improved=img_all[0:128, 128+128:256+128]
    article_ref=img_all[0:128, 128+256:256+256]
    ground_truth=img_all[0:128, 128+384:256+384]

    images=[]
    images.append(ground_truth) #index 0 is ground truth
    images.append(article_ref) #index 1 is article ref
    images.append(my) #index 2 is my
    images.append(my_improved) #index 3 is my_improved
    
    updated_score=get_score_from_img(images)
    current_score=update_score(current_score,updated_score,count)
    
    
    if count % 50 == 0:
      print ("****************** evaluating picture "+str(count)+" out of "+ str(num_of_files)+" ******************")
      # write score
      f = open(save_folder_path+'/'+'aa_score.txt', 'w')
      line = '\n'.join(str(x) for x in current_score)
      f.write(line + '\n')
      f.write("total num of pics is "+ str(count))
      f.close()
    if max_pics_to_evaluate==(count+1):
      # write score
      f = open(save_folder_path+'/'+'aa_score.txt', 'w')
      line = '\n'.join(str(x) for x in current_score)
      f.write(line + '\n')
      f.write("total num of pics is "+ str(count))
      f.close()
      break
    count=count+1
#score
f = open(save_folder_path+'/'+'aa_score.txt', 'w')
line = '\n'.join(str(x) for x in current_score)
f.write(line + '\n')
f.write("total num of pics is "+ str(count-1))
f.close()
print('Finished!!!!!!!!!!!!!!!')
print('All files were saved to '+save_folder_path)