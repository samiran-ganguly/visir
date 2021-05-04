'''
Copyright (c) 2020, Samiran Ganguly
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Virginia nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Samiran Ganguly ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Samiran Ganguly BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import tensorflow as tf
from tf.keras.models import Model
from tf.keras.layers import Input
from tf.keras.layers import Dense
from tf.keras.layers import Conv2D
from tf.keras.layers import Conv2DTranspose
from tf.keras.layers import LeakyReLU
from tf.keras.layers import LocallyConnected2D
from tf.keras.layers import BatchNormalization
from tf.keras.layers import MaxPooling2D
from tf.keras.layers import Dropout
from tf.keras.layers import Flatten
from tf.keras.layers import Activation
from tf.keras.layers import Lambda
from tf.keras.optimizers import Adam
from tf.keras.utils.vis_utils import plot_model

class vizir:
	def __init__(self,Name,LearningRate = 0.0001,img_shape,latent_dim):
		self.Name = Name
		self.LearningRate = LearningRate
		
	def feature_extractor_network(self,in_image):
		# input
		in_image = Input(shape = in_shape)
		# C1 Layer
		nett = Conv2D(32,(5,5))(in_image)		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		# M2 Layer
		nett = MaxPooling2D(pool_size = (3,3))(nett)
		# C3 Layer
		nett = Conv2D(64,(3,3))		
		nett = BatchNormalization(pool_size = (3,3))(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		# L4 Layer
		nett = LocallyConnected2D(128,(3,3))(nett)
		# L5 Layer
		nett = LocallyConnected2D(256,(3,3))(nett)
		# F6 Layer
		nett = Dense(512,activation='relu')(nett)
		nett = Dropout(0.2)(nett)
		# F7 Layer 
		out_features = Dense(activation='tanh')(nett)
		# output
		model = Model(in_image,out_features)
		return model

	def generator_network(self):
		# input
		in_latents = Input(shape = (self.latent_dim,))
		#DC1
		nett = Conv2DTranspose(512,(3,3))(in_latents)		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		#DC2
		nett = Conv2DTranspose(128,(3,3))(nett)	
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		#DC3
		nett = Conv2DTranspose(64,(3,3))		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		#DC4
		nett = Conv2DTranspose(32,(5,5))(nett)		
		nett = BatchNormalization()(nett)
		out_image = Dense(alpha = 0.2)(nett)
		#output
		model = Model(in_latents,out_image)
		return model
	
	def discriminator_network(self):
        # input
        in_image = Input(shape=self.img_shape)
        # C1 layer
        nett = Conv2D(64,(5,5))(in_image)		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		# C2 layer
        nett = Conv2D(128,(5,5))(nett)		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		nett = Dropout(0.2)(nett)
		# C3 layer
		nett = Conv2D(256,(5,5))(nett)		
		nett = BatchNormalization()(nett)
		nett = LeakyReLU(alpha = 0.2)(nett)
		nett = Dropout(0.2)(nett)
        # F4 layer
        nett = Flatten()(nett)
        validity = Dense(1,alpha = 0.2)(nett)
        #output
        model =  Model(in_image,validity)
        return model
    
    
        
	
	
	




