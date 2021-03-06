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
import numpy as np
from tf.keras.models import Model
from tf.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, LocallyConnected2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Activation, Lambda
from tf.keras.optimizers import Adam
from tf.keras.losses import binary_crossentropy, mse
from tf.keras.utils.vis_utils import plot_model

class vizir:
    def __init__(self,Name,LearningRate = 0.0001,img_shape,latent_dim):
        self.Name = Name
        self.LearningRate = LearningRate
        self.img_shape = img_shape
        self.latent_dim = latent_dims
        self.trained = 0
        #self.source_img = tf.Variable()
        #self.target_img = tf.Variable()
        self.g_optimizer = Adam(learning_rate=self.LearningRate)
        self.d_optimizer = Adam(learning_rate=self.LearningRate)
        
        self.optimizer = Adam(learning_rate = self.LearningRate)
        
        # compile the IR Disciminator model - Target
        self.IR_discriminator = self.discriminator_network()
        self.IR_discriminator.compile(optimizer=self.optimizer,loss='binary_crossentropy',metric=['accuracy'])
        
        # compile the IR Generator model - Target
        self.IR_f_net = self.feature_extractor_network()
        self.IR_g_net = self.generator_network()(self.IR_f_net)
        self.IR_generator = Model(inputs = in_image_IR, outputs = self.IR_g_net)
        self.IR_generator.compile(optimizer=self.optimizer,loss='binary_crossentropy',metric=['accuracy'])
        
        # compile the Visual Disciminator model - Source
        self.Visual_discriminator = self.discriminator_network()
        self.Visual_discriminator.compile(optimizer=self.optimizer,loss='binary_crossentropy',metric=['accuracy'])
        
        # compile the Visual Generator model - Visual
        self.Visual_f_net = self.feature_extractor_network()
        self.Visual_g_net = self.generator_network()(self.Visual_f_net)
        self.Visual_generator = Model(inputs = in_image_Vis, outputs = self.Visual_g_net)
        self.Visual_generator.compile(optimizer=self.optimizer,loss='binary_crossentropy',metric=['accuracy'])                
		
    def feature_extractor_network(self):
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
        model = Model(inputs = in_image, outputs = out_features)
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
        model = Model(inputs = in_latents, outputs = out_image)
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
        model =  Model(inputs = in_image, outputs = validity)
        return model
    
    def latent_noise_space(batch_size = 64):
        return np.random.normal(0,1,(batch_size,self.latent_dim))
    
    def Generator_loss(srcimg,trgimg):
        L_GAN_g1 = binary_crossentropy(self.IR_generator(srcimg),tf.ones_like(srcimg),from_logits=True)
        L_GAN_g2 = binary_crossentropy(self.Visual_generator(trgimg),tf.ones_like(trgimg),from_logits=True)
        L_Const = mse(self.Visual_f_net(srcimg),self.Visual_f_net(self.Visual_generator(srcimg))
        L_TID = mse(trgimg,self.IR_generator(trgimg))
        return L_GAN_g1 + L_GAN_g2 + L_Const + L_TID
        
        
    def Discriminator_loss(srcimg,trgimg):
        L_GAN_d_1 = binary_crossentropy(srcimg,tf.zeros_like(srcimg),from_logits=True)
        L_GAN_d_2 = binary_crossentropy(trgimg,tf.zeros_like(trgimg),from_logits=True)
        L_GAN_d_3 = binary_crossentropy(self.IR_discriminator(trgimg),tf.ones_like(trgimg),from_logits=True)
        return L_GAN_d_1 + L_GAN_d_2 + L_GAN_d_1
    
    def train(self,epoch,batch_size=64,training_set_vis,training_set_ir):        
        for epoch in range(epoch):                
            with tf.GradientTape(persistant=True) as g_tape, tf.GradientTape(persistant=True) as d_tape:
                G_loss = self.Generator_loss(training_set_vis,training_set_ir)
                D_loss = self.Discriminator_loss(training_set_vis,training_set_ir)
                
                g_gradient = g_tape.gradient(target=G_loss,source=self.Visual_generator.trainable_variables+self.IR_generator.trainable_variables)
                d_gradient = d_tape.gradient(target=D_loss,source=self.Visual_discriminator.trainable_variables+self.IR_discriminator.trainable_variables)
                
                self.g_optimizer.apply_gradients(zip(g_gradient,self.Visual_generator.trainable_variables+self.IR_generator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(d_gradient,self.Visual_discriminator.trainable_variables+self.IR_discriminator.trainable_variables)
            

                
            
        
        
        
        
	
	
	




