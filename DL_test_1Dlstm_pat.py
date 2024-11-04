import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Multiply, Subtract, concatenate, Dropout, Add, Activation, Average, GaussianNoise, LeakyReLU, BatchNormalization, GaussianDropout, Lambda, Reshape, LSTM, Dense, GRU, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import ConvLSTM1D, Conv1DTranspose, MaxPooling1D, UpSampling1D,AveragePooling1D,ReLU,Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from scipy import signal
import numpy as np
import os
import scipy   
import h5py
import scipy.io
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import mat73

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class prediction_history(Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        np.save('output/DL_fit_{}_unet.npy'.format(epoch),new_autoencoder.predict(CTP,batch_size=2233))

def custom_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.math.abs(y_pred[:,:,0] - y_true[:,:,0]))
    return loss
        
def custom_function(x,t):
    x=x*0.02
    Pred_CTP=x[:,0:t,:]*AIF[:,72-t:72,:]
    Pred_CTP=tf.math.reduce_sum(Pred_CTP,1)
    return Pred_CTP*0.5

images1 = mat73.loadmat('Pat_data1/Pat92_CTP_1D_norm.mat')
CTP = images1['Vn']
CTP = np.expand_dims(CTP, axis=2)

images2 = mat73.loadmat('Pat_data1/Pat92_AIF_1D_norm.mat')
AIF = images2['AIF']
AIF = np.expand_dims(AIF, axis=0)
AIF = np.expand_dims(AIF, axis=2)
AIF = tf.convert_to_tensor(AIF)
AIF = tf.cast(AIF,dtype=tf.float32)
AIF = tf.reverse(AIF,[1])

input_img = Input(shape=(72,1))

c0 = Conv1D(32, 3,activation="relu",data_format="channels_last",padding="same")(input_img)
c0 = Conv1D(32, 5,activation="relu",data_format="channels_last",padding="same")(c0)
c0 = Conv1D(32, 7,activation="relu",data_format="channels_last",padding="same")(c0)

c1 = Bidirectional(LSTM(32,return_sequences="True"))(c0)
c2 = Bidirectional(LSTM(32,return_sequences="True"))(c1)
c3 = Bidirectional(LSTM(32,return_sequences="True"))(c2)
c4 = Bidirectional(LSTM(32,return_sequences="False"))(c3)
c5 = SeqSelfAttention(attention_width=15,use_attention_bias=False,attention_activation=None,attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(c4)
c6 = Dense(1)(c5)
c10 = ReLU(max_value=1.0)(c6)

p1 = Lambda(custom_function,output_shape=(1,1),arguments={'t':1})(c10)
p2 = Lambda(custom_function,output_shape=(1,1),arguments={'t':2})(c10)
p3 = Lambda(custom_function,output_shape=(1,1),arguments={'t':3})(c10)
p4 = Lambda(custom_function,output_shape=(1,1),arguments={'t':4})(c10)
p5 = Lambda(custom_function,output_shape=(1,1),arguments={'t':5})(c10)
p6 = Lambda(custom_function,output_shape=(1,1),arguments={'t':6})(c10)
p7 = Lambda(custom_function,output_shape=(1,1),arguments={'t':7})(c10)
p8 = Lambda(custom_function,output_shape=(1,1),arguments={'t':8})(c10)
p9 = Lambda(custom_function,output_shape=(1,1),arguments={'t':9})(c10)
p10 = Lambda(custom_function,output_shape=(1,1),arguments={'t':10})(c10)
p11 = Lambda(custom_function,output_shape=(1,1),arguments={'t':11})(c10)
p12 = Lambda(custom_function,output_shape=(1,1),arguments={'t':12})(c10)
p13 = Lambda(custom_function,output_shape=(1,1),arguments={'t':13})(c10)
p14 = Lambda(custom_function,output_shape=(1,1),arguments={'t':14})(c10)
p15 = Lambda(custom_function,output_shape=(1,1),arguments={'t':15})(c10)
p16 = Lambda(custom_function,output_shape=(1,1),arguments={'t':16})(c10)
p17 = Lambda(custom_function,output_shape=(1,1),arguments={'t':17})(c10)
p18 = Lambda(custom_function,output_shape=(1,1),arguments={'t':18})(c10)
p19 = Lambda(custom_function,output_shape=(1,1),arguments={'t':19})(c10)
p20 = Lambda(custom_function,output_shape=(1,1),arguments={'t':20})(c10)
p21 = Lambda(custom_function,output_shape=(1,1),arguments={'t':21})(c10)
p22 = Lambda(custom_function,output_shape=(1,1),arguments={'t':22})(c10)
p23 = Lambda(custom_function,output_shape=(1,1),arguments={'t':23})(c10)
p24 = Lambda(custom_function,output_shape=(1,1),arguments={'t':24})(c10)
p25 = Lambda(custom_function,output_shape=(1,1),arguments={'t':25})(c10)
p26 = Lambda(custom_function,output_shape=(1,1),arguments={'t':26})(c10)
p27 = Lambda(custom_function,output_shape=(1,1),arguments={'t':27})(c10)
p28 = Lambda(custom_function,output_shape=(1,1),arguments={'t':28})(c10)
p29 = Lambda(custom_function,output_shape=(1,1),arguments={'t':29})(c10)
p30 = Lambda(custom_function,output_shape=(1,1),arguments={'t':30})(c10)
p31 = Lambda(custom_function,output_shape=(1,1),arguments={'t':31})(c10)
p32 = Lambda(custom_function,output_shape=(1,1),arguments={'t':32})(c10)
p33 = Lambda(custom_function,output_shape=(1,1),arguments={'t':33})(c10)
p34 = Lambda(custom_function,output_shape=(1,1),arguments={'t':34})(c10)
p35 = Lambda(custom_function,output_shape=(1,1),arguments={'t':35})(c10)
p36 = Lambda(custom_function,output_shape=(1,1),arguments={'t':36})(c10)
p37 = Lambda(custom_function,output_shape=(1,1),arguments={'t':37})(c10)
p38 = Lambda(custom_function,output_shape=(1,1),arguments={'t':38})(c10)
p39 = Lambda(custom_function,output_shape=(1,1),arguments={'t':39})(c10)
p40 = Lambda(custom_function,output_shape=(1,1),arguments={'t':40})(c10)
p41 = Lambda(custom_function,output_shape=(1,1),arguments={'t':41})(c10)
p42 = Lambda(custom_function,output_shape=(1,1),arguments={'t':42})(c10)
p43 = Lambda(custom_function,output_shape=(1,1),arguments={'t':43})(c10)
p44 = Lambda(custom_function,output_shape=(1,1),arguments={'t':44})(c10)
p45 = Lambda(custom_function,output_shape=(1,1),arguments={'t':45})(c10)
p46 = Lambda(custom_function,output_shape=(1,1),arguments={'t':46})(c10)
p47 = Lambda(custom_function,output_shape=(1,1),arguments={'t':47})(c10)
p48 = Lambda(custom_function,output_shape=(1,1),arguments={'t':48})(c10)
p49 = Lambda(custom_function,output_shape=(1,1),arguments={'t':49})(c10)
p50 = Lambda(custom_function,output_shape=(1,1),arguments={'t':50})(c10)
p51 = Lambda(custom_function,output_shape=(1,1),arguments={'t':51})(c10)
p52 = Lambda(custom_function,output_shape=(1,1),arguments={'t':52})(c10)
p53 = Lambda(custom_function,output_shape=(1,1),arguments={'t':53})(c10)
p54 = Lambda(custom_function,output_shape=(1,1),arguments={'t':54})(c10)
p55 = Lambda(custom_function,output_shape=(1,1),arguments={'t':55})(c10)
p56 = Lambda(custom_function,output_shape=(1,1),arguments={'t':56})(c10)
p57 = Lambda(custom_function,output_shape=(1,1),arguments={'t':57})(c10)
p58 = Lambda(custom_function,output_shape=(1,1),arguments={'t':58})(c10)
p59 = Lambda(custom_function,output_shape=(1,1),arguments={'t':59})(c10)
p60 = Lambda(custom_function,output_shape=(1,1),arguments={'t':60})(c10)
p61 = Lambda(custom_function,output_shape=(1,1),arguments={'t':61})(c10)
p62 = Lambda(custom_function,output_shape=(1,1),arguments={'t':62})(c10)
p63 = Lambda(custom_function,output_shape=(1,1),arguments={'t':63})(c10)
p64 = Lambda(custom_function,output_shape=(1,1),arguments={'t':64})(c10)
p65 = Lambda(custom_function,output_shape=(1,1),arguments={'t':65})(c10)
p66 = Lambda(custom_function,output_shape=(1,1),arguments={'t':66})(c10)
p67 = Lambda(custom_function,output_shape=(1,1),arguments={'t':67})(c10)
p68 = Lambda(custom_function,output_shape=(1,1),arguments={'t':68})(c10)
p69 = Lambda(custom_function,output_shape=(1,1),arguments={'t':69})(c10)
p70 = Lambda(custom_function,output_shape=(1,1),arguments={'t':70})(c10)
p71 = Lambda(custom_function,output_shape=(1,1),arguments={'t':71})(c10)
p72 = Lambda(custom_function,output_shape=(1,1),arguments={'t':72})(c10)


p = concatenate([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,p50,p51,p52,p53,p54,p55,p56,p57,p58,p59,p60,p61,p62,p63,p64,p65,p66,p67,p68,p69,p70,p71,p72],axis=1)

pred_ctp = Reshape((72,1))(p)

autoencoder = Model(input_img,pred_ctp)

autoencoder.summary()

new_autoencoder = Model(inputs=autoencoder.input,outputs=autoencoder.get_layer("re_lu").output)

adam = Adam(learning_rate=1e-2)

autoencoder.compile(optimizer='adam',loss=custom_loss)

predictions=prediction_history()

autoencoder.fit(CTP,CTP,epochs=100,batch_size=8700,shuffle=True,callbacks=[predictions])

del autoencoder


