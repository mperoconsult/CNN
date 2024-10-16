

#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=Sm54KXD-L1k

################################
# SENOIDE WAVE
################################

#from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

plt.close('all')

# 20 % test
pTest = 0.2

#df = pd.read_csv(r'D:\work\quadrado.csv')
#dfNoise = pd.read_csv(r'D:\work\quadradoNoise.csv')
#x_total= df.to_numpy()
#x_totalNoise = dfNoise.to_numpy()

x_total = np.loadtxt(r'D:\work\Seno.csv',delimiter=",", dtype=float)
x_totalNoise =  np.loadtxt(r'D:\work\SenoNoise.csv',delimiter=",", dtype=float)

lin, col = x_total.shape
linN, colN = x_totalNoise.shape



x_test = x_total[0:round(pTest*lin), :]
x_train  = x_total[1+round(pTest*lin):,]

x_testN = x_totalNoise[0:round(pTest*lin), :]
x_trainN  = x_totalNoise[1+round(pTest*lin):,]


#(x_train, _), (x_test, _) = mnist.load_data()
########## Joga valores igual ou acima de zero
x_train = x_train.astype('float32') - x_train.min()
x_test = x_test.astype('float32') - x_test.min()

x_trainN = x_trainN.astype('float32') - x_trainN.min()
x_testN = x_testN.astype('float32') - x_testN.min()
########## normaliza sinais entre zero e um
x_train =x_train.astype('float32') / x_train.max()
x_test = x_test.astype('float32') / x_test.max()

x_trainN =x_trainN.astype('float32') / x_trainN.max()
x_testN = x_testN.astype('float32') / x_testN.max()


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_trainN = np.reshape(x_trainN, (x_trainN.shape[0], x_trainN.shape[1], 1))
x_testN = np.reshape(x_testN, (x_testN.shape[0], x_testN.shape[1], 1))



# testa a visualizacao desses vetores
# plt.plot(x_train[1,:,],'r')
# plt.plot(x_trainN[1,:,],'g')
# plt.show()



#adding some noise
#noise_factor = 0.5
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#Displaying images with noise
#plt.figure(figsize=(20, 2))
plt.figure(1)
for i in range(1,10):
    ax = plt.subplot(1, 10, i)
    plt.plot(x_testN[i])
plt.show()



# Marcelo: aqui ele comeca a criar a rede
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(col,1)))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(8, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(8, 3, activation='relu', padding='same'))

model.add(MaxPooling1D(2, padding='same'))
 
model.add(Conv1D(8, 3, activation='relu', padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(8, 3, activation='relu', padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(32, 3, activation='relu', padding='same'))
model.add(UpSampling1D(2))
model.add(Conv1D(1, 3, activation='relu', padding='same'))


# Rede Original
#model = Sequential()
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 
#model.add(MaxPooling2D((2, 2), padding='same'))
 
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

history =model.fit(x_trainN, x_train, epochs=50, batch_size=256, shuffle=True, 
          validation_data=(x_testN, x_test),verbose=1)


model.evaluate(x_testN, x_test)

model.save('denoising_autoencoderSeno.model')

saida = model.predict(x_testN)

plt.figure(2)
num_plots=10
for i in range(num_plots):
    print(i)
    plt.subplot(num_plots,1,i+1)  
    plt.plot(x_testN[i,:,0],'r')
    plt.plot(saida[i,:,0],'g')

plt.show()
plt.savefig("D:\Artigos\CNN\RITA2018\WaveformsSeno.svg")

# para ver os campos (keys) do history callback:
print(history.history.keys())

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'],'r')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.subplot(2,1,2)
plt.plot(history.history['loss'],'g')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['acc', 'loss'], loc='upper left')
plt.show()
plt.savefig("D:\Artigos\CNN\RITA2018\HistoryKeysSeno.svg")


# computa SNR aqui ENTRADA
numerador = np.square(x_testN)
numerador=np.sum(numerador,1)
#denominador = np.square(x_testN) # aqui acho que a relacao eh com x_test, que eh o limpo original
denominador = np.square(x_test)
denominador=np.sum(denominador,1)
SNRInput=  np.divide(numerador,denominador)
SNRInput=10*np.log10(SNRInput)

plt.figure(4)
plt.hist(SNRInput, bins=15, color = 'red')
#plt.show()
#plt.savefig("D:\Artigos\CNN\RITA2018\HistogramaInputRetangular.svg")

# computa SNR aqui SAIDA
numerador = np.square(saida)
numerador=np.sum(numerador,1)
#denominador = np.square(x_testN) # aqui acho que a relacao eh com x_test, que eh o limpo original
denominador = np.square(x_test)
denominador=np.sum(denominador,1)
SNROutput =  np.divide(numerador,denominador)
SNROutput=10*np.log10(SNROutput)

#plt.figure(4)
plt.hist(SNROutput, bins=15, color = 'green')
plt.xlabel("SNR ")
plt.ylabel("Frequency ")
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Input", "Output"], loc=0, frameon=legend_drawn_flag)
plt.show()
plt.savefig("D:\Artigos\CNN\RITA2018\HistogramaSeno.svg")




end_time = time.time()

execution_time = end_time - start_time
