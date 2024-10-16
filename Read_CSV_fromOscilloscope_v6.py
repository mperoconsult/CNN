# -*- coding: utf-8 -*-
"""
Created on Mon May 22 06:35:09 2023

@author: m_per
"""

# DO JEITO QUE TA TEM UMA CURVA QUE ATE DA PRA USAR. PASSE PARA
# V2 PARA CONTINUAR TESTES

from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import load_model

plt.close('all')
#comeca com 90,91, 92 e depois 98 ate 111
# 93 ruim
dados = np.loadtxt(r'D:/Artigos/CNN/MedidasRuidoZener_10Fev23/scope_105.csv',delimiter=",",
                   skiprows=2, dtype=float)

dados= np.float32(dados)

#faz comecar em zero
dados[:,0]= dados[:,0] - min(dados[:,0])
tempo =  (dados[0:1000,0])

# tem que ter 1000 elementos -- tem que normalizar entre 0 e 1
dados1=(dados[0:1000,1]) #- (dados[0:1000,1]).min()
dados2=(dados[0:1000,2]) #- (dados[0:1000,2]).min()



# A CHAVE ESTA NESSES OFFSETS, ELES PRECISAM SER AJUSTADOS PARA TIPO CIMA DEIXANDO 
# SEM FICA RUIM
dados1 = dados1 - dados1.min() - 0.5
dados2 = dados2 - dados2.min() - 0.5

dados1max = np.amax(dados1)
dados2max = np.amax(dados2)

dados1 = dados1 / np.amax(dados1) 
dados2 = dados2 / np.amax(dados2)



# dados1 = np.clip(dados1, -.25,1)
# dados2 = np.clip(dados2, -.25,1)


# dados1 = dados1/ (dados1.min())
# dados2 = dados2/ (dados2.min())



plt.figure(1)
plt.plot(tempo, dados1, '-r', linewidth=2)
plt.plot(tempo, dados2, '-g', linewidth=2)
plt.show()


#precisa fazer o reshape para ficar consistente com o CNN
dados1= dados1.reshape(1,1000,1)
dados2= dados2.reshape(1,1000,1)


loaded_model = load_model('denoising_autoencoder.model') # RECTANGULAR
#loaded_model = load_model('denoising_autoencoderTri.model') # TRIANGULAR
#loaded_model = load_model('denoising_autoencoderTodos.model') # TODOS
loaded_model = load_model('denoising_autoencoderSeno.model') # SENO
loaded_model.summary()

saida1 = loaded_model.predict(dados1)
saida2 = loaded_model.predict(dados2)

# reshape noch wieder diese sheisse
dados1 = dados1.reshape(1000,)
dados2 = dados2.reshape(1000,)
saida1 = saida1.reshape(1000,)
saida2 = saida2.reshape(1000,)


plt.figure(2)
plt.plot(tempo/1e-3, dados1,'-g',linewidth=2)
plt.plot(tempo/1e-3, saida1, '-r', linewidth=2)
plt.xlabel("Time [ms] ")
plt.ylabel("Amplitude")
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Input", "Output"], loc=0, frameon=legend_drawn_flag)
plt.show()
# plt.savefig("D:\Artigos\CNN\RITA2018\Oscilosc_105_col2.svg")


samp_rate = 1/(tempo[2]-tempo[1])
Fdados1 = fft.fft(dados1)/np.size(dados1)
Fsaida1 = fft.fft(saida1)/np.size(saida1)

Fdados1 =  Fdados1[range(int(len(dados1)/2))]
Fsaida1 =  Fsaida1[range(int(len(saida1)/2))]

tpCount     = len(dados1)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samp_rate
frequencies = values/timePeriod

Fsaida1dB = 10*np.log10(np.absolute(Fsaida1)**2) - max(10*np.log10(np.absolute(Fsaida1)**2))
Fdados1dB = 10*np.log10(np.absolute(Fdados1)**2) - max(10*np.log10(np.absolute(Fdados1)**2))
plt.figure(22)
plt.plot(frequencies/1e3, Fdados1dB,'g')
plt.plot(frequencies/1e3, Fsaida1dB,'r')

plt.xlabel("Frequency [kHz] ")
plt.ylabel("Normalized Power Spectrum [dB]")
# plt.savefig("D:\Artigos\CNN\RITA2018\Oscilosc_105_col2FFT.svg")



plt.figure(3)
plt.plot(tempo/1e-3, dados2,'-g',linewidth=2)
plt.plot(tempo/1e-3, saida2, '-r', linewidth=2)
plt.xlabel("Time [ms] ")
plt.ylabel("Amplitude")
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Input", "Output"], loc=0, frameon=legend_drawn_flag)
plt.show()
# plt.savefig("D:\Artigos\CNN\RITA2018\Oscilosc_105_col3.svg")


Fdados2 = fft.fft(dados2)/np.size(dados2)
Fsaida2 = fft.fft(saida2)/np.size(saida2)

Fdados2 =  Fdados1[range(int(len(dados2)/2))]
Fsaida2 =  Fsaida1[range(int(len(saida2)/2))]

tpCount     = len(dados2)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/samp_rate
frequencies = values/timePeriod

Fsaida2dB = 10*np.log10(np.absolute(Fsaida2)**2) - max(10*np.log10(np.absolute(Fsaida2)**2))
Fdados2dB = 10*np.log10(np.absolute(Fdados2)**2) - max(10*np.log10(np.absolute(Fdados2)**2))
plt.figure(33)
plt.plot(frequencies/1e3, Fdados2dB,'g')
plt.plot(frequencies/1e3, Fsaida2dB,'r')

plt.xlabel("Frequency [kHz] ")
plt.ylabel("Normalized Power Spectrum [dB]")
# plt.savefig("D:\Artigos\CNN\RITA2018\Oscilosc_105_col3FFT.svg")