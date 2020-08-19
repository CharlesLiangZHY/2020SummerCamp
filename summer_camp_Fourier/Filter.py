import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import pyaudio
import time

def play(filename):
    chunk = 1024  
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
    data = wf.readframes(chunk) 
    while data != b'':  
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream() 
    stream.close()
    p.terminate() 


print("Now play the original signal.")
play('./audio/noised.wav')
sameple_rate,sigs = wf.read('./audio/noised.wav')
sigs = sigs/(2**15)
times = np.arange(len(sigs))/sameple_rate
plt.subplot(221)
plt.title('Time Domain',fontsize=16)
plt.ylabel('Signal',fontsize=12)
plt.grid(linestyle=':')
plt.plot(times[:200],sigs[:200],color='dodgerblue',label='Noised Signal')
plt.legend()
freqs = nf.fftfreq(sigs.size, 1/sameple_rate)
complex_arry = nf.fft(sigs)
pows = np.abs(complex_arry)
plt.subplot(222)
plt.title('Frequence Domain',fontsize=16)
plt.grid(linestyle=':')
plt.semilogy(freqs[freqs>0],pows[freqs>0],color='dodgerblue',label='Noised Freq')
plt.legend()
fun_freq = freqs[pows.argmax()] 
noised_idx = np.where(freqs != fun_freq)[0]
ca = complex_arry[:]
ca[noised_idx] = 0 
filter_pows = np.abs(complex_arry)
filter_sigs = nf.ifft(ca).real
plt.subplot(223)
plt.title('Time Domain',fontsize=16)
plt.ylabel('Signal',fontsize=12)
plt.grid(linestyle=':')
plt.plot(times[:200],filter_sigs[:200],color='dodgerblue',label='Filter Signal')
plt.legend()
plt.subplot(224)
plt.title('Frequence Domain',fontsize=16)
plt.grid(linestyle=':')
plt.plot(freqs[freqs>0],filter_pows[freqs>0],color='dodgerblue',label='Filter Freq')
plt.legend()
filter_sigs = (filter_sigs*(2**15)).astype('i2')
wf.write('./audio/filter.wav',sameple_rate,filter_sigs)
time.sleep(1.5)
print("Now play the signal after filtering.")
play('./audio/filter.wav')
plt.tight_layout()
plt.show()