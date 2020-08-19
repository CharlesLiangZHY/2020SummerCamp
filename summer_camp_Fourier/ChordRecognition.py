from A440_Standard import FreqDict
from A440_Standard import FreqList
import wave
import pyaudio
import scipy.stats
from scipy import signal
from scipy.fftpack import fft,ifft
import pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np
import math
import sys
import warnings
warnings.filterwarnings('ignore')

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

def detect_frequency(sig, chunkSize, num=1):
    sampleWidth= sig.getsampwidth()
    frameRate = sig.getframerate() 
    frequencies = []
    frequenciesInt = []
    frequency = -1

    window = np.blackman(chunkSize*2) # Window should be double chunk size because values are interpolated.
    chunk = sig.readframes(chunkSize) # Break the data into pieces in line with the window.
    # Go through the signal chunk-by-chunk.
    while (len(chunk) > chunkSize * sampleWidth) and (len(chunk) % chunkSize * sampleWidth == 0):
        data = np.array(wave.struct.unpack('%dh' % (len(chunk)/sampleWidth), chunk))*window
        # Take the square of the real fft because we only care about reals.
        fftVals = abs(np.fft.rfft(data))**2
        # Find the peak and then use quadratic interpolation to pull out the frequency unless we're just pulling out an endpiece.
        maximum = fftVals[1:].argmax() + 1
        if maximum != len(fftVals) - 1:
            a, b, c = np.log(fftVals[maximum-1:maximum+2:])
            interp = (c - a) * .5 / (2 * b - c - a)
            frequency = ((maximum + interp) * frameRate) / (chunkSize * 2)
        else:
            frequency = (maximum * frameRate) / chunkSize       
        if frequency > 0 and frequency < 20000: # Frequencies that human can hear.
            frequencies.append(frequency)
            frequenciesInt.append(int(round(frequency)))
        chunk = sig.readframes(chunkSize) # Go to next chunk.
    chord = []
    for i in range(num*num):
        frequency = scipy.stats.mode(frequenciesInt) # grab the most common of the frequencies.
        chord.append(frequency[0])
        frequenciesInt = [f for f in frequenciesInt if f != chord[i]]
    return chord

def specify_chord(chord, n):
    cur = 0
    notes = []
    find_closest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
    for f in chord:
        closestFrequency = find_closest(f, FreqList)
        note = FreqDict[closestFrequency]
        if note in notes:
            continue
        else:
            notes.append(note)
            cur += 1
            if cur == n:
                break
    return notes

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python NoteDectection.py -f filename -n NumOfNotes E.g. python NoteDectection.py -f DFA3.wav -n 3")
    else:
        filename = None
        n = 3
        plot = False
        p3d = False
        for i in range(len(sys.argv)):
            if sys.argv[i] == '-f':
                filename = sys.argv[i+1]
            elif sys.argv[i] == '-n':
                n = int(sys.argv[i+1])
            elif sys.argv[i] == '-p':
                plot = True
            elif sys.argv[i] == '-3d':
                p3d = True
        sig = wave.open(filename, 'rb')
        samplingChunk = 2048
        chord = detect_frequency(sig, samplingChunk, n)
        notes = specify_chord(chord, n)
        sig.close() # Read over. 
        n = len(notes)
        print("Detected %d notes in the chord." %(n))
        for note in notes:
            freq = list(FreqDict.keys())[list(FreqDict.values()).index(note)]
            print("%s with frequency: %f" %(note, freq))
        print("Now play the chord.")
        play(filename)
        if plot:
            sig = wave.open(filename,"rb")
            params = sig.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            strData  = sig.readframes(nframes)
            sig.close()
            wave_data = np.fromstring(strData,dtype=np.int16)
            wave_data = wave_data*1.0 / (max(abs(wave_data)))
            time=np.arange(0,nframes)/framerate
            xs = wave_data [:nframes]
            xf= np.fft.fft(xs)
            freqs = np.fft.fftfreq(nframes,1.0/framerate)
            plt.figure(figsize=(5,8))
            plt.subplot(211)
            plt.plot(time/2, xs)
            plt.ylabel('Signal')
            plt.xlabel('Time [sec]')
            plt.subplot(212)
            fx = freqs[freqs>0]
            fy = np.abs(xf)[freqs>0]
            maxfreq = 0
            for n in notes:
                temp = list(FreqDict.keys())[list(FreqDict.values()).index(n)]
                if maxfreq < list(FreqDict.keys())[list(FreqDict.values()).index(n)]:
                    maxfreq = temp
            ceil = math.ceil(maxfreq) // 1000  + 1000
            plt.plot(fx[fx < ceil],fy[fx < ceil]) 
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency [hz]')
            plt.figure()
            f, t, Sxx = signal.spectrogram(wave_data, framerate)
            plt.pcolormesh(t/2, f, Sxx, shading='gouraud')
            plt.ylim(0,ceil)
            plt.ylabel('log of Frequency')
            plt.xlabel('Time [sec]')
            if p3d:
                fig = plt.figure()
                axes3d = Axes3D(fig)
                X,Y = np.meshgrid(t/2,f)
                axes3d.plot_surface(X,Y,Sxx, cmap=cm.coolwarm, linewidth=0)
                axes3d.set_ylim3d(0, ceil)
            plt.show()
        sig.close()
