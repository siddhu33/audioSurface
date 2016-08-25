import numpy as np
import scipy.io.wavfile as scio
import matplotlib
matplotlib.use('QT4Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
import sys
import sounddevice as sd
import pygame
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools
from collections import deque

class songSurface:

    # Use ``__slots__`` for tuple-like access speed.
    __slots__ = ['x', 'y', 'z', 'count']    
    
    def __init__(self,length):
        self.x = deque(maxlen=length)
        self.y = deque(maxlen=length)
        self.z = deque(maxlen=length)
        self.count = 0

def getFrames(fnum,samples,rate,samp,memory,interval,s,ax,plot):
    offset = int((interval/1000.0)*rate)
    sl0 = samples[offset*fnum:offset*fnum+samp,0]
    sl0 = np.multiply(np.hamming(len(sl0)),sl0)
    freq0 = (abs(np.fft.fft(sl0)))
    #dsample0 = [np.mean(j) for j in np.split(freq0,samp/1)]
    r = np.arange(-rate/2,rate/2,rate/len(freq0))[0:len(freq0)]
    s.x.append(r)
    s.z.append([20*np.log10(i) for i in freq0])
    s.y.append([(interval/1000.0) * fnum for j in freq0])
    s.count = s.count + 1
    
    ax.clear()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (s)')
    ax.set_zlim(-10.0,80,0)
    ax.set_zlabel('Volume (dB)')
    
    #merge deque into lists.
    x0 = list(itertools.chain.from_iterable(s.x))
    y0 = list(itertools.chain.from_iterable(s.y))
    z0 = list(itertools.chain.from_iterable(s.z))
    plot = ax.plot_trisurf(x0,y0,z0, cmap=cm.jet, linewidth=0.2)
    return s,plot,

def main():
    #define key parameters
    memory = 5
    interval = 60
    if len(sys.argv) < 2:
        print("USAGE - needs an input file argument")
        return
    fname = sys.argv[1]
    print(fname)
    if not fname.lower().endswith('.wav'):
        print("file needs to be of a .wav format")
        return
    
    #Once filename is correctly obtained, obtain samples and sample rate.
    rate, samples = scio.read(fname)
    s = np.shape(samples)
    print("Sample Rate: {0}Hz, Samples : {1}".format(rate,s))
    r = np.arange(0,s[0],rate)
    #samp is the number of samples in the analysis frame. Has to be smaller than rate.
    samp = 512
    bins = 1
    count = 0
    
    fig = plt.figure(1)
    fig.set_size_inches(14.4, 10.8)
    ax = Axes3D(fig)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (s)')
    ax.set_zlim(-10.0,80,0)
    ax.set_zlabel('Volume (dB)')
    
    i = 0
    sl0 = samples[i:i+samp,0]
    sl0 = np.multiply(np.hamming(len(sl0)),sl0)
    freq0 = abs(np.fft.fft(sl0))
    if bins > 1:
        freq0 = [np.mean(j) for j in np.split(freq0,samp/bins)]
    x0 = np.arange(-rate/2,rate/2,rate/len(freq0))[0:len(freq0)]
    z0 = [20*np.log10(j) for j in freq0]
    y0 = [count for j in freq0]
    y1 = [count+0.01 for j in freq0]
    x0 = np.append(x0,x0)
    y0 = np.append(y0,y1)
    z0 = np.append(z0,z0)
    s = songSurface(memory)
    s.x.append(x0)
    s.y.append(y0)
    s.z.append(z0)
    plot = ax.plot_trisurf(x0, y0, z0, cmap=cm.jet, linewidth=0.2)
    #pygame.mixer.init()
    #pygame.mixer.music.load(fname)
    #pygame.mixer.music.play()
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    mywriter = animation.FFMpegWriter(fps=15, bitrate=5000)
    ani = animation.FuncAnimation(fig, getFrames,fargs=(samples,rate,samp,memory,interval,s,ax,plot),
                              interval=interval, blit=False)
    plt.show()
    #ani.save("video.mp4",mywriter)
main()
