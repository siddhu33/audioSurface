import argparse
from operator import length_hint
import numpy as np
import scipy.io.wavfile as scio
import matplotlib
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


class SongSurface:
    def __init__(self, length, samples):
        self.x = np.ndarray((length, samples), np.float64)
        self.y = np.ndarray((length, samples), np.float64)
        self.z = np.ndarray((length, samples), np.float64)
        self.length = length
        self.samples = samples
        self.count = 0

    def append_buf(self, x, y, z):
        idx = self.count % self.length
        self.x[idx] = x
        self.y[idx] = y
        self.z[idx] = z
        self.count += 1

    def get_buf(self):
        min_size = min(self.count, self.length)
        x0 = self.x[:min_size].reshape(-1)
        y0 = self.y[:min_size].reshape(-1)
        z0 = self.z[:min_size].reshape(-1)
        return x0, y0, z0


def getFrames(fnum, samples, rate, samp, interval, song_surface, ax, plot):
    offset = int((interval / 1000.0) * rate)
    sl0 = samples[offset * fnum : offset * fnum + samp, 0]
    sl0 = np.multiply(np.hamming(len(sl0)), sl0)
    freq0 = abs(np.fft.fft(sl0))
    # dsample0 = [np.mean(j) for j in np.split(freq0,samp/1)]
    r = np.arange(-rate / 2, rate / 2, rate / len(freq0))[0 : len(freq0)]
    song_surface.append_buf(
        r,
        [20 * np.log10(i) for i in freq0],
        [(interval / 1000.0) * fnum for j in freq0],
    )

    ax.clear()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    ax.set_zlim(-10.0, 80, 0)
    ax.set_zlabel("Volume (dB)")

    # merge ndarray into lists for plot
    x0 = song_surface.x.reshape(-1)
    y0 = song_surface.y.reshape(-1)
    z0 = song_surface.z.reshape(-1)
    plot = ax.plot_trisurf(x0, y0, z0, cmap=cm.jet, linewidth=0.2)
    return (
        song_surface,
        plot,
    )

def run_plot(file_name, memory, interval):
    # Once filename is correctly obtained, obtain samples and sample rate.
    rate, samples = scio.read(file_name)
    song_surface = np.shape(samples)
    print(f"Sample Rate: {rate}Hz, Samples : {song_surface}", file=sys.stderr)
    r = np.arange(0, song_surface[0], rate)
    # samp is the number of samples in the analysis frame. Has to be smaller than rate.
    samp = 512
    bins = 1
    count = 0
    fig = plt.figure(1)
    fig.set_size_inches(14.4, 10.8)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    ax.set_zlim(-10.0, 80, 0)
    ax.set_zlabel("Volume (dB)")
    i = 0
    sl0 = samples[i : i + samp, 0]
    sl0 = np.multiply(np.hamming(len(sl0)), sl0)
    freq0 = abs(np.fft.fft(sl0))
    if bins > 1:
        freq0 = [np.mean(j) for j in np.split(freq0, samp / bins)]
    x0 = np.arange(-rate / 2, rate / 2, rate / len(freq0))[0 : len(freq0)]
    z0 = [20 * np.log10(j) for j in freq0]
    y0 = [count for j in freq0]
    y1 = [count + 0.01 for j in freq0]
    song_surface = SongSurface(memory, samp)
    song_surface.append_buf(x0, y0, z0)
    song_surface.append_buf(x0, y1, z0)
    xo, yo, zo = song_surface.get_buf()
    plot = ax.plot_trisurf(xo, yo, zo, cmap=cm.jet, linewidth=0.2)
    # pygame.mixer.init()
    # pygame.mixer.music.load(fname)
    # pygame.mixer.music.play()
    plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
    mywriter = animation.FFMpegWriter(fps=30, bitrate=5000)
    ani = animation.FuncAnimation(
        fig,
        getFrames,
        fargs=(samples, rate, samp, interval, song_surface, ax, plot),
        interval=interval,
        save_count=50,
        blit=False,
    )
    plt.show()



def main():
    # define key parameters
    memory = 5
    interval = 60
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", help="file name to convert into a surface plot. Should end in .wav"
    )
    args = parser.parse_args()
    if args.file_name and args.file_name.endswith(".wav"):
        run_plot(args.file_name, memory, interval)
    else:
        print(f"File name: {args.file_name} invalid!", file=sys.stderr)


if __name__ == "__main__":
    main()
