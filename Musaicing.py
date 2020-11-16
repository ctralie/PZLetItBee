"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for Driedger's Musaicing Technique

Modified by Alex Russo Nov 7 2020
Purpose: Add stereo processing and ability to create and use corpus as source
"""
import argparse
import os
import sndhdr
import wave
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage
from pydub import AudioSegment

from NMF import *
from SpectrogramTools import *


def doMusaicing(source, sourceCorpus, target, result, mono, sr = 22050, winSize = 2048, hopSize = 1024, \
                    NIters = 80, r = 7, p = 10, c = 3, savePlots = True):
    """
    :param source: Source audio filename
    :param sourceCorpus: Comma separated list of source audio filenames to be used as a corpus
    :param target: Target audio filename
    :param result: Result wavfile path
    :param winSize: Window Size of STFT in samples
    :param hopSize: Hop size of STFT in samples
    :param NIters: Number of iterations of Driedger's technique
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony; i.e. number of values in each column\
        of H which should be un-shrunken
    :param c: Half length of time-continuous activation filter
    :param savePlots: Whether to save plots showing progress of NMF \
        every 20 iterations
    :param mono: If True the unmodified LetItBee source will be used \
        and a mono file will be returned but you can still use sourceCorpus feature
    """
    if sourceCorpus:
        combined = AudioSegment.empty()
        if os.path.isdir(sourceCorpus):
            directory = sourceCorpus
            for filename in os.listdir(directory):
                filename = os.path.join(directory, filename)

                # check if file is a valid audio file, if not don't include in corpus
                if sndhdr.what(filename):
                    audio_file = AudioSegment.from_file(filename)
                    combined += audio_file
                else:
                    continue
        else:
            sourceCorpus = sourceCorpus.strip().split(',')
            for audio_file in sourceCorpus:
                audio_file = AudioSegment.from_file(audio_file)
                combined += audio_file
                
        # process and export corpus
        directory = "./audio_files/corpus/"
        filename = directory + "corpus_" + result
        Path(directory).mkdir(parents=True, exist_ok=True)
        combined.export(filename)
        source = filename

    if mono == "False":
        print(f"Starting using stereo processing, sample rate: {sr}, winSize: {winSize}, hopSize: {hopSize}, NIters: {NIters}, r: {r}, p: {p}, c: {c}, savePlots: {savePlots}")
        # load source, split channels, duplicate if mono, and process data by channel
        X, sr = librosa.load(source, mono=False, sr=sr)
        try:
            print("Starting PitchShift L")
            WComplexL = getPitchShiftedSpecs(X[0], sr, winSize, hopSize, 6)
            print("Starting PitchShift R")
            WComplexR = getPitchShiftedSpecs(X[1], sr, winSize, hopSize, 6)
        except:
            print("Starting PitchShift L MONO")
            WComplexL = getPitchShiftedSpecs(X, sr, winSize, hopSize, 6)
            print("Starting PitchShift R MONO")
            WComplexR = getPitchShiftedSpecs(X, sr, winSize, hopSize, 6)
        WL = np.abs(WComplexL)
        WR = np.abs(WComplexR)

        # load target, split channels, duplicate if mono, and process data by channel
        X, sr = librosa.load(target, mono=False, sr=sr)
        try:
            VL = np.abs(STFT(X[0], winSize, hopSize))
            VR = np.abs(STFT(X[1], winSize, hopSize))
        except:
            VL = np.abs(STFT(X, winSize, hopSize))
            VR = np.abs(STFT(X, winSize, hopSize))
        fn = None
        fnw = None
        if savePlots:
            fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
            fnw = lambda W: plotInitialW(W, hopSize)

        # additional processing per channel
        print("Starting Driedger Channel L")
        HL = doNMFDriedger(VL, WL, NIters, r=r, p=p, c=c, plotfn=fn, plotfnw = fnw)
        print("Starting Driedger Channel R")
        HR = doNMFDriedger(VR, WR, NIters, r=r, p=p, c=c, plotfn=fn, plotfnw = fnw)
        HL = np.array(HL, dtype=np.complex)
        HR = np.array(HR, dtype=np.complex)
        V2L = WComplexL.dot(HL)
        V2R = WComplexR.dot(HR)

        print("Doing phase retrieval L")
        YL = griffinLimInverse(V2L, winSize, hopSize, NIters=30)
        print("Doing phase retrieval R")
        YR = griffinLimInverse(V2R, winSize, hopSize, NIters=30)
        YL = YL/np.max(np.abs(YL))
        YR = YR/np.max(np.abs(YR))
        
        resultL = result[:-4] + "L.wav"
        resultR = result[:-4] + "R.wav"

        # convert from float64 to 32 for final bounce (bit depth can't be higher than 32)
        YL = YL.astype('float32')
        YR = YR.astype('float32')

        # write left and right channels
        sio.wavfile.write(resultL, sr, YL)
        sio.wavfile.write(resultR, sr, YR)

        # combine left and right channels
        finalL = AudioSegment.from_file(resultL)
        finalR = AudioSegment.from_file(resultR)
        finalL.export()
        finalR.export()
        final_stereo_file = AudioSegment.from_mono_audiosegments(finalL, finalR)
        # export processed file
        directory = "./audio_files/processed/"
        result = directory + result
        Path(directory).mkdir(parents=True, exist_ok=True)
        final_stereo_file.export(result, format='wav')
        # remove left and right files
        os.remove(resultL)
        os.remove(resultR)
    else:
        print("Starting using original LetItBee Mono code")
        # if mono = True use original LetItBee mono code
        X, sr = librosa.load(source, sr=sr)
        WComplex = getPitchShiftedSpecs(X, sr, winSize, hopSize, 6)
        W = np.abs(WComplex)
        X, sr = librosa.load(target, sr=sr)
        V = np.abs(STFT(X, winSize, hopSize))
        fn = None
        fnw = None
        if savePlots:
            fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
            fnw = lambda W: plotInitialW(W, hopSize)
        H = doNMFDriedger(V, W, NIters, r=r, p=p, c=c, plotfn=fn, plotfnw = fnw)
        H = np.array(H, dtype=np.complex)
        V2 = WComplex.dot(H)
        print("Doing phase retrieval...")
        Y = griffinLimInverse(V2, winSize, hopSize, NIters=30)
        Y = Y/np.max(np.abs(Y))
        sio.wavfile.write(result, sr, Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=str, help="Path to audio file for source sounds")
    group.add_argument('--sourceCorpus', type=str, help="Comma separated list of paths to audio files for source sounds to be used as a corpus")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--mono', type=str, default="False", help='If mono is True the unmodified LetItBee code will be used and a mono file will be returned')
    parser.add_argument('--sr', type=int, default=22050, help="Sample rate")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--hopSize', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--NIters', type=int, default=60, help="Number of iterations of NMF")
    parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument('--p', type=int, default=10, help="Degree of polyphony; i.e. number of values in each column of H which should be un-shrunken")
    parser.add_argument('--c', type=int, default=3, help="Half length of time-continuous activation filter")
    parser.add_argument('--saveplots', type=int, default=0, help='Save plots of iterations to disk')
    opt = parser.parse_args()
    doMusaicing(opt.source, opt.sourceCorpus, opt.target, opt.result, opt.mono, sr=opt.sr, winSize=opt.winSize, \
                hopSize=opt.hopSize, NIters=opt.NIters, r=opt.r, p=opt.p, c=opt.c, \
                savePlots=opt.saveplots)
