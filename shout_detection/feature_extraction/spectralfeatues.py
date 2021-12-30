import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path  # For writing videos into the data folder
import pandas as pd 
import os

SAMPLE_RATE = 16000
DURATION = 6 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE*DURATION

plt.rcParams['figure.dpi'] = 100
 
#This code is related to the parsel mouth python package. The following is doable in this package
# We can extract pitch, harmonicity
#References: https://www.kaggle.com/ashishpatel26/feature-extraction-from-audio
#https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
#can use librosa for MFCC extraction. I don't think this feature can be easily compared as the words need to be similar for direct comparison. 

# idea: I think lombard has to do with the loudness/intensity of the speaker when in a loud room. 
            # so this could involve a separate dataset that has speakers in a loud room verse in a quiet room. 

#prosody involves the pitch, loudness, AND energy

def analyse_mfcc(filepath):
    '''
    Creates MFCC 
    Input:
        row of dataset
    Output:
        outputs 20 by # array containing the mfcc 
    '''

    x, sr = librosa.load(filepath, sr=41000)
    x = librosa.to_mono(x)
    mfcc = librosa.feature.mfcc(x, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.savefig(f'{filepath}_mfcc.png')
    return np.mean(mfcc)

def spectral_slope(filepath):
    '''
    A spectral slope function that uses the mel spectrogram as input. 
    #reference: https://www.audiocontentanalysis.org/code/audio-features/spectral-slope-2/
    Input:
        row of dataset
    Output:
        mean spectral slope
    '''

    y, sr = librosa.load(filepath, sr=41000)
    y = librosa.to_mono(y)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, )
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(melspec, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=16000,
                             fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.savefig(f'{filepath}_mel_spec.png')
    mean = melspec.mean(axis=0, keepdims=True)
    kmu = np.arange(0, melspec.shape[0]) - melspec.shape[0] / 2
    melspec = melspec - mean
    spec_slope = np.dot(kmu, melspec) / np.dot(kmu, kmu)
    return np.mean(spec_slope)

def mean_spectral_rollof(filepath):
    '''
    The spectral roll-off , which indicates liveliness of audio signal.  
    Input:
        row of dataset
    Output:
        mean spectral roll-off
    '''
    y, sr = librosa.load(filepath, sr=41000)
    y = librosa.to_mono(y)
    rolloff_max = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0]
    spec_rf = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.50)[0] #from paper about education style
    rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)[0]
    S, phase = librosa.magphase(librosa.stft(y))
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(librosa.times_like(spec_rf), spec_rf, label='Roll-off frequency (0.5)')
    ax.plot(librosa.times_like(spec_rf), rolloff_min, color='w',
            label='Roll-off frequency (0.01)')
    ax.plot(librosa.times_like(spec_rf), rolloff_max, color='g',
            label='Roll-off frequency (0.99)')
    ax.legend(loc='lower right')
    ax.set(title='log Power spectrogram')
    plt.savefig(f'{filepath}_rolloff.png')
    return np.mean(spec_rf)
    
if __name__ =='__main__':
    pathlist = sorted(Path("C:/Users/Paige/Documents/directed_readings/VAD/finished/6pTk4Q4Gc_g_17/").glob('**/*.wav'), key=os.path.getmtime)
    slope = []
    roll = []
    duration = []
    for path in pathlist:
        print(str(path))
        ############## SPECTRAL FEATURES ###############################
        audio = AudioSegment.from_file(path)
        duration.append(audio.duration_seconds)
        s_slope = spectral_slope(path)
        print("############# spectral slope completed ##################")
        mfcc = analyse_mfcc(path)
        print("############# mfcc completed ##################")
        rolloff = mean_spectral_rollof(path)
        print("############# mean spectral roll-off completed ##################")
        slope.append(s_slope)
        roll.append(rolloff)

    spectral_df = pd.DataFrame(
      {'duration' : duration,
        'slope': slope,
       'rolloff': roll
      })

    spectral_df.to_csv("spectral_features.csv",index=False)