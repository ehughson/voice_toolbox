import parselmouth
from parselmouth.praat import call
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path

import speech_recognition as sr
import syllables
import soundfile as sf

from scipy import signal
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from pydub import AudioSegment
from pydub.silence import split_on_silence



SAMPLE_RATE = 22050
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

#TODO get pitch range
#TODO for preprocessing:
    # Get everything into mono and remove noise in background (DONE!!!!!!!!:D)
    # cut down clips!!! Maybe split long ones into two separate clips? (DONE!! KIND-OF)
    # Add in rate of speech as a feature!!!!!! DONE!
    # Normalize all values
    # words per minute = total words/time
    # syllables per second (done!!!)
    # Text aligners -- will tell you how the wave form and words align - future project
    # GET ENVELOPE (POWER VS FREQUENCY PLOT)
    
#Audacity removes background noise better!!

def max_jump(filepath, sample_rate=21000):
    '''
    The max jump from peak/valley to peak in the F0 contour. 
    Input:
        row of dataset
    Output:
        mean of the max jump between peak/valley to peak. 
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']

    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    t =librosa.get_duration(y=y, sr=s)
    F0[F0==0] = np.nan
    peak_differences = []
    valleys = argrelextrema(F0, comparator=np.less)
    peaks = argrelextrema(F0, np.greater)
    i = 0
    j = 0

    while j < len(valleys[0]) and i < len(peaks[0])-1:
        if peaks[0][i] < valleys[0][j]:
            peak_differences.append(np.abs(F0[peaks[0][i]]-F0[peaks[0][i+1]])) #TODO: find out if this should be absolute?
            i = i+1
        else:
            peak_differences.append(np.abs(F0[valleys[0][j]]-F0[peaks[0][i+1]])) 
            j = j+1

    
    return np.mean(peak_differences)/100

def peak_to_valley(filepath, sample_rate=21000):
    '''
    Mean peak to valley distance in F0 contour. 
    Input:
        row of dataset
    Output:
        mean peak to valley. 
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']

    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    t =librosa.get_duration(y=y, sr=s)
    F0[F0==0] = np.nan
    peak_differences = []
    valleys = argrelextrema(F0, comparator=np.less)
    peaks = argrelextrema(F0, np.greater)
    i = 0
    j = 0

    while j < len(valleys[0]) and i < len(peaks[0]):
        if peaks[0][i] <= valleys[0][j]:
            peak_differences.append(np.abs(peaks[0][i]-valleys[0][j])) #TODO: find out if this should be absolute?
            i = i+1
        else:
            j = j+1
    
    return (np.mean(peak_differences)/100)

def analyse_pitch(filepath, sample_rate=21000):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        mean of the fundamental frequency found  
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']
    F0[F0==0] = np.nan
    return np.nanmedian(F0)

def analyze_pitch_range(filepath, sample_rate=21000):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        mean of the fundamental frequency found  
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    F0 = sound.to_pitch().selected_array['frequency']
    F0[F0==0] = np.nan
    minval = np.nanmin(F0)
    maxval = np.nanmax(F0)
    return maxval - minval

def analyse_formants(f, filepath):
    '''
    "A formant is acoustic energy around a frequency"
    CURRENTLY: Measures formants ONLY at glottal pulses
    Input:
        row of dataset
        f: formant we want
    Output:
        mean of the given formant (e.g., f1, f2, f3, f4)
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    #pointProcess = parselmouth.praat.call(sound,"To PointProcess (periodic, cc)...", 75, 300)
    #formants = parselmouth.praat.call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    pitch = call(sound, "To Pitch", 0.0, 75, 500)  # check pitch to set formant settings
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")  # get mean pitch
    if meanF0 > 150:
        maxFormant = 5500
    else:
        maxFormant = 5000
    formants = sound.to_formant_burg(time_step=0.010, maximum_formant=maxFormant)
    #numPoints = parselmouth.praat.call(pointProcess, "Get number of points")
    f_list = []
    
    for t in formants.ts():
        #t = parselmouth.praat.call(pointProcess, "Get time from index", point)
        #f_val = formants.get_value_at_time(f, t, 'Hertz', 'Linear')
        f_val = parselmouth.praat.call(formants, "Get value at time", f, t, 'Hertz', 'Linear')
        
        if str(f_val) != 'nan':
            f_list.append(f_val)
        else:
            f_list.append(0)

    return np.mean(f_list)

def analyse_mfcc(filepath, outputpath, sample_rate=21000):
    '''
    Creates MFCC 
    Input:
        row of dataset
    Output:
        outputs 20 by # array containing the mfcc 
    '''
    x, sr = librosa.load(filepath, sr=sample_rate)
    x = librosa.to_mono(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    filname,ext = os.path.splitext(os.path.basename(os.path.normpath(filepath)))

    S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,
                                   fmax=8000)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')

    fig.savefig(f'{outputpath}{filname}MFCC.pdf')
    return np.mean(mfcc)

def get_energy(filepath,sample_rate=21000):
    '''
    Energy of a signal corresponds to the total magnitude of the signal. 
    For audio signals, that roughly corresponds to how loud the signal is. 
    Input:
        row of dataset
    Output:
        energy of the signal. 
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    energy = sound.get_energy()
    y, s = librosa.load(filepath)
    t =librosa.get_duration(y=y, sr=s)

    return energy

def analyse_intensity(filepath,sample_rate=21000):
    '''
    Intensity represents the power that the sound waves produce
    Input:
        row of dataset
    Output:
        Returns mean intensity or loudness of sound extracted from Praat. 
    '''
    #mean_intensity = parselmouth.praat.call(parselmouth.Sound(filepath).convert_to_mono().to_intensity(), "Get mean", 0, 0, "energy")
    average_intensity = parselmouth.Sound(filepath).convert_to_mono().to_intensity()
    y, s = librosa.load(filepath)
    t =librosa.get_duration(y=y, sr=s)
    return average_intensity.get_average() #the duration will weight the average down for longer clips

def get_max_intensity(filepath, sample_rate=21000):
    '''
    Intensity represents the power that the sound waves produce
    references: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
    Input:
        row of dataset
    Output:
        Returns max intensity or power in dB. 
    '''
    y, s = librosa.load(filepath, mono=True, sr = sample_rate) #TODO rremove sr her
    #### section from code taken from reference
    win_len_sec = 0.2
    power_ref=10**(-12)
    win_len = round(win_len_sec * s)
    win = np.ones(win_len) / win_len
    if len(win) == 0 or len(y) == 0:
        return "NA"
    power = 10 * np.log10(np.convolve(y**2, win, mode='same') / power_ref)
    #TODO: putting the z-score here ruins the audio signal :/
    '''
    z = np.abs(stats.zscore(power))
    indexes = np.where(z > 5)
    plt.show()
    power[indexes] = 0
    print(len(power))
    import soundfile as sf
    sf.write('stereo_file.wav', power, s)
    '''
    return np.max(power)

def analyze_zero_crossing(filepath, sample_rate=21000):
    '''
    Zero crossing tells us where the voice and unvoice speech occurs. 
    "Large number of zero crossings tells us there is no dominant low frequency oscillation"
    https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b#:~:text=Zero%2Dcrossing%20rate%20is%20a,feature%20to%20classify%20percussive%20sounds.
    Input:
        row of dataset
    Output:
        Returns the number of zero crossing points that occur 
    '''
    x, s = librosa.load(filepath)
    x = librosa.to_mono(x)
    
    t =librosa.get_duration(y=x, sr=s)
    zero_crossings = librosa.feature.zero_crossing_rate(x)
    return np.mean(zero_crossings)

def clean_audio(input_file, clean_file): 
    noise_file=input_file[0:-4]+'_noise.wav'
    print(noise_file)
    os.system('sox %s %s trim 0 1.000'%(input_file, noise_file))
    os.system('sox %s -n noiseprof noise.prof'%(noise_file))
    os.system('sox %s %s noisered noise.prof 0.3'%(input_file, clean_file))
    os.remove(noise_file)
    os.remove('noise.prof')
    return clean_file

def cleaning(filepath):
    '''
    Cleaning function to remove noise using sox.  
    Input:
        row of dataset
    Output:
        returns a clean audio file with noise removed. 
    '''
    wav_audio = AudioSegment.from_file(filepath, format="m4a")
    wav_audio.export(filepath[0:-4] + ".wav", format="wav")
    input_file = filepath[0:-4] + ".wav"
    print(filepath[0:-4])
    clean_file = input_file[0:-4]+'_cleaned.wav'
    cleaned_file = clean_audio(filepath, clean_file)
    return cleaned_file 

def get_number_sylls(filepath, sample_rate=21000):
    '''
    Rate of speech using number of syllables per second. 
    Input:
        row of dataset
    Output:
        syllables/second. 
    '''
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:              
        audio = r.record(source)                        

    try:
        list = r.recognize_google(audio, key=None, language='en-IN')     
        #print(len(list.split()))

    except LookupError:                                
        print("Could not understand audio")
    except sr.RequestError:
        # API was unreachable or unresponsive
        response = False
    except sr.UnknownValueError:    
        r = sr.Recognizer()
        with sr.AudioFile(filepath) as source:              
            audio = r.record(source)  
        try:
            list = r.recognize_google(audio, key=None, language='en-IN')     
            #print(len(list.split()))
        except LookupError:                                
            print("Could not understand audio")
        except sr.RequestError:
            # API was unreachable or unresponsive
            return 'NA'        
        except sr.UnknownValueError: 
            return 'NA'

    syll = syllables.estimate(list)
    y, s = librosa.load(filepath, sr = sample_rate)
    y = librosa.to_mono(y)
    duration =librosa.get_duration(y=y, sr=s)
    return syll/duration

def get_number_words(filepath, sample_rate=21000):
    '''
    Rate of speech using number of words per second. 
    Input:
        row of dataset
    Output:
        words/second. 
    '''
    
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:              
        audio = r.record(source)                        

    try:
        list = r.recognize_google(audio, key=None, language='en-IN')     
        #print(len(list.split()))

    except LookupError:                                
        print("Could not understand audio")
    except sr.RequestError:
        # API was unreachable or unresponsive
        response = False
    except sr.UnknownValueError:    
        r = sr.Recognizer()
        with sr.AudioFile(filepath) as source:              
            audio = r.record(source)  
        try:
            list = r.recognize_google(audio, key=None, language='en-IN')     
            #print(len(list.split()))
        except LookupError:                                
            print("Could not understand audio")
        except sr.RequestError:
            # API was unreachable or unresponsive
            return 'NA'        
        except sr.UnknownValueError: 
            return 'NA'

    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    duration =librosa.get_duration(y=y, sr=s)
    return len(list.split())/duration

def spectral_slope(filepath, sample_rate=21000):
    '''
    A spectral slope function that uses the mel spectrogram as input. 
    #reference: https://www.audiocontentanalysis.org/code/audio-features/spectral-slope-2/
    Input:
        row of dataset
    Output:
        mean spectral slope
    '''
    y, s = librosa.load(filepath, sr = sample_rate)
    y = librosa.to_mono(y)
    melspec = librosa.feature.melspectrogram(y=y, sr=s)
    t =librosa.get_duration(y=y, sr=s)
    mean = melspec.mean(axis=0, keepdims=True)
    kmu = np.arange(0, melspec.shape[0]) - melspec.shape[0] / 2
    melspec = melspec - mean
    spec_slope = np.dot(kmu, melspec) / np.dot(kmu, kmu)
    return np.mean(spec_slope)

def get_envelope(filepath, sample_rate=21000):
    '''
    Returns spectral envelope. INCOMPLETE: NEEDS TO BE FINISHED.  
    Input:
        row of dataset
    Output:
        spectral envelope
    '''
    y, s = librosa.load(filepath,sr = sample_rate)
    y = librosa.to_mono(y)
    t =librosa.get_duration(y=y, sr=s)

    ps = np.abs(np.fft.fft(y))**2


    freqs = np.fft.fftfreq(y.size)
    idx = np.argsort(freqs)

    plt.plot(freqs[idx], ps[idx])
    plt.show()

    f, Pxx_spec = signal.welch(y, 1/30, 'flattop', 1024, scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectrum (scipy.signal.welch)')
    plt.show()
    exit(-1)
    return

def analyse_harmonics(filepath, sample_rate=21000):
    '''
    Harmonics to noise which is the ratio of noise to harmonics in the audio signal.  
    Input:
        row of dataset
    Output:
        hnr
    '''
    sound = parselmouth.Sound(filepath).convert_to_mono()
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr

def mean_spectral_rollof(filepath, sample_rate=21000):
    '''
    The spectral roll-off , which indicates liveliness of audio signal.  
    Input:
        row of dataset
    Output:
        mean spectral roll-off
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    spec_rf = librosa.feature.spectral_rolloff(y=y, sr=s, roll_percent=0.50)[0] #from paper about education style
    return np.mean(spec_rf)

def pauses(filepath, sample_rate=21000):
    '''
    Pause rate which is an indicate of rate of speech. Calculated by dividing the duration by 
    the total number of pauses.
    Input:
        row of dataset
    Output:
        pause rate
    '''
    #reference:https://www.geeksforgeeks.org/python-speech-recognition-on-large-audio-files/
    file = AudioSegment.from_wav(filepath)
    chunks = split_on_silence(file,
        min_silence_len = 50,
        silence_thresh = -45
    )
    y, s = librosa.load(filepath, sr=sample_rate)
    y = librosa.to_mono(y)
    t =librosa.get_duration(y=y, sr=s)
    if t == 0:
        return "NA"
    try:
        pause_length=len(chunks)/t
    except:
        chunks = split_on_silence(file,
        min_silence_len = 500,
        silence_thresh = -50
        )
        pause_length=len(chunks)/t

    return pause_length

def analyse_jitter(filepath, sample_rate=21000):
    '''
    Deviations in individual consecutive F0 period lengths
    Input:
        row of dataset
    Output:
        mean local jitter
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return localJitter

def analyse_shimmer(filepath, sample_rate=21000):
    '''
    Difference of the peak amplitudes of consecutive F0 periods.
    Input:
        row of dataset
    Output:
        mean local shimmer
    '''
    y, s = librosa.load(filepath, sr=sample_rate)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return localShimmer
