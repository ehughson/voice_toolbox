import parselmouth
from parselmouth.praat import call
from scipy.signal import argrelextrema
import librosa
import librosa.display
import numpy as np
from pathlib import Path  # For writing videos into the data folder
import pandas as pd
import os

def max_jump(filepath):
    '''
    The max jump from peak/valley to peak in the F0 contour. 
    Input:
        row of dataset
    Output:
        mean of the max jump between peak/valley to peak. 
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    F0 = sound.to_pitch(time_step = 0.0001, pitch_floor = 40).selected_array['frequency']
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

def analyse_pitch(filepath):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        mean of the fundamental frequency found  
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    F0 = sound.to_pitch(time_step = 0.0001).selected_array['frequency']
    F0[F0==0] = np.nan
    return np.nanmedian(F0)

def analyze_pitch_range(filepath):
    '''
    Pitch is the quality of sound governed by the rate of vibrations. Degree of highness and lowness of a tone.
    F0 is the lowest point in a periodic waveform. WARNING: this may not be applicable to current dataset 
    Input:
        row of dataset
    Output:
        range of the fundamental frequency found  
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    F0 = sound.to_pitch(time_step = 0.0001).selected_array['frequency']
    F0[F0==0] = np.nan
    minval = np.nanmin(F0)
    maxval = np.nanmax(F0)
    return maxval - minval

def get_energy(filepath):
    '''
    Energy of a signal corresponds to the total magnitude of the signal. 
    For audio signals, that roughly corresponds to how loud the signal is. 
    Input:
        row of dataset
    Output:
        energy of the signal. 
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    energy = sound.get_energy()
    return energy

def analyse_intensity(filepath):
    '''
    Intensity represents the power that the sound waves produce
    Input:
        row of dataset
    Output:
        Returns mean intensity or loudness of sound extracted from Praat. 
    '''
    y, sr = librosa.load(filepath, sr=80000)
    average_intensity = parselmouth.Sound(y).convert_to_mono().to_intensity()
    return average_intensity.get_average() #the duration will weight the average down for longer clips

def get_max_intensity(filepath):
    '''
    Intensity represents the power that the sound waves produce
    references: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
    Input:
        row of dataset
    Output:
        Returns max intensity or power in dB. 
    '''
    y, s = librosa.load(filepath, mono=True, sr = 41000)
    #### section from code taken from reference
    win_len_sec = 0.2
    power_ref=10**(-12)
    win_len = round(win_len_sec * s)
    win = np.ones(win_len) / win_len
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

def analyse_harmonics(filepath):
    '''
    Harmonics to noise which is the ratio of noise to harmonics in the audio signal.  
    Input:
        row of dataset
    Output:
        hnr
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr

def analyse_jitter(filepath):
    '''
    Deviations in individual consecutive F0 period lengths
    Input:
        row of dataset
    Output:
        mean local jitter
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return localJitter

def analyse_shimmer(filepath):
    '''
    Difference of the peak amplitudes of consecutive F0 periods.  
    Input:
        row of dataset
    Output:
        mean local shimmer
    '''
    y, sr = librosa.load(filepath, sr=41000)
    sound = parselmouth.Sound(y).convert_to_mono()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 400)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return localShimmer

if __name__ =='__main__':
    pathlist = sorted(Path("C:/Users/Paige/Documents/directed_readings/VAD/finished/6pTk4Q4Gc_g_17/").glob('**/*.wav'), key=os.path.getmtime)
    jumps = []
    pitches = []
    p_ranges = []
    jittery = []
    shimmery = []
    m_insensities = []
    intensities = []
    energies = []
    harmonics = []
    for path in pathlist:
      path = str(path)
      print(path)
      ################### PITCH FEATURES ###################
      jump = max_jump(path)
      print("############# max jump completed ##################")
      pitch = analyse_pitch(path)
      print("############# median F0 completed ##################")
      p_range = analyze_pitch_range(path)
      print("############# F0 range completed ##################")
      jitter = analyse_jitter(path)
      print("############ jitter completed ##################")
      shimmer = analyse_shimmer(path)
      print("############# shimmer completed ##################")

      ################### RATE OF SPEECH FEATURES and LOUDNESS ###################
      m_intensity = get_max_intensity(path)
      print("############# max intensity completed ##################")
      intensity = analyse_intensity(path)
      print("############# mean intensity completed ##################")
      energy = get_energy(path)
      print("############# energy completed ##################")

      ################### HARMONICS ##############################
      hnr = analyse_harmonics(path)
      print("############# HNR completed ##################")
      jumps.append(jump)
      pitches.append(pitch)
      p_ranges.append(p_range)
      jittery.append(jitter)
      shimmery.append(shimmer)
      m_insensities.append(m_intensity)
      intensities.append(intensity)
      energies.append(energy)
      harmonics.append(hnr)


    features_df = pd.DataFrame(
      {
          'pitches' : pitches,
          'P_ranges' : p_ranges,
          'jumps' : jumps,
          'jittery' : jittery ,
          'shimmery' : shimmery,
          'energies' : energies,
          'm_insensities' : m_insensities,
          'intensities' : intensities,
          'harmonics' :  harmonics 
      })

    features_df.to_csv("features.csv",index=False)
