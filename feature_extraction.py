from pydub import AudioSegment
from pathlib import Path  # For writing videos into the data folder
import pandas as pd
import os
import argparse
#from spectral_functions import *
from parsel_process import *

def add_filpath(filepath):
  return filepath

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description = "Provide the sampling rate and file path and features to be extracted")
    parser.add_argument(
      "samplingRate", type = int,
      help="desired sample rate for the file to be read with"
    )
    parser.add_argument(
      "filePath", type = str,
      help="path to the file (including '.wav extension), or folder containing the files, audio files must be in .wav format"
    )
    parser.add_argument(
        "--writePath", type = str, default="spectral_features.csv",
        help = "file to append result to, default is a new file in the current location"
    )
    parser.add_argument(
      "--formants", action="store_true",
      help="extract 4 formants"
    )
    parser.add_argument(
      "--ZCR", action="store_true",
      help="extract zero crossing rate"
    )
    parser.add_argument(
      "--harmonics", action="store_true",
      help="extract harmonics")
    parser.add_argument(
      "--rate_of_speech", action="store_true",
      help="extract number of syllables, words and pauses"
      )
    parser.add_argument(
      "--loudness", action="store_true",
      help="extract max intensity and intensity"
    )
    parser.add_argument(
      "--pitch_features", action="store_true",
      help="extract max pitch jump, peak to valley, pitch, shimmer, jitter and pitch range"
      )
    parser.add_argument(
      "--spectral_features", action="store_true",
      help="extract spectral envelope, spectral slope, mfcc and mean spectral roll off"
    )
    parser.add_argument(
      "--energy", action="store_true",
      help="extract energy")

    args = parser.parse_args()

    SAMPLING_RATE = args.samplingRate
    PATH = args.filePath
    WRITE_PATH = args.writePath

    function_dic = {"filepath":[add_filpath],
                    "formants": [analyse_formants, analyse_formants, analyse_formants, analyse_formants],
                    "ZCR": [analyze_zero_crossing],
                    "harmonics": [analyse_harmonics],
                    "rate_of_speech": [get_number_sylls, get_number_words, pauses],
                    "loudness": [get_max_intensity, analyse_intensity],
                    "pitch_features":[max_jump, peak_to_valley, analyse_pitch, analyze_pitch_range, analyse_shimmer, analyse_jitter],
                    "spectral_features":[get_envelope, spectral_slope, analyse_mfcc, mean_spectral_rollof],
                    "energy":[get_energy]}

    # Files are read in order of the time created
    if ".wav" in PATH:
      pathlist = [PATH]
    else :
      pathlist = sorted(Path(PATH).glob('**/*.wav'), key=os.path.getmtime)
    dic = {"filepath": []}
    for k in args.__dict__:
      if (args.__dict__[k] == True):
        dic[k] = []

    filename = []
    itr = 0
    store_formants = []
    if not pathlist:
      raise ValueError("The filepath must be a .wav file or a folder containing .wav files")
    for path in pathlist:
      filename_ext = os.path.basename(os.path.normpath(path))
      filename_no_ext = filename_ext.split('.', 1)[0]
      filename.append(filename_no_ext)
      for feature in dic:
        for func in function_dic[feature]:
          if feature == "formants" and itr <4:
            store_formants.append(func(itr+1, path))
            itr+=1
            if itr == 4:
              itr = 0
              dic[feature].append(store_formants)
              store_formants = []
          else:
            value = func(path)
            dic[feature].append(value)
    """
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
    """

    #spectral_df = pd.DataFrame(
    #  {'duration' : duration,
    ##    'slope': slope,
    #   'rolloff': roll
    #  })
    spectral_df = pd.DataFrame.from_dict(dic)
    spectral_df.to_csv(WRITE_PATH,index=False)
