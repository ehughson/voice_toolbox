from pydub import AudioSegment
from pathlib import Path  # For writing videos into the data folder
import pandas as pd 
import os
import argparse
#from spectral_functions import *
from parsel_process import * 


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description = "Provide the sampling rate and file path and features to be extracted")
    parser.add_argument(
      "samplingRate",
      help="desired sample rate for the file to be read with"
    )
    parser.add_argument(
      "filePath",
      help="path to the file (including '.wav extension), or folder containing the files, audio files must be in .wav format"
    )
    parser.add_argument(
        "--writePath", default="spectral_features.csv",
        help = "file to append result to, default is a new file in the current location"
    )
    parser.add_argument(
      "--formants", type=str, default=False,
      help="extract 4 formants"
    )
    parser.add_argument(
      "--ZCR", type=str, default=False,
      help="extract zero crossing rate"
    )
    parser.add_argument(
      "--harmonics", type=str, default=False,
      help="extract harmonics")
    parser.add_argument(
      "--rate_of_speech", type=str,  default=False,
      help="extract number of syllables, words and pauses"
      )
    parser.add_argument(
      "--loudness", type=str,  default=False,
      help="extract max intensity and intensity"
    )
    parser.add_argument(
      "--pitch_features", type=str,  default=False,
      help="extract max pitch jump, peak to valley, pitch and pitch range"
      )
    parser.add_argument(
      "--spectral_features", type=str,  default=False,
      help="extract spectral envelope, spectral slope, mfcc and mean spectral roll off"
    )
    parser.add_argument("--energy", type=str,  default=False, help="feature 8") 

    args = parser.parse_args()

    SAMPLING_RATE = args.samplingRate
    PATH = args.filePath
    WRITE_PATH = args.writePath
    
    function_dic = {"formants": [analyse_formants(1), analyse_formants(2), analyse_formants(3), analyse_formants(4)],
                    "ZCR": [analyze_zero_crossing],
                    "harmonics": [analyse_harmonics], 
                    "rate_of_speech": [get_number_sylls, get_number_words, pauses],
                    "loudness": [get_max_intensity, analyse_intensity], 
                    "pitch_features":[max_jump, peak_to_valley, analyse_pitch, analyze_pitch_range],
                    "spectral_features":[get_envelope, spectral_slope, analyse_mfcc, mean_spectral_rollof],
                    "energy":[get_energy]}
    
    # Files are read in order of the time created
    if ".wav" in PATH:
      pathlist = [PATH]
    else :
      pathlist = sorted(Path(PATH).glob('**/*.wav'), key=os.path.getmtime)
    dic = {}
    for k in args.__dict__:
      print(k)
      if args.__dict__[k] == "True":
        dic[k] = []
      
    #print(dic)
    slope = []
    roll = []
    duration = []
    filename = []
    if not pathlist:
      raise ValueError("The filepath must be a .wav file or a folder containing .wav files")
    for path in pathlist:
      filename_ext = os.path.basename(os.path.normpath(path))
      filename_no_ext = filename_ext.split('.', 1)[0]
      filename.append(filename_no_ext)
      for feature in dic:
        for func in function_dic[feature]:
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
    spectral_df.to_csv("spectral_features.csv",index=False)
