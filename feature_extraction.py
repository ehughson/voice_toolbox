from pydub import AudioSegment
from pathlib import Path  # For writing videos into the data folder
import pandas as pd 
import os
import argparse
from spectral_functions import *

'''
This file will return the ..... spectral slope, mfcc, and mean spectral roll-off
'''

# TODO consider allowing the user to input the functions to call
parser = argparse.ArgumentParser(description = "Provide the sampling rate and file path")
parser.add_argument(
  "samplingRate",
  help="desired sample rate for the file to be read with"
)
parser.add_argument(
  "filePath",
  help="path to the file, or folder containing the files, must be in .wav format"
)
parser.add_argument(
    "--writePath", default="spectral_features.csv",
    help = "optional file to append result to, default is a new file in the current location"
)
args = parser.parse_args()

SAMPLING_RATE = args.samplingRate
PATH = args.filePath
WRITE_PATH = args.writePath


#SAMPLE_RATE = 41000
#PATH = "C:/Users/Paige/Documents/directed_readings/VAD/finished/6pTk4Q4Gc_g_17/"
    
if __name__ =='__main__':
    # Files are read in order of the time created
    if ".wav" in PATH:
      pathlist = [PATH]
    else :
      pathlist = sorted(Path(PATH).glob('**/*.wav'), key=os.path.getmtime)
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