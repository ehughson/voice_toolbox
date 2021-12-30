
# Prosodic Speech Analysis of an Argument

This repository contains the code to perform a preliminary analysis
on high arousal audio data to better understand how this arousal may
be classified

* [Requirements](#requirements)
* [Project Files](#project-files)
* [To Run](#to-run)
* [Audio Data](audio-data)

<!-- Requirements -->
## Requirements

* run

`pip install -r requirements.txt`

* In order to run tensorflow1 you must have Python < 3.7 and in order to intall Matlab engine Python > 3.6 must be installed

* Must have Matlab installed

* Install matlab engine
  * follow instructions [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)


<!-- PROJECT FILES -->
## Project Files

* `VAD/`
  * Only the files I have edited are listed, the other comes from this 
  [repo](https://github.com/jtkim-kaist/VAD)
  * `VAD.py` : Wrapper for calling the matlab functions and returns predictions and activation changes
  * `predict.py` : wraps all other code and segements files into actvations
  * `plot_preds.m` : plots the waveform and predicted activations
  * lib
    * matlab
      * `vad_funct` : performs MRCG feature extraction
  * `preds.jpg` : last prediction plot
  * 6pTk4Q4Gc_g : the clip currently under analysis
    * original 10-second segments
  * 2021-12-16 : for future analysis
    * original 10-second segements
  * K3r3TKmWZkA : for future analysis
    * original 10-second segements
* feature_extraction
  * `features.py` : extracts all non spectral features
  * `spectralfeatures.py` : extracts spectral features
  * `features.csv` : where the extracted non spectral features are written
  * `spectral_features.csv` : where all extracted spectral features are written
* results_cleaning
  * cleaned_data
    * `full.csv` : features and labels for all voices
    * `husband.csv` : features and labels for the husband
    * `long.cv` : features and labels for all voices for combined vad files
    * `man.csv` : features and labels for the other man
    * `wife.csv` : features and labels for the wife
  * `combine.py` : combines the dtaframes for individual segments and sections them by speaker
  * `visualizations.ipby` : code to plot bar and line plots, also found [here](https://colab.research.google.com/drive/1h1rcQgGmXOMpEoe5bAu_nmoP2PEQXEV9#scrollTo=jkEe93huiucZ)

## To Run

Start to finish instructions on how to begin with 10 second segments, extract audio features and plot for analysis

SET line 44 to desired folder for processing
pathlist = Path("$path/$to/VAD/K3r3TKmWZkA").glob('**/*.wav')

RUN
```
python predict.py
```

CREATE a `csv` inside each newly generated folder with columns including the label, speakers, and featueres and rows of each audio filee.\\
An existing `csv` may be used as a template

SET line 101 to desired folder, this is an original 10-second clip for divided into activtions
pathlist = sorted(Path("$path/$to/directed_readings/VAD/finished/6pTk4Q4Gc_g_17/").glob('**/*.wav'), key=os.path.getmtime)

RUN
```
python spectralfeatures.csv
```

PASTE from `spectralfeatures.csv` to repective lines in the current `csv`

SET 

line 168 to desired folder, this is an original 10-second clip divided into activations
pathlist = sorted(Path("$path/$to/directed_readings/VAD/finished/6pTk4Q4Gc_g_17/").glob('**/*.wav'), key=os.path.getmtime)

RUN
```
python features.csv
```

PASTE from `features.csv` to repective lines in the current `csv`

RUN
```
$ python combine.py
```

TO RENDER PLOTS you may run this in jypter notebook, or in google Colab
the file with plotting visualizations is `visualsation.py`

## Audio data

to access the original data check here `$path/%to/directed_readings/VAD/6pTk4Q4Gc_g`

to access processed dats look here `$path/%to/directed_readings/VAD//finished`