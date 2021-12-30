# Voice Toolbox <img align="left" width="90" height="90" src="soundwave.jpeg">
The place to solve all your audio signal processing needs. 

## Files
To start: Setup a conda environment and run 'pip3 install -r requirements.txt' file before running the scripts. 
>>Important: if you get an error with parselmouth make sure the installation is 'pip3 install praat-parselmouth'

The script for extracting features is parsel_process.py. 
 * To run: 'python3 parsel_process.py'
> * all features will be saved to "processed_results.csv"
 
For visualization:
 1. visualize_voice.py for all scatter plots along with other plotting features from praat. 
 * To run: 'python3 visualize_voice.py'
 2. radar_plot.py for all radar plots
 * To run: 'python3 radar_plot.py'

 voice_pca.py is for PCA, RFE and Correlation plot:
* - To run: 'voice_pca.py'
