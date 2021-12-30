import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import parselmouth

def plotOnGraph(pitch, color):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2.5, color=color)
    
def setupGraph(ymin, ymax):
    sns.set() # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 150 # Show images nicely
    plt.figure()
    plt.ylim(ymin, ymax)
    plt.ylabel("frequency [Hz]")
    plt.xlabel("seconds")
    plt.grid(True)




filepath = "edited_clips_emma/emma_condition6_subset3_cleaned.wav"
sound = parselmouth.Sound(filepath).convert_to_mono()
pitchZh = sound.to_pitch()

setupGraph(50, 375)

plotOnGraph(pitchZh, 'r')

plt.gca().legend(('zh'))

plt.show()