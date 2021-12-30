import numpy as np
import parselmouth 
import librosa
import librosa.display
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


sns.set()
plt.rcParams['figure.dpi'] = 100
#GOAL: visualize the different styles of voice in each of the feature spaces
def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

def facet_util(data, **kwargs):
    digit, speaker_id = data[['digit', 'speaker_id']].iloc[0]
    sound = parselmouth.Sound("audio/{}_{}.wav".format(digit, speaker_id))
    draw_spectrogram(sound.to_spectrogram())
    plt.twinx()
    draw_pitch(sound.to_pitch())
    # If not the rightmost column, then clear the right side axis
    if digit != 5:
        plt.ylabel("")
        plt.yticks([])

def show_contour(filepath):
    plt.figure()
    sound = parselmouth.Sound(filepath)
    pitch = sound.to_pitch()  
    spectrogram = sound.to_spectrogram()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([sound.xmin, sound.xmax])
    plt.show()
    
def wave_graph(row):
    title = row['Title']
    filepath = "audio/{}.wav".format(title)
    snd = parselmouth.Sound(filepath).convert_to_mono()
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

def zero_crossing(row):
    title = row['Title']
    filepath = "audio/{}.wav".format(title)
    x, sr = librosa.load(filepath)
    #Plot the signal:
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()
        # Zooming in
    n0 = 9000
    n1 = 9100
    plt.figure(figsize=(14, 5))
    plt.plot(x[n0:n1])
    plt.grid()
    plt.show()
    #zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    #print(sum(zero_crossings))

def power_spectrum(filepath, title):
    signal, sr = librosa.load(filepath)
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    left_frequency = frequency[: int(len(frequency)/2)]
    left_magnitude = magnitude[: int(len(magnitude)/2)]
    plt.plot(left_frequency, left_magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("magnitude")
    plt.title(title)
    plt.show()


def plot_mfcc(filepath):
    fig, ax = plt.subplots()
    y, sr = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()

def plot_intensity(filepath):
    snd = parselmouth.Sound(filepath)
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show()


#power_spectrum("audio/friendly_voice_7_cleaned.wav", "Power Spectrum of Friendly Style")
#power_spectrum("audio/calming_voice_6_cleaned.wav", "Power Spectrum of Calm Style")
#power_spectrum("audio/educational_voice_10_cleaned.wav", "Power Spectrum of Educational Style")

df = pd.read_csv("processed_results.csv")
df_mi = df.get(['style', 'max_intensity'])
sns.histplot(data=df_mi, x="max_intensity", hue="style", multiple="stack")
plt.xlabel("dB")
plt.title("Max Intensity")

plt.savefig("images/max intensity histogram")
plt.show()


df_pitch = df.get(['style', 'pitch'])
#df_educational = df[df['style'] == 'educational']
sns.histplot(data=df_pitch, x='pitch', hue="style", multiple="stack")
plt.xlabel("Hz")
plt.title("Median Pitch")

plt.savefig("images/pitch histogram")
plt.show()

df_energy = df.get(['style', 'energy', 'syll_count'])

groups = df_energy.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.energy, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("energy (P$a^2$·s)")
plt.xlabel("syllables per second")
plt.title("Syllables vs Energy")

plt.savefig("images/energy_sp")
plt.show()


df_max = df.get(['style', 'max_intensity', 'syll_count'])

groups = df_max.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.max_intensity, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("Max Intensity (dB)")
plt.xlabel("syllables per second")
plt.title("Syllables vs Max Intensity")

plt.savefig("images/Max Intensity sp")
plt.show()

df_hnr = df.get(['style', 'harmonics_to_noise', 'syll_count'])

groups = df_hnr.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.harmonics_to_noise, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("HNR (dB)")
plt.xlabel("syllables per second")
plt.title("syllables vs HNR")

plt.savefig("images/HNR sp")
plt.show()

df_rf= df.get(['style', 'mean_spectral_rf', 'syll_count'])

groups = df_rf.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.mean_spectral_rf, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("spectral roll-off (Hz)")
plt.xlabel("syllables per second")
plt.title("Syllables vs Spectral Roll-off")

plt.savefig("images/rolloff sp")
plt.show()

df_rf= df.get(['style', 'spectral_slope', 'syll_count'])

groups = df_rf.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.spectral_slope, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("spectral slope (Hz)")
plt.xlabel("syllables per second")
plt.title("Syllables vs Mean Spectral Slope")

plt.savefig("images/spectral slope sp")
plt.show()

df_ptv = df.get(['style', 'peak_to_valley', 'syll_count'])

groups = df_ptv.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.syll_count, group.peak_to_valley, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("Peak to Valley (Hz)")
plt.xlabel("syllables per second")
plt.title("Syllables vs Peak to Valley")

plt.savefig("images/PtV sp")
plt.show()

df_ptv = df.get(['style', 'energy','spectral_slope'])

groups = df_ptv.groupby('style')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.spectral_slope, group.energy, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.ylabel("energy (P$a^2$·s)")
plt.xlabel("spectral slope")
plt.title("Energy vs Spectral Slope")

plt.savefig("images/energy_specslope sp")
plt.show()



