import numpy as np
import librosa.display
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


def radar_plot(df, cols, title):
    #print(df[cols].head())
    scaler = MinMaxScaler()
    df[cols] =scaler.fit_transform(df[cols])
    #print(df[cols].head())

    df = df.groupby('style')[cols].median().reset_index()
    categories=list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.5,1.0], ["0.1","0.5","1.0"], color="grey", size=7)
    plt.ylim(0,1)
    
    values=df.loc[0].drop('style').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="calming")
    ax.fill(angles, values, 'b', alpha=0.1)

    values=df.loc[1].drop('style').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="educational")
    ax.fill(angles, values, 'r', alpha=0.1)

    values=df.loc[2].drop('style').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="friendly")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # Show the graph
    plt.savefig("images/"+title)
    plt.show()


if __name__ =='__main__':

    df = pd.read_csv("processed_results.csv")

    df.rename(columns={'num_zero_crossings': 'ZCR', 'pause_length': 'pause rate', 'pitch': 'median F0'}, inplace=True)


    cols1 = ['max_intensity', 'ZCR',  'pause rate', 'mean_spectral_rf', 'spectral_slope']
    radar_plot(df, cols1, "PCA Important Features")

    cols3 = ['max_intensity', 'mean_intensity', 'ZCR', 'mean_spectral_rf']
    radar_plot(df, cols3, "PCA1 Important Features")

    cols4 = ['pause rate', 'spectral_slope', 'energy', 'peak_to_valley']
    radar_plot(df, cols4, "PCA2 Important Features")

    cols2 = ['peak_to_valley', 'median F0', 'mean_intensity',  'ZCR' ]
    radar_plot(df, cols2, "RFE Important Features")


    cols5 = ['spectral_slope', 'mean_spectral_rf', 'energy','ZCR' ]
    radar_plot(df, cols5, "Highly Correlated Features")


