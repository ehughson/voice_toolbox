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

    df = df.groupby('condition')[cols].median().reset_index()
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
    
    
    values=df.loc[1].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond2")
    ax.fill(angles, values, 'b', alpha=0.1)
    
    values=df.loc[3].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond4")
    ax.fill(angles, values, 'r', alpha=0.1)
    '''
    values=df.loc[5].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond6")
    ax.fill(angles, values, 'r', alpha=0.1)

    
    values=df.loc[7].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond8")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    values=df.loc[9].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond10")
    ax.fill(angles, values, 'r', alpha=0.1)
    '''
    
    values=df.loc[11].drop('condition').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="cond12")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    #Condition 12 is similar to 6 (Mexican) and 10 (Noisy Bar). 
    #Condition 12 is disimilar to 8 (quiet bar) and 2 and 4 (cafe and formal).
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # Show the graph
    plt.savefig("images/"+title)
    plt.show()


if __name__ =='__main__':

    df = pd.read_csv("processed_results.csv")

    df.rename(columns={'num_zero_crossings': 'ZCR', 'pause_length': 'pause rate', 'pitch': 'median F0'}, inplace=True)
    features = df.columns
    cols = ['condition','median F0']
    print(cols)
    df2 = df[cols]
    print(df2.groupby(['condition']).mean())

    cols = ['condition','pitch_range']
    print(cols)
    df2 = df[cols]
    print(df2.groupby(['condition']).mean())


    cols = ['condition','pause rate']
    print(cols)
    df2 = df[cols]
    print(df2.groupby(['condition']).mean())


    cols = ['condition','syll_count']
    print(cols)
    df2 = df[cols].loc[df['syll_count']!= 'NA']
    print(df2.groupby(['condition']).mean())
    '''
    print(features)
    for i in features[3:]:
        cols = ['condition',i]
        print(cols)
        df2 = df[cols]
        df2 = df2.groupby(['condition']).mean()
        df2.plot(kind='bar')
        plt.show()
    '''
    #cols7 = ['energy', 'spectral_slope', 'pause rate' ]
    #radar_plot(df, cols7, "Highly Correlated Features")  
    
    '''
    cols1 = ['max_intensity', 'ZCR',  'pause rate', 'mean_spectral_rf', 'spectral_slope']
    radar_plot(df, cols1, "Scripted Condition Results 1")

    cols3 = ['max_intensity', 'mean_intensity', 'ZCR', 'mean_spectral_rf']
    radar_plot(df, cols3, "PCA1 Important Features")

    cols4 = ['pause rate', 'spectral_slope', 'energy', 'peak_to_valley']
    radar_plot(df, cols4, "PCA2 Important Features")

    cols2 = ['peak_to_valley', 'median F0', 'mean_intensity',  'ZCR']
    radar_plot(df, cols2, "RFE Important Features")

    cols5 = ['spectral_slope', 'mean_spectral_rf', 'energy','ZCR' ]
    radar_plot(df, cols5, "Highly Correlated Features")

    
    cols6 = ['spectral_slope', 'mean_intensity', 'max_intensity' ]
    radar_plot(df, cols6, "Highly Correlated Features")

      

    cols7 = ['pitch_range', 'median F0', 'energy' ]
    radar_plot(df, cols7, "Highly Correlated Features")
    '''

'''
for i in range(1, 13):
    ##################### Correlation Analysis ##############################
    #df.rename(columns={'num_zero_crossings': 'ZCR', 'pause_length': 'pause rate', 'pitch': 'median F0'}, inplace=True)
    condition = df.loc[df['condition'] == i]
    features = ['spectral_slope','mean_spectral_rf', 'max_jump', 'peak_to_valley', 'median F0', 'pitch_range', 'pause rate', 'energy', 'mean_intensity', 'max_intensity', 'harmonics_to_noise', 'ZCR']
    heatmap = sns.heatmap(condition[features].corr(), annot=True)
    heatmap.set_title('Correlation Heatmap for condition ' + str(i), fontdict={'fontsize':12}, pad=12)
    plt.savefig("images/heatmap")
    plt.show()
'''

