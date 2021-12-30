from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns

##################### PCA Analysis ##############################
df = pd.read_csv("processed_results.csv")
features = ['syll_count','spectral_slope','mean_spectral_rf', 'max_jump', 'peak_to_valley', 'pitch', 'pause_length', 'energy', 'mean_intensity', 'max_intensity', 'harmonics_to_noise', 'num_zero_crossings']
X = df.loc[:, features].values
y = df.loc[:,['style']].values
x = StandardScaler().fit_transform(X)
print(np.mean(x),np.std(x))
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
#col = ['principal component 1','principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8']
col = ['principal component 1','principal component 2']
principalDf = pd.DataFrame(data = principalComponents
             , columns = col)
finalDf = pd.concat([principalDf, df[['style']]], axis = 1)
print(finalDf.head())
print(pca.explained_variance_ratio_)
print(abs( pca.components_ ))


##################### RFE Analysis ##############################
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(x, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


##################### Correlation Analysis ##############################
df.rename(columns={'num_zero_crossings': 'ZCR', 'pause_length': 'pause rate', 'pitch': 'median F0'}, inplace=True)
features = ['syll_count','spectral_slope','mean_spectral_rf', 'max_jump', 'peak_to_valley', 'median F0', 'pause rate', 'energy', 'mean_intensity', 'max_intensity', 'harmonics_to_noise', 'ZCR']
heatmap = sns.heatmap(df[features].corr(), annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.savefig("images/heatmap")
plt.show()


