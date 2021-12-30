import pandas as pd
from pathlib import Path  # For writing videos into the data folder
import numpy as np
import matplotlib.pyplot as plt

def addClipIndex(dataframe):
  '''
    Add the index of the order in the reconstructed clip
  '''
  num_rows_long = len(dataframe.index)
  index_long = range(1,num_rows_long+1)
  dataframe['clip index'] = index_long

pathlist = sorted(Path("C:/Users/Paige/Documents/directed_readings/VAD/finished/").glob('**/*.csv'))

#conctenate all csvs into a df
full_df = pd.concat((pd.read_csv(f) for f in pathlist))

#this creates ampty rows, get rid of them
full_df = full_df.dropna(how='all')

# relabel and unsual labels as 1
full_df['clean_label'] = np.where(
    full_df['label'] == 2, 2, np.where(
        full_df['label'] == 3, 3, 1
    )
)

# separate full clips to its own df
long_df = full_df[full_df.name != 'full']

full_df = full_df[full_df.name != 'full']

#fill null with 0
full_df = full_df.fillna(0)
long_df = long_df.fillna(0)

#split different speakers to their own df
man_wife_df, husband_df = [x for _, x in full_df.groupby(full_df['speaker'] == 1)]

wife_df, man_df = [x for _, x in man_wife_df.groupby(man_wife_df['speaker'] == 2)]

addClipIndex(full_df)
addClipIndex(man_df)
addClipIndex(wife_df)
addClipIndex(husband_df)
addClipIndex(long_df)

wife_df.to_csv("wife.csv",index=False)
husband_df.to_csv("husband.csv",index=False)
man_df.to_csv("man.csv",index=False)
full_df.to_csv("full.csv",index=False)
long_df.to_csv("long.csv",index=False)
