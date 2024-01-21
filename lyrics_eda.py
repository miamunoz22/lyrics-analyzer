import pandas as pd
import numpy as np
import os

import lyrics

''' ADD SUMMARY '''

lyrics_df = lyrics.lyrics_df

# Checking the imbalance in the Target Variable
plt.figure(figsize=[18,8])
plot = sb.countplot(lyrics_df['cluster'], palette = 'inferno')
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Cluster Data Count & Spread', fontdict={'fontsize': 20, 'fontweight': 5, 'color': 'Green'})
plt.xticks(rotation=90)
plt.show()


