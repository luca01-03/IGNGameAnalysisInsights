import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


try: 
    df = pd.read_csv('IGN.csv', 
                     names=['title', 'score','score_phrase','platform','genre','release_year','release_month','release_day'])
except Exception as e:
    print("Error: ", e)

shape = df.shape
describe = df.describe()
nulls = df.isnull().sum() # 36 nulls in genre
duplicates = df.duplicated().sum() # 48 duplicate rows
df_new = df.dropna().drop_duplicates()

df_new['score'] = pd.to_numeric(df['score'], errors='coerce') 
df_new['release_month'] = pd.to_numeric(df['release_month'], errors='coerce')

platform_average_scores = df_new.groupby('platform')['score'].mean().sort_values() # 60 platforms
top_20_platforms = platform_average_scores.head(20)


# Bar showing the average scores for the top 20 platforms
plt.figure(figsize=(15,10))
plt.barh(top_20_platforms.index, top_20_platforms.values, color='lightcoral', edgecolor='black')
plt.title('Average Game Scores by Platform')
plt.xlabel('Average Score')
plt.ylabel('Platform')

# Histogram showing distribution of game scores
plt.figure(figsize=(12, 6))
bins = np.arange(1, 10, 0.5)
plt.hist(df_new['score'], bins=bins, edgecolor='black', color='skyblue')
plt.title('Distribution of Game Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.show()
genre_count = df_new['genre'].value_counts()
platform_count = df_new['platform'].value_counts()
score_phrase_frequency = df_new['score_phrase'].value_counts()

# 'score_phrase' appears in index/axis 0 so removing
if 'score_phrase' in score_phrase_frequency.index:
    score_phrase_frequency = score_phrase_frequency.drop('score_phrase')

# Bar plot showing frequency of score phrases
plt.figure(figsize=(15, 10))
plt.bar(score_phrase_frequency.index, score_phrase_frequency.values, color='skyblue', edgecolor='black')
plt.title('Frequency of Score Phrases')
plt.xlabel('Score Phrase')
plt.ylabel('Frequency')
plt.xticks(rotation=45) 

release_years = df_new['release_year'].value_counts()
release_months = df_new['release_month'].value_counts().sort_index()

for_removal = ['release_year', '1970'] # These appear in the index, so removing before plotting

for i in for_removal:
    if i in release_years.index:
        release_years = release_years.drop(i)

# Biplot showing number of yearly and monthly game releases
fig, axes = plt.subplots(2, 1, figsize=(10, 20))

# Yearly subplot
axes[0].bar(release_years.index.astype(str), release_years.values, color='skyblue')
axes[0].set_title('Games Released Per Year', fontsize=13)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Games Released')
axes[0].tick_params(axis='x', rotation=45)

# Monthly subplot
axes[1].bar(release_months.index, release_months.values, color='lightgreen')
axes[1].set_title('Games Released Per Month', fontsize=13)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Games Released')
axes[1].set_xticks(range(1, 13)) 
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.subplots_adjust(hspace=0.4)


single_genre_df = df_new[~df_new['genre'].str.contains(',')]
single_genre_counts = single_genre_df['genre'].value_counts()

# Bar plot of number of games per platform
plt.figure(figsize=(15, 8))
plt.bar(platform_count.index, platform_count.values, color='skyblue', edgecolor='black')
plt.title('Number of Games Per Platform')
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)  



single_genre_df = df_new[~df_new['genre'].str.contains(',')]
single_genre_counts = single_genre_df['genre'].value_counts()
print(single_genre_counts.to_string)
# Bar plot of number of games for each single genre game
plt.figure(figsize=(15, 8))
plt.bar(single_genre_counts.index, single_genre_counts.values, color='skyblue', edgecolor='black')
plt.title('Number of Games Per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Games')
plt.xticks(rotation=90)  
plt.show()

