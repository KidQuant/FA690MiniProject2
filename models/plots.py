import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

news_df = pd.read_csv('data/news.csv')
price_df = pd.read_csv('data/price.csv')


# Count the number of articles and sort in descending order
# Convert date column to datetime and create month_year column
news_df['date'] = pd.to_datetime(news_df['publication_datetime'])
news_df['month_year'] = news_df['date'].dt.to_period('M')

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Top 10 Tickers
article_counts = news_df['tickers'].value_counts().sort_values(ascending=False).head(10)
sns.barplot(x=article_counts.index, y=article_counts.values, ax=axes[0, 0], color='#A32638')
axes[0, 0].set_title('(a)', fontsize=15)
axes[0, 0].set_xlabel('Ticker')
axes[0, 0].set_ylabel('Number of Articles')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Monthly Article Counts
monthly_counts = news_df['month_year'].value_counts().sort_index()
sns.barplot(x=monthly_counts.index.astype(str), y=monthly_counts.values, ax=axes[0, 1], color='#A32638')
axes[0, 1].set_title('(b)', fontsize=15)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Number of Articles')
axes[0, 1].tick_params(axis='x', rotation=45)
# Set x-axis ticks to show every other month
axes[0, 1].set_xticks(axes[0, 1].get_xticks()[::2])

# Plot 3: AMZN Monthly Articles
amzn_news = news_df[news_df['tickers'].str.contains('AMZN', na=False)]
amzn_monthly_counts = amzn_news['month_year'].value_counts().sort_index()
sns.barplot(x=amzn_monthly_counts.index.astype(str), y=amzn_monthly_counts.values, ax=axes[1, 0], color='#A32638')
axes[1, 0].set_title('(c)', fontsize=15)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Number of Articles')
axes[1, 0].tick_params(axis='x', rotation=45)
# Set x-axis ticks to show every other month
axes[1, 0].set_xticks(axes[1, 0].get_xticks()[::2])

# Plot 4: YUM Monthly Articles
yum_news = news_df[news_df['tickers'].str.contains('YUM', na=False)]
yum_monthly_counts = yum_news['month_year'].value_counts().sort_index()
sns.barplot(x=yum_monthly_counts.index.astype(str), y=yum_monthly_counts.values, ax=axes[1, 1], color='#A32638')
axes[1, 1].set_title('(d)', fontsize=15)
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Articles')
axes[1, 1].tick_params(axis='x', rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
