import pandas as pd
import matplotlib.pyplot as plt
import config as cf

path = cf.DATA_FILE
df = pd.read_csv(path)

df['num_moves'] = df['moves'].str.split().apply(len)

max_moves = df['num_moves'].max()
min_moves = df['num_moves'].min()
avg_moves = df['num_moves'].mean()

print(f"Maximum  moves: {max_moves}")
print(f"Minimum moves: {min_moves}")
print(f"Average moves: {avg_moves}")

plt.figure(figsize=(10, 6))
plt.hist(df['black_rating'].dropna(), bins=range(min(df['black_rating'].dropna().astype(int)), max(df['black_rating'].dropna().astype(int)) + 5, 5), alpha=0.7, label='Black Elo')
plt.hist(df['white_rating'].dropna(), bins=range(min(df['white_rating'].dropna().astype(int)), max(df['white_rating'].dropna().astype(int)) + 5, 5), alpha=0.7, label='White Elo')
plt.xlabel('Elo Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Black and White Elo Ratings')
plt.legend()
plt.grid(True)
plt.show()

plt.hist(df['num_moves'], bins=range(min_moves, max_moves + 5, 5), edgecolor='black')
plt.xlabel('Number of Moves')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Moves')
plt.show()

victory_status_counts = df['victory_status'].value_counts()