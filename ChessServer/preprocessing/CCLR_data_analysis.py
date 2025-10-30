import pandas as pd
import numpy as np
import re

# Read the CSV file
df = pd.read_csv('./data/CCLR_validation_dataset_95:5.csv')

# Win rates
print("\n1. GAME RESULTS:")
total_games = len(df)
white_wins = len(df[df['Result'] == 1])
black_wins = len(df[df['Result'] == 0])
draws = len(df[df['Result'] == '1/2'])

white_win_rate = (white_wins / total_games) * 100
black_win_rate = (black_wins / total_games) * 100
draw_rate = (draws / total_games) * 100

print(f"Total Games: {total_games}")
print(f"White Win Rate: {white_win_rate:.2f}% ({white_wins} games)")
print(f"Black Win Rate: {black_win_rate:.2f}% ({black_wins} games)")
print(f"Draw Rate: {draw_rate:.2f}% ({draws} games)")

#Game termination
print("\n2. GAME TERMINATION TYPES:")
def infer_termination(moves_str, result):
    if pd.isna(moves_str):
        return 'Unknown'
    
    moves_str = str(moves_str).strip()
    
    if '#' in moves_str:
        return 'Checkmate'
    
    elif result in [1, 0]:
        return 'Resignation'
    
    else:
        return 'Unknown'
    
df['termination_type'] = df.apply(lambda row: infer_termination(row['pgn'], row['Result']), axis=1)
termination_counts = df['termination_type'].value_counts()
for termination_type, count in termination_counts.items():
    percentage = (count / total_games) * 100
    print(f"{termination_type}: {percentage:.2f}% ({count} games)")
    
# 3. Average number of moves
print("\n3. AVERAGE MOVES PER GAME:")
def extract_move_count(moves_str):
    if pd.isna(moves_str):
        return np.nan
    
    moves_str = str(moves_str).strip()
    move_numbers = re.findall(r'(\d+)\.', moves_str)
    
    if move_numbers:
        last_move_num = int(move_numbers[-1])
        last_num_pos = moves_str.rfind(f'{last_move_num}.')
        after_last_num = moves_str[last_num_pos + len(f'{last_move_num}.'):].strip()
        remaining_moves = [m for m in after_last_num.split() if m and not m.endswith('.')]
    
        total_plies = (last_move_num - 1) * 2 + len(remaining_moves)
        return total_plies
    
    return np.nan

df['move_count'] = df['pgn'].apply(extract_move_count)
avg_moves = df['move_count'].mean()
median_moves = df['move_count'].median()
min_moves = df['move_count'].min()
max_moves = df['move_count'].max()

print(f"\nAverage Full Moves: {avg_moves/2:.2f}")
print(f"Median Full Moves: {median_moves/2:.0f}")
print(f"Min Full Moves: {min_moves/2:.0f}")
print(f"Max Full Moves: {max_moves/2:.0f}")


# 4. Rating difference vs Result
print("\n4. RATING DIFFERENCE vs RESULT:")
df['rating_diff'] = df['WhiteElo'] - df['BlackElo']

# Create rating gap categories
def categorize_rating_gap(diff):
    abs_diff = abs(diff)
    if abs_diff < 50:
        return '0-49'
    elif abs_diff < 100:
        return '50-99'
    elif abs_diff < 200:
        return '100-199'
    elif abs_diff < 300:
        return '200-299'
    else:
        return '300+'

df['rating_gap_category'] = df['rating_diff'].apply(categorize_rating_gap)

def higher_rated_result(row):
    if row['rating_diff'] > 0:  # White is higher rated
        if row['Result'] == 1:
            return 'Higher rated wins'
        elif row['Result'] == 0:
            return 'Higher rated loses'
        else:
            return 'Draw'
    elif row['rating_diff'] < 0:  # Black is higher rated
        if row['Result'] == 0:
            return 'Higher rated wins'
        elif row['Result'] == 1:
            return 'Higher rated loses'
        else:
            return 'Draw'
    else:  # Equal rating
        return 'Equal rating'

df['higher_rated_result'] = df.apply(higher_rated_result, axis=1)

# Analyze by rating gap
print("\nResults by Rating Gap:")
for category in ['0-49', '50-99', '100-199', '200-299', '300+']:
    category_games = df[df['rating_gap_category'] == category]
    if len(category_games) > 0:
        print(f"\n  Rating Gap: {category} points")
        print(f"  Total games: {len(category_games)}")
        
        result_counts = category_games['higher_rated_result'].value_counts()
        for result, count in result_counts.items():
            percentage = (count / len(category_games)) * 100
            print(f"    {result}: {percentage:.2f}% ({count} games)")

print("\n\nOVERALL HIGHER RATED PLAYER DATA:")
higher_rated_stats = df[df['rating_diff'] != 0]['higher_rated_result'].value_counts()
games_with_diff = len(df[df['rating_diff'] != 0])
for result, count in higher_rated_stats.items():
    percentage = (count / games_with_diff) * 100
    print(f"{result}: {percentage:.2f}% ({count} games)")
