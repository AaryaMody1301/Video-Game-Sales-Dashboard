import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load the dataset
df = pd.read_csv(os.path.join(parent_dir, 'vgchartz-2024.csv'))

# Check for PS5 data
ps5_games = df[df['console'] == 'PS5']
print(f"PS5 games in dataset: {len(ps5_games)}")

# Print some PS5 games if they exist
if len(ps5_games) > 0:
    print("\nSample PS5 games:")
    print(ps5_games[['title', 'publisher', 'total_sales']].head(5))
else:
    print("No PS5 games found in the dataset.")

# Check if there might be alternative ways PS5 is represented
print("\nUnique consoles in dataset:")
consoles = sorted(df['console'].unique())
print(consoles)

# Check for similar name patterns
ps_consoles = [console for console in consoles if 'PS' in str(console)]
print("\nPlayStation consoles:")
print(ps_consoles) 