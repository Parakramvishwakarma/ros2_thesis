import pandas as pd
import matplotlib.pyplot as plt

# Function to read and process a CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    print(file_path, df['r'].mean())
    df['cumulative_l'] = df['l'].cumsum()
    return df

# List of CSV files
csv_files = ['./results/DDPG_training_results.csv', './results/SAC_training_results.csv', './results/TD3_training_results.csv']
label = ['DDPG', 'SAC', 'TD3']

# Colors for the plots
colors = ['r', 'g', 'b']

# Plotting
plt.figure(figsize=(10, 6))

for i, file in enumerate(csv_files):
    df = process_csv(file)
    # df['r_ma'] = df['r'].rolling(window=10).mean()
    plt.plot(df['cumulative_l'].values, df['r'].values, color=colors[i], label=label[i])

plt.xlabel('Timestep')
plt.ylabel('Total return')
plt.title('Total Return comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison.png')
plt.show()
plt.close()