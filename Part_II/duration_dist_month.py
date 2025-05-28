import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load the Excel File ===
file_path = 'Book2.xlsx'  # Ensure the file is in the same directory
df = pd.read_excel(file_path)

# === Step 2: Preprocess the Data ===
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create a column for the sign of the spread
df['Spread_Sign'] = df['Spread'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# === Step 3: Calculate Durations ===
duration_data = []
current_duration = 0
waiting_for_positive = False
start_date = None

for i in range(1, len(df)):
    prev_sign = df.loc[i - 1, 'Spread_Sign']
    current_sign = df.loc[i, 'Spread_Sign']
    
    if prev_sign < 0:
        if not waiting_for_positive:
            start_date = df.loc[i - 1, 'Date']
        waiting_for_positive = True
        current_duration += 1
    elif waiting_for_positive:
        current_duration += 1

    if waiting_for_positive and current_sign > 0:
        end_date = df.loc[i, 'Date']
        duration_data.append({
            'Duration': current_duration,
            'Month': end_date.month,
            'Month_Name': end_date.strftime('%B'),
            'Year': end_date.year,
            'End_Date': end_date
        })
        current_duration = 0
        waiting_for_positive = False

# Convert to DataFrame
duration_df = pd.DataFrame(duration_data)

# === Step 4: Summary Statistics ===
mean_duration = duration_df['Duration'].mean()
max_duration = duration_df['Duration'].max()
min_duration = duration_df['Duration'].min()
median_duration = duration_df['Duration'].median()
mode_duration = duration_df['Duration'].mode().iloc[0] if not duration_df['Duration'].mode().empty else None

print("=== Duration Statistics ===")
print(f"Mean Duration   : {mean_duration:.2f} days")
print(f"Max Duration    : {max_duration} days")
print(f"Min Duration    : {min_duration} days")
print(f"Median Duration : {median_duration} days")
print(f"Mode Duration   : {mode_duration} days")

# === Step 5: Plot Distribution by Month ===
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='Month_Name',
    y='Duration',
    data=duration_df,
    order=['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']
)
plt.title('Distribution of Duration Until Spread Turns Positive by Month')
plt.xlabel('Month')
plt.ylabel('Duration (days)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# === Step 6: Plot Distribution by Year ===
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Year',
    y='Duration',
    data=duration_df,
    palette='Set2'
)
plt.title('Distribution of Duration Until Spread Turns Positive by Year')
plt.xlabel('Year')
plt.ylabel('Duration (days)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 7 (Improved): Frequency Plot Without Outliers ===
from collections import Counter

# Count frequency of each duration
duration_counts = duration_df['Duration'].value_counts().sort_index()
mode_values = duration_df['Duration'].mode()

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(duration_counts.index, duration_counts.values, color='skyblue', edgecolor='black')

# Highlight the mode(s)
for bar in bars:
    if bar.get_x() in mode_values.values:
        bar.set_color('orange')

# Annotate mode bars
for mode_val in mode_values:
    plt.text(mode_val, duration_counts[mode_val] + 0.5, 'Mode', ha='center', color='orange', weight='bold')

plt.title('Frequency of Duration Values (Days to Positive Spread)')
plt.xlabel('Duration (days)')
plt.ylabel('Frequency (count)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# Calculate IQR to remove outliers
Q1 = duration_df['Duration'].quantile(0.25)
Q3 = duration_df['Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
filtered_df = duration_df[(duration_df['Duration'] >= lower_bound) & (duration_df['Duration'] <= upper_bound)]

# Count frequency of each duration
duration_counts = filtered_df['Duration'].value_counts().sort_index()
mode_values = filtered_df['Duration'].mode()

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(duration_counts.index, duration_counts.values, color='skyblue', edgecolor='black')

# Highlight mode(s)
for bar in bars:
    if bar.get_x() in mode_values.values:
        bar.set_color('orange')

# Annotate mode bars
for mode_val in mode_values:
    plt.text(mode_val, duration_counts[mode_val] + 0.5, 'Mode', ha='center', color='orange', weight='bold')

plt.title('Frequency of Duration Values (Without Outliers)')
plt.xlabel('Duration (days)')
plt.ylabel('Frequency (count)')
plt.xticks(duration_counts.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

