import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'Book2.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract month name (or number if preferred)
df['Month'] = df['Date'].dt.month_name()

# Optional: To maintain correct month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

# Plotting
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Month', y='Spread')
plt.title('Monthly Distribution of Spread')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
