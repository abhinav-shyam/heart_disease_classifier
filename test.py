import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve

df = pd.read_csv('heart.csv')
random_state = 42

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

missing_values = df.isnull().sum(axis=0)
if missing_values.any():
    print(f"\nMissing values:")
    print(missing_values[missing_values > 0])  # displays the columns which have missing values, and their count
else:
    print("\nNo missing values found!")

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
print(f"\n--- CATEGORICAL FEATURES ANALYSIS ---")
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} unique values:")
        print(df[col].value_counts())


categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

features = [col for col in df_encoded.columns if col != 'HeartDisease']
X = df_encoded[features]
y = df_encoded['HeartDisease']

print(f"Number of features after encoding: {len(features)}")
print(f"Features: {features}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train target distribution: {np.bincount(y_train)}")
print(f"Test target distribution: {np.bincount(y_test)}")

plt.style.use('default')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Heart Disease Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Target distribution
ax1 = axes[0, 0]
target_counts = df['HeartDisease'].value_counts()
ax1.pie(target_counts.values, labels=['No Disease', 'Heart Disease'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Target Variable Distribution')

# 2. Age distribution by target
ax2 = axes[0, 1]
df.boxplot(column='Age', by='HeartDisease', ax=ax2)
ax2.set_title('Age Distribution by Heart Disease')
ax2.set_xlabel('Heart Disease (0=No, 1=Yes)')

# 3. Chest pain type vs heart disease
ax3 = axes[0, 2]
chest_pain_crosstab = pd.crosstab(df['ChestPainType'], df['HeartDisease'])
chest_pain_crosstab.plot(kind='bar', ax=ax3, rot=45)
ax3.set_title('Chest Pain Type vs Heart Disease')
ax3.legend(['No Disease', 'Heart Disease'])

# 4. Correlation heatmap
ax4 = axes[1, 0]
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
ax4.set_title('Feature Correlation Heatmap')

# 5. MaxHR vs Age colored by target
ax5 = axes[1, 1]
scatter = ax5.scatter(df['Age'], df['MaxHR'], c=df['HeartDisease'], cmap='viridis', alpha=0.6)
ax5.set_xlabel('Age')
ax5.set_ylabel('Max Heart Rate')
ax5.set_title('Max Heart Rate vs Age')
plt.colorbar(scatter, ax=ax5, label='Heart Disease')

# 6. Exercise Angina vs Heart disease
ax6 = axes[1, 2]
exercise_crosstab = pd.crosstab(df['ExerciseAngina'], df['HeartDisease'])
exercise_crosstab.plot(kind='bar', ax=ax6)
ax6.set_title('Exercise Angina vs Heart Disease')
ax6.set_xticklabels(['No', 'Yes'], rotation=0)
ax6.legend(['No Disease', 'Heart Disease'])

plt.tight_layout()
# plt.savefig('results/eda_analysis.png', dpi=300, bbox_inches='tight')
# plt.close()
plt.show()



