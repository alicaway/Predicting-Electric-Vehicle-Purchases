"""
Preprocessing script for train.csv to predict Will_Buy_EV.
"""
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
print("Loading train.csv...")
df = pd.read_csv('train.csv')
print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")

# Step 1: Encode Target Variable
print("\n=== Encoding Target Variable ===")
df['Will_Buy_EV'] = df['Will_Buy_EV'].map({'Yes': 1, 'No': 0})
print(f"Target: 0={sum(df['Will_Buy_EV']==0):,}, 1={sum(df['Will_Buy_EV']==1):,}")

# Step 2: Identify column types
numeric_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
binary_cols = ['Home_Charging_Possible', 'Subsidy_Available']
ordinal_cols = ['Range_Anxiety_Level']
nominal_cols = ['Gender', 'City_Type', 'Current_Car_Type']

# Step 3: Encode Binary columns
print("\n=== Binary Encoding ===")
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    print(f"  {col}: Yes->1, No->0")

# Step 4: Ordinal Encoding for Range_Anxiety_Level
print("\n=== Ordinal Encoding ===")
range_anxiety_order = {'Low': 0, 'Medium': 1, 'High': 2}
df['Range_Anxiety_Level'] = df['Range_Anxiety_Level'].map(range_anxiety_order)

# Step 5: Gender Label Encoding
print("\n=== Gender Encoding ===")
gender_order = {'Female': 0, 'Male': 1, 'Other': 2}
df['Gender'] = df['Gender'].map(gender_order)

# Step 6: One-Hot Encoding for nominal columns
print("\n=== One-Hot Encoding ===")
city_dummies = pd.get_dummies(df['City_Type'], prefix='City_Type', drop_first=True)
df = pd.concat([df, city_dummies], axis=1)
df = df.drop('City_Type', axis=1)
print(f"  City_Type: {city_dummies.shape[1]} dummies (dropped Rural)")

car_dummies = pd.get_dummies(df['Current_Car_Type'], prefix='Car_Type', drop_first=True)
df = pd.concat([df, car_dummies], axis=1)
df = df.drop('Current_Car_Type', axis=1)
print(f"  Current_Car_Type: {car_dummies.shape[1]} dummies (dropped Hatchback)")

# Step 7: Scale Numeric Features
print("\n=== Scaling Numeric Features ===")
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), 
                         columns=[f"{c}_scaled" for c in numeric_cols], index=df.index)
df = pd.concat([df, df_scaled], axis=1)
df = df.drop(columns=numeric_cols)
print(f"  Scaled {len(numeric_cols)} numeric columns")

# Step 8: Save Processed Data
print("\n=== Saving Processed Data ===")
id_col = 'id'
target_col = 'Will_Buy_EV'
final_features = [c for c in df.columns if c not in [id_col, target_col]]
df_final = df[[id_col] + final_features + [target_col]]
df_final.to_csv('train_processed.csv', index=False)
print(f"✓ Saved train_processed.csv: {df_final.shape}")

# Save preprocessing config
config = {
    'encodings': {
        'Target': {'Yes': 1, 'No': 0},
        'Binary': {'Home_Charging_Possible': {'Yes': 1, 'No': 0}, 'Subsidy_Available': {'Yes': 1, 'No': 0}},
        'Ordinal': {'Range_Anxiety_Level': {'Low': 0, 'Medium': 1, 'High': 2}},
        'Label': {'Gender': {'Female': 0, 'Male': 1, 'Other': 2}}
    },
    'one_hot_dropped': {'City_Type': 'Rural', 'Current_Car_Type': 'Hatchback'},
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'final_feature_count': len(final_features),
    'class_imbalance_ratio': float((df_final['Will_Buy_EV']==0).sum() / (df_final['Will_Buy_EV']==1).sum())
}
with open('preprocessing_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✓ Saved preprocessing_config.json")
print("\nPREPROCESSING COMPLETE")