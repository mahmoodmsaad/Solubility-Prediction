# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

import lazypredict
from lazypredict.Supervised import LazyRegressor
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# Load dataset
data = pd.read_csv(r"D:\Aqueous solubility_9945.csv")
data.dropna(inplace=True)

# Data visualization
sns.set_theme()
sns.histplot(data=data, x="logS", binwidth=1)
sns.boxplot(data=data, x='logS')

# Generate canonical SMILES
def canonical_smiles(smiles):
    return [Chem.CanonSmiles(smi) for smi in smiles]

data["SMILES"] = canonical_smiles(data.SMILES)

# Remove duplicate SMILES
data_cleaned = data.drop_duplicates(subset=['SMILES'])

# Load and preprocess test set
test_set = pd.read_csv(r"d:/Drug_Like_Solubility _100.csv")
test_set["SMILES"] = canonical_smiles(test_set.SMILES)

data_cleaned_final = data_cleaned[~data_cleaned['SMILES'].isin(test_set.SMILES)]
test_set = test_set[test_set['LogS exp (mol/L)'].between(-7.5, 1.7)]

# Calculate RDKit descriptors
def calculate_descriptors(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    return [calc.CalcDescriptors(Chem.AddHs(mol)) for mol in mols], calc.GetDescriptorNames()

desc_values, desc_names = calculate_descriptors(data_cleaned_final['SMILES'])
desc_df = pd.DataFrame(desc_values, columns=desc_names)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(desc_df, data_cleaned_final.logS, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Train using LazyRegressor
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, random_state=42)
models, predictions = lazy_reg.fit(X_train_scaled, X_valid_scaled, y_train, y_valid)

# Train final model
best_model = LGBMRegressor(n_estimators=1150, max_depth=26, learning_rate=0.04, random_state=42)
best_model.fit(X_train_scaled, y_train)
y_preds = best_model.predict(X_valid_scaled)

# Plotting function
def plot_results(actual, predicted, title):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    plt.figure(figsize=(8,6))
    sns.regplot(x=predicted, y=actual, line_kws={"lw": 2, 'ls': '--', 'color': 'red', "alpha": 0.7})
    plt.title(title, color='red')
    plt.xlabel('Predicted logS (mol/L)', color='blue')
    plt.ylabel('Experimental logS (mol/L)', color='blue')
    plt.xlim(-8, 1)
    plt.grid(alpha=0.3)
    plt.legend(handles=[mpatches.Patch(label=f"R2={r2:.2f}"), mpatches.Patch(label=f"RMSE={rmse:.2f}")])
    plt.show()

plot_results(y_valid, y_preds, 'Validation data')

# Prepare test set descriptors
test_desc_values, _ = calculate_descriptors(test_set['SMILES'])
test_desc_df = pd.DataFrame(test_desc_values, columns=desc_names)
X_test_scaled = scaler.transform(test_desc_df)

# Predict test set
y_test_preds = best_model.predict(X_test_scaled)
plot_results(test_set['LogS exp (mol/L)'], y_test_preds, 'Test data')

# Save model and scaler
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
