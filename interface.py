import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from matplotlib import pyplot as plt

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to calculate descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(Chem.AddHs(mol))
    return np.array(descriptors).reshape(1, -1)

# Streamlit app
st.title('Aqueous Solubility Prediction')
smiles_input = st.text_input('Enter SMILES notation of the compound:')
if smiles_input:
    descriptors = calculate_descriptors(smiles_input)
    descriptors_scaled = scaler.transform(descriptors)
    prediction = model.predict(descriptors_scaled)
    st.write(f'Predicted logS (mol/L): {prediction[0]:.2f}')

# File uploader for batch prediction
uploaded_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    batch_data['Descriptors'] = batch_data['SMILES'].apply(calculate_descriptors)
    batch_descriptors = np.vstack(batch_data['Descriptors'].values)
    batch_descriptors_scaled = scaler.transform(batch_descriptors)
    batch_predictions = model.predict(batch_descriptors_scaled)
    batch_data['Predicted logS'] = batch_predictions
    st.write(batch_data[['SMILES', 'Predicted logS']])
