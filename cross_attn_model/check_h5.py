import h5py
import pandas as pd
import numpy as np

df = pd.read_csv('./data/PrimateDock_inter.csv')
df.columns = ['idx','voc_name', 'or_name', 'label']

# protein_embeddings = {}
# with h5py.File('./data/per_residue_embeddings_PrimateDock.h5', 'r') as f:
#     for key in f.keys():
#         protein_embeddings[key] = np.array(f[key])

ligand_embeddings = {}
with h5py.File('./data/per_atom_embeddings_PrimateDock.h5', 'r') as f:
    for key in f.keys():
        ligand_embeddings[key] = np.array(f[key])

for i in range(10):
    voc_name = df.iloc[i]['voc_name']
    or_name = df.iloc[i]['or_name']
    if str(voc_name) in ligand_embeddings:
        print(str(or_name))
