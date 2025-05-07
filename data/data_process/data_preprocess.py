import numpy as np
import pandas as pd

seq = pd.read_csv('./csv_file/seq.csv', header=None)
seq.columns = ['idx', 'OR', 'seq']
voc = pd.read_csv('./csv_file/voc.csv', header=None)
voc.columns = ['idx', 'CID', 'SMILES']
inter = pd.read_csv('./csv_file/inter.csv', header=None)

idx = 0
result = []

for row in inter.itertuples():
    OR = str(row[3])
    sequence = seq.loc[seq['OR'] == row[3]]['seq'].iloc[0]
    CID = str(row[2])
    SMILES = voc.loc[voc['CID'] == row[2]]['SMILES'].iloc[0]
    tag = str(row[4])
    r = [OR, sequence, CID, SMILES, tag]
    result.append(r)
    print(idx)
    idx += 1

result = np.array(result)
np.save('data', result)
