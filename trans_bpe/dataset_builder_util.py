# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import torch
import pandas as pd
import numpy as np
import json
import re
from operator import itemgetter
from subword_nmt.apply_bpe import BPE
import codecs
import glob

from rdkit import Chem
from rdkit.Chem import AllChem

class dataset_builder():
    def __init__(self, data_path, **kwargs):
        super(dataset_builder, self).__init__(**kwargs)
        self.data_path = data_path

    def get_data(self):
        dataset = np.load(self.data_path['data'])
        prot_dictionary = json.load(open(self.data_path['prot_dic']))
        smiles_dictionary = json.load(open(self.data_path['smiles_dic']))
        clusters = []
        bpe_codes_prot = ''
        bpe_codes_map_prot = ''
        bpe_codes_smiles = ''
        bpe_codes_map_smiles = ''
        prot_matrix = ''
        prot_ser = ''

        for i in self.data_path['clusters']:
            if 'test' in i:
                clusters.append(('test', pd.read_csv(i, header=None)))
            else:
                clusters.append(('train', pd.read_csv(i, header=None)))

        if self.data_path['prot_bpe'] != '':
            bpe_codes_prot = codecs.open(self.data_path['prot_bpe'][0])
            bpe_codes_map_prot = pd.read_csv(self.data_path['prot_bpe'][1])

        if self.data_path['smiles_bpe'] != '':
            bpe_codes_smiles = codecs.open(self.data_path['smiles_bpe'][0])
            bpe_codes_map_smiles = pd.read_csv(self.data_path['smiles_bpe'][1])

        if self.data_path['prot_matrix'] != '':
            prot_matrix = np.load(self.data_path['prot_matrix'], allow_pickle=True)
            prot_ser = np.load(self.data_path['prot_name'], allow_pickle=True)

        return (dataset, prot_dictionary, smiles_dictionary, clusters, bpe_codes_prot, bpe_codes_map_prot,
                bpe_codes_smiles, bpe_codes_map_smiles, prot_matrix, prot_ser)

    def data_conversion(self, data, dictionary, max_len):
        keys = list(i for i in dictionary.keys() if len(i) > 1)

        if len(keys) == 0:
            data = pd.DataFrame([list(i) for i in data])
        else:
            char_list = []
            for i in data:
                positions = []
                for j in keys:
                    positions.extend([(k.start(), k.end() - k.start()) for k in re.finditer(j, i)])

                positions = sorted(positions, key=itemgetter(0))

                if len(positions) == 0:
                    char_list.append(list(i))

                else:
                    new_list = []
                    j = 0
                    positions_start = [k[0] for k in positions]
                    positions_len = [k[1] for k in positions]

                    while j < len(i):
                        if j in positions_start:
                            new_list.append(str(i[j] + i[j + positions_len[positions_start.index(j)] - 1]))
                            j = j + positions_len[positions_start.index(j)]
                        else:
                            new_list.append(i[j])
                            j = j + 1
                    char_list.append(new_list)

            data = pd.DataFrame(char_list)
        data.replace(dictionary, inplace=True)
        data = data.fillna(0)
        
        if len(data.iloc[0, :]) == max_len:
            return data
        else:
            zeros_array = np.zeros(shape=(len(data.iloc[:, 0]), max_len - len(data.iloc[0, :])))
            data = np.concatenate([data, pd.DataFrame(zeros_array)], axis=1)
            return data

    def encoding_bpe(self, data, codes, codes_map, max_len):
        bpe = BPE(codes, merges=-1, separator='')
        idx2word = codes_map['index'].values
        words2idx = dict(zip(idx2word, range(0, len(idx2word))))

        vectors = []

        for i in data:
            t1 = bpe.process_line(i).split()  # split
            try:
                i1 = np.asarray([words2idx[j] + 1 for j in t1])  # index
            except:
                i1 = np.array([0])

            l = len(i1)

            if l < max_len:
                k = np.pad(i1, (0, max_len - l), 'constant', constant_values=0)
            else:
                k = i1[:max_len]
            vectors.append(torch.IntTensor(k[None, :]))

        return torch.concat(vectors, dim=0)

    def transform_dataset(self,
                          protein_column, smiles_column, ec50_column,
                          is_prot_bpe, prot_max_len, smiles_max_len,
                          is_prot_add_positional_info,
                          is_smile_add_positional_info
                          ):

        DATA = self.get_data()
        SMILES = DATA[0][:, smiles_column]
        PROTS = DATA[0][:, protein_column]

        #prot_name = DATA[0][:, 0]
        #smile_name = DATA[0][:, 2]
        #name = np.array([str(x) + '@' + str(y) for x,y in zip(prot_name, smile_name)])

        if is_prot_bpe:
            protein_data = self.encoding_bpe(DATA[0][:, protein_column + 1], DATA[4],
                                            DATA[5], prot_max_len)
        else:
            protein_data = torch.IntTensor(np.array(self.data_conversion(DATA[0][:, protein_column + 1],
                                            DATA[1], prot_max_len)))

        smiles_data = torch.IntTensor(np.array(self.data_conversion(SMILES,
                                            DATA[2], smiles_max_len)))

        ec50_values = DATA[0][:,ec50_column].astype('float32')
        ec50_values = torch.FloatTensor(np.array(ec50_values)[:,None])

        prot_matrix = None
        smile_matrix = None
        smile_list = None
        prot_list = None

        if is_prot_add_positional_info:
            prot_list = list(DATA[9])
            prot_matrix = []
            for matrix in DATA[8]:
                padding_matrix = np.zeros((prot_max_len + 1, prot_max_len + 1))
                L = matrix.shape[0]

                matrix = 1/(matrix + 1)
                padding_matrix[1:L + 1, 1:L + 1] = matrix
                padding_matrix[0, 1:L + 1] += 1 / L
                prot_matrix.append(padding_matrix)

        if is_smile_add_positional_info:
            smile_list = []
            smile_matrix = []

            for smile in SMILES:
                if not smile in smile_list:
                    smile_list.append(smile)
                    mol = Chem.MolFromSmiles(smile)
                    AllChem.EmbedMolecule(mol)
                    
                    matrix = AllChem.Get3DDistanceMatrix(mol)
                    matrix = np.exp(-matrix)

                    padding_matrix = np.zeros((smiles_max_len + 1, smiles_max_len + 1))
                    L = matrix.shape[0]
                    padding_matrix[1:L+1, 1:L+1] = matrix
                    padding_matrix[0, 1:L + 1] += 1 / L
                    smile_matrix.append(padding_matrix)

        return protein_data, smiles_data, ec50_values, \
                prot_matrix, smile_matrix, \
                smile_list, prot_list, SMILES, PROTS
