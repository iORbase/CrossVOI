# -*- coding: utf-8 -*-
import sys
sys.path.append("../..")
sys.path.append("../../..")
import random
import os

from torch.utils.tensorboard import SummaryWriter
from Human.trans_utils.layers_utils import *
from dataset_builder_util import *
from argument_parser import *
from Human.trans_utils.utils import *
from model import Trans

from sklearn import metrics
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
import datetime

#RATIO = 1.0

def test_model(testset, model, BATCH_SIZE, device,
                    prot_matrix, prot_list, prot_set,
                    smile_matrix, smile_list, smile_set
               ):
    start = 0
    done = False
    preds = []
    residueF, atomF, ec50s = testset[0], testset[1], testset[2]
    testnum = ec50s.shape[0]

    while not (done or start == testnum):
        end = start + BATCH_SIZE

        if end > testnum:
            done = True
            end = end % testnum
            index = torch.arange(start=start, end=testnum)

        else:
            index = torch.arange(start=start, end=end)

        if smile_matrix is not None:
            smile_matrix_test = get_matrix(
                    gpu = device,
                    matrix=smile_matrix,
                    list=smile_list,
                    Set=smile_set[index])
        else:
            smile_matrix_test = None

        if prot_matrix is not None:
            prot_matrix_test = get_matrix(
                gpu = device,
                matrix=prot_matrix,
                list=prot_list,
                Set=prot_set[index])
        else:
            prot_matrix_test = None

        with torch.no_grad():
            pred = model(atomF[index], residueF[index], smile_matrix_test, prot_matrix_test)
        start = end

        preds.append((pred>0.5).int())

    preds = torch.concatenate(preds, dim=0).cpu().numpy()
    #print(preds)
    ec50s = ec50s.cpu().numpy()

    # 计算F1-score
    f1_score = metrics.f1_score(ec50s, preds, average='macro')
    print(f"F1-score: {f1_score}")
    # 计算AUC
    auc_value = roc_auc_score(ec50s, preds)
    print(f"auc: {auc_value}")
    
    accuracy = accuracy_score(ec50s, preds)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(ec50s, preds)
    print(f"Precision: {precision}")

    recall = recall_score(ec50s, preds)
    print(f"Recall: {recall}")

    false_positives = np.sum((ec50s == 0) & (preds == 1))
    actual_negatives = np.sum(ec50s == 0)
    fpr = false_positives / actual_negatives if actual_negatives > 0 else 0
    print("False Positive Rate (FPR):", fpr)

    precision, recall, _ = precision_recall_curve(ec50s, preds, pos_label=1)
    auprc = auc(recall, precision)
    print("AUPRC:", auprc)

    #return f1_score#, auc_value

def build_dtitr_model(FLAGS, prot_trans_depth, smiles_trans_depth, cross_attn_depth,
                      prot_trans_heads, smiles_trans_heads, cross_attn_heads,
                      prot_d_ff, smiles_d_ff, d_model, dropout_rate,
                      out_mlp_depth, out_mlp_units):

    dtitr_model = Trans(
        d_model=d_model,
        prot_atten_layers=prot_trans_depth,
        prot_atten_heads=prot_trans_heads,
        smile_atten_layers=smiles_trans_depth,
        smile_atten_heads=smiles_trans_heads,
        cross_atten_layers=cross_attn_depth,
        cross_atten_heads=cross_attn_heads,
        prot_d_ff = prot_d_ff,
        smiles_d_ff = smiles_d_ff,
        x1_num_heads=smiles_trans_heads,
        x2_num_heads=prot_trans_heads,
        x1_d_ff =smiles_d_ff,
        x2_d_ff =prot_d_ff,
        mlp_depth=out_mlp_depth,
        mlp_units=out_mlp_units,
        dropout_rate=dropout_rate,
        lr=FLAGS.optimizer_fn[0],
        betas=[FLAGS.optimizer_fn[1], FLAGS.optimizer_fn[2]],
        eps=FLAGS.optimizer_fn[3],
        weight_decay=FLAGS.optimizer_fn[4],
        atomemb=FLAGS.smiles_dict_len, resemb=FLAGS.protein_dict_bpe_len,
        is_grad_dec = FLAGS.is_grad_dec, step=FLAGS.step, gamma=FLAGS.gamma,
        device=FLAGS.gpu
    )

    return dtitr_model

def run_test_model(FLAGS):
    BATCH_SIZE = FLAGS.batch_size

    protein_data, smiles_data, ec50_values, \
    prot_matrix, smile_matrix, \
    smile_list, prot_list, SMILES, PROTS = dataset_builder(FLAGS.data_path).\
        transform_dataset(
                protein_column=0,
                smiles_column=3,
                ec50_column=4,
                is_prot_bpe=FLAGS.is_prot_bpe,
                prot_max_len=FLAGS.protein_len,
                smiles_max_len=FLAGS.smiles_len,
                is_prot_add_positional_info=FLAGS.is_prot_add_positional_info,
                is_smile_add_positional_info=FLAGS.is_smile_add_positional_info
                )

    if FLAGS.is_prot_bpe:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
    else:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

    smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

    protein_data = protein_data.to(FLAGS.gpu)
    smiles_data = smiles_data.to(FLAGS.gpu)
    ec50_values = ec50_values.to(FLAGS.gpu)

    prot_test = protein_data
    smiles_test = smiles_data
    ec50_test = ec50_values

    if prot_matrix is not None:
        PROTS_test = PROTS

    else:
        PROTS_test = None

    if smile_matrix is not None:
        SMILES_test = SMILES

    else:
        SMILES_test = None

    dtitr_model = build_dtitr_model(FLAGS=FLAGS,
                                    prot_trans_depth=FLAGS.prot_transformer_depth,
                                    smiles_trans_depth=FLAGS.smiles_transformer_depth,
                                    cross_attn_depth=FLAGS.cross_block_depth,
                                    prot_trans_heads=FLAGS.prot_transformer_heads,
                                    smiles_trans_heads=FLAGS.smiles_transformer_heads,
                                    cross_attn_heads=FLAGS.cross_block_heads,
                                    prot_d_ff=FLAGS.prot_ff_dim,
                                    smiles_d_ff=FLAGS.smiles_ff_dim,
                                    d_model=FLAGS.d_model,
                                    dropout_rate=FLAGS.dropout_rate,
                                    out_mlp_depth=FLAGS.out_mlp_depth,
                                    out_mlp_units=FLAGS.out_mlp_units)

    state_dict = torch.load('./saved_models/model_checkpoint.pt')
    dtitr_model.load_state_dict(state_dict)

    dtitr_model.eval()
    test_model([prot_test, smiles_test, ec50_test],
                                    dtitr_model, BATCH_SIZE, FLAGS.gpu,
                                    prot_matrix, prot_list, PROTS_test,
                                    smile_matrix, smile_list, SMILES_test
                                        )

if __name__ == "__main__":
    FLAGS = argparser()

    FLAGS.data_path = {'data': os.path.abspath('../data/data.npy'),
                       'prot_matrix': os.path.abspath('../data/struc_all/res_distmat.npy'),
                       'prot_name': os.path.abspath('../data/struc_all/prot_name.npy'),
                       'prot_ser': os.path.abspath('../data/struc_all/res_ser.npy'),
                       'prot_dic': os.path.abspath('../data/dictionary/davis_prot_dictionary.txt'),
                       'smiles_dic': os.path.abspath('../data/dictionary/davis_smiles_dictionary.txt'),
                       'smiles_h5': os.path.abspath('../data/per_atom_embeddings.h5'),
                       'prot_h5': os.path.abspath('../data/per_residue_embeddings.h5'),
                       'prot_bpe': [os.path.abspath('../data/dictionary/protein_codes_uniprot.txt'),
                                    os.path.abspath('../data/dictionary/subword_units_map_uniprot.csv')],
                       'smiles_bpe': [os.path.abspath('../data/dictionary/drug_codes_chembl.txt'),
                                      os.path.abspath('../data/dictionary/subword_units_map_chembl.csv')],
                       'clusters': glob.glob('../data/davis/clusters/*')}

    FLAGS.log_path = "log/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    run_test_model(FLAGS)
