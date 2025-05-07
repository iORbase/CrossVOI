# -*- coding: utf-8 -*-
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=str,
        default='cuda:0'
    )

    parser.add_argument(
        '--is_prot_bpe',
        type=int,
        default=1)

    parser.add_argument(
        '--is_grad_dec',
        type=int,
        default=0)

    parser.add_argument(
        '--step',
        type=int,
        default=300)

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.995)

    parser.add_argument(
        '--is_prot_add_positional_info',
        type=int,
        default=0
    )

    parser.add_argument(
        '--is_smile_add_positional_info',
        type=int,
        default=1
    )

    parser.add_argument(
        '--protein_dict_bpe_len',
        type=int,
        default=16693,
        help='Protein BPE Dictionary Length')

    parser.add_argument(
        '--smiles_len',
        type=int,
        default=200,
        help='SMILES Strings Max Length')

    parser.add_argument(
        '--smiles_dict_len',
        type=int,
        default=22,
        help='SMILES Char Dictionary Length')

    parser.add_argument(
        '--protein_len',
        type=int,
        default=400,
        help='protein Strings Max Length')

    parser.add_argument(
        '--protein_dict_len',
        type=int,
        default=20,
        help='protein Char Dictionary Length')

    parser.add_argument(
        '--prot_transformer_depth',
        type=int,
        default=4,
        help='Protein Transformer Encoder Depth')

    parser.add_argument(
        '--smiles_transformer_depth',
        type=int,
        default=3,
        help='SMILES Transformer Encoder Depth')

    parser.add_argument(
        '--cross_block_depth',
        type=int,
        default=1,
        help='Cross Attention Block Depth')

    parser.add_argument(
        '--d_model',
        type=int,
        default=128,
        help='Emb Size')

    parser.add_argument(
        '--prot_transformer_heads',
        type=int,
        default=4,
        help='Protein Transformer Encoder Heads')

    parser.add_argument(
        '--smiles_transformer_heads',
        type=int,
        default=4,
        help='SMILES Transformer Encoder Heads')

    parser.add_argument(
        '--cross_block_heads',
        type=int,
        default=4,
        help='Cross Attention Block')

    parser.add_argument(
        '--prot_ff_dim',
        type=int,
        default=512,
        help='Protein PosWiseFF Dim')

    parser.add_argument(
        '--smiles_ff_dim',
        type=int,
        default=512,
        help='SMILES PosWiseFF Dim')

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.1,
        help='Dropout Rate')

    parser.add_argument(
        '--out_mlp_depth',
        type=int,
        default=3,
        help='Output MLP Block Depth')

    parser.add_argument(
        '--out_mlp_units',
        type=list,
        default=[512, 1024, 512],
        help='Output MLP Block Hidden Neurons')

    parser.add_argument(
        '--optimizer_fn',
        type=list,
        default=[1e-4, 0.9, 0.999, 1e-8, 1e-5],
        help='Optimizer Function Parameters')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch Dim')

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
        help='Number of Epochs')

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random Seed')

    FLAGS, _ = parser.parse_known_args()
    return FLAGS
