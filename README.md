# CrossVOI

## Introduction
This repository contains the model code used for attention-based analysis of human olfactory receptors (ORs) in our study. The code implements the CrossVOI framework, which integrates a protein language model with a cross-attention mechanism to predict VOC-OR interactions and extract residue-level attention scores. These attention scores are further analyzed to explore the functional regions of ORs at both the family and individual levels. For more detailed information on CrossVOI, please refer to: "**Cross-attention and language models reveal the interpretability of functional predictions for the human olfactory receptor family**" (to be published).

## Project Structure
**data**: Store the data file and perform the data preprocessing process.

**attention_weights**: The folder used to store the extracted attention scores.

**models**: The model file saved after training.

**models.py**: Model structure definition file.

**train.py**: Training model script.

**predict.py**: Testing model script.

**extra_attention.py**: Extract the cross-attention scores and save them in a pkl file.

**extra_self_attention.py**: Extract the self-attention scores and save them in a pkl file.

## Note
Due to file size limitations, the preprocessed feature files and the protein language model checkpoint used for feature extraction are not included in this repository. To reproduce the analysis, please download the required language model separately and run the provided preprocessing scripts to generate the feature files. 

This repository is intended for research purposes — specifically, to facilitate the exploration and interpretation of attention distribution patterns in olfactory receptors. It is not designed as a production-ready software tool or a general-purpose prediction service. As such, the code may not be as rigorously organized, documented, or optimized as a software engineering project would be. Users interested in reproducibility are encouraged to refer to the methodology section of the accompanying paper for detailed experimental settings.

Additional scripts for statistical analysis and visualization will be added in the near future. The current release includes the core model code and basic extraction utilities. The supplementary notebooks and plotting scripts used for generating the figures in the paper are not yet included but will be uploaded soon.




