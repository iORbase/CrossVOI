# CrossVOI

## Introduction
CrossVOI is a deep learning model designed for predicting the functionality of human olfactory receptors. It enables rapid and efficient batch prediction of the interaction between ORs (olfactory receptors) and VOCs (volatile organic compounds). CrossVOI utilizes a self-attention module based on the Transformer architecture to extract features from the input sequences and employs cross-attention to fuse features from both OR and VOC. For more detailed information on CrossVOI, please refer to: "**AI-powered virtual screening improves function prediction of human olfactory receptors**" (to be published).

## Dependence

`subword_nmt==0.3.8`

`numpy==1.26.4`

`pandas==2.2.2`

`pytorch==2.4.1`

`scikit_learn==1.0.2`

`tqdm==4.66.5`

`transformers==4.19.2`

`tensorboard==2.17.0`

## Project Structure
**data**: Store the data file and perform the data preprocessing process.

&emsp;**data_preprocess.py**:  Data preprocessing process.

&emsp;**csv_file**: Dataset for training or testing.

**trans_bpe**: The main working path includes the model framework, training and testing sections.

&emsp;**model.py**: Model framework definition.

&emsp;**trainOlf.py**: Model training process.

&emsp;**testOlf.py**: Model testing process

&emsp;**run_train.sh**: Training scrtpt.

&emsp;**run_test.sh**: Testing script.

**trans_utils**: Utilities.

## Usage
### 1.Data preprocessing
Before proceeding with further functional predictions, you need to preprocess your data.

（1）Organize the experimental data into CSV format files, where OR, VOC, and their interaction relationships are saved as seq.csv, voc.csv, and inter.csv respectively. Please refer to the example files in the csv_file directory for the specific format of CSV files.

（2）Execute the preprocessing script (You can modify the corresponding parameters in the script to specify the name of the data file to be saved):

`python data_preprocess.py`

The processed data file will be saved in the data path, and you can read it later when training the model or making predictions.

### 2.Train the model using experimental data
Execute the training script (You can modify the corresponding parameters in the script to specify the name of the data file to be loaded):

`bash run_train.sh`

### 3.Utilize the trained model to predict the interaction relationships for the target OR-VOC pairs
Execute the training script (You can modify the corresponding parameters in the script to specify the name of the data file to be loaded):

`bash run_test.sh`




