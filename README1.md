# NeuroPred-PSCG

This repository contains code for "**NeuroPpred-PSCG: A multimodal framework for neuropeptide prediction via bidirectional cross-attention and gated convolution of ProtT5 and structural features**.
![image](https://github.com/xinyizhang186/NeuroPpred-PSCG/blob/main/model.png)
Please see our manuscript for more details.

## 1 Requirements

Before running, please create a new environment using this command:

```python
conda create -n NeuroPred-PSCG python=3.10
conda activate NeuroPred-PSCG
```

Next, run the following command to install the required packages:

```python
cd NeuroPred-PSCG
pip install -r requirements.txt --no-cache-dir
```

# 2 Running

- **`dataset/`**: Ccomprises two folders, `NeuroP_data` and `NeuroP_feature`. Among these, the imbalanced dataset (*3755.txt*) comprises labels distributed as [1]409 + [0]2290+ [1]45+ [0]252+ [1]115+[0]644.
- **`protT5/`**: Extracts ProtT5-based embeddings from protein sequences.
- **`demo/`**: Provides example notebooks (train.ipynb and predict.ipynb) for model training and inference.
- **`util/`**:
- `csv_to_fasta.py`：Converts CSV files into FASTA format.
- `data_loader.py`: Handles data loading and preprocessing.
- `util_metric.py`: Computes and evaluates model performance metrics.

In addition, the main scripts and files are as follows:

- **`requirements.txt`**: Lists all dependencies and their versions for quick environment setup.
- **`model.py`**: Defines the NeuroPred-PSCG model architecture.
- **`train.py`**: Training script can be run directly to train the model.
- **`predict.py`**: Testing script can be run directly to evaluate the model and reproduce results.

If you aim to train the NeuroPred-PSCG model or use a successfully trained model for testing, please run the following code:

```python
python train.py 
python predict.py
```

# 3 Predict

To predict neuropeptides from your FASTA sequences, please follow these steps:

1. **Generate Features**: First, encode your input sequences into ProtT5 embeddings and 8-state secondary structure (SS8) features. The scripts for this process are located in the `ProtT5` directory.
2. **Run Prediction**: Once the features are generated, specify their path within the `predict.py` script and execute it to obtain the prediction labels.

Refer to **`demo/predict.ipynb `** for a usage example.

