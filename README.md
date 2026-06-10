# HHAT-Net for Motor Imagery EEG Classification

This repository contains HHAT-Net, a lightweight hybrid EEG decoding model inspired by TCFormer and TMSA-Net.

## Architecture

Raw EEG → Multi-Kernel CNN → Spatial CNN → Grouped SE Attention → TMSA Branch + GQA Branch → Feature Fusion → Compact TCN → Embedding → Classifier

## Dataset

BCI Competition IV 2a GDF files:

A01T.gdf to A09T.gdf

The evaluation files A01E.gdf to A09E.gdf contain hidden labels, so this code uses subject-wise train/test split on T files.

## Run on Kaggle

```bash
!git clone https://github.com/YOUR_USERNAME/HHATNet_EEG.git
%cd HHATNet_EEG
!pip install -r requirements.txt -q
!python test_model.py
!python train_bcic2a.py --data_path /kaggle/input/datasets/lalithasriswarna/bcic2a-dataset --epochs 100 --out results/bcic2a_100epochs.csv