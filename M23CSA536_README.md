SepID-Enhance: Speaker Separation, Identification, and Enhancement Pipeline

Introduction
This project explores the application of advanced speech processing techniques in challenging audio environments, with a focus on multi-speaker speech enhancement, speaker verification, and language classification using MFCC features. The first part of the pipeline tackles speech signal enhancement and separation in overlapping speaker scenarios through the SepFormer model. Speaker verification is achieved using the pre-trained WavLM Base Plus model, followed by fine-tuning with Low-Rank Adaptation (LoRA) and ArcFace loss on a subset of the VoxCeleb2 dataset. Performance is evaluated using metrics like Equal Error Rate (EER), TAR@1%FAR, and Speaker Identification Accuracy. A unique pipeline is designed to combine speaker separation, identification, and enhancement through joint training, offering significant improvements in real-world multi-speaker conditions.

The second part of the project involves language classification, where MFCC-based feature extraction is applied to audio samples in 10 Indian languages. These features are then used to build a language classification model, employing a Random Forest classifier. The task focuses on understanding how MFCCs capture language-specific acoustic characteristics, while addressing challenges such as speaker variability and background noise.

Overview
SepID-Enhance is a comprehensive and modular pipeline aimed at solving multi-speaker audio processing challenges. By integrating state-of-the-art techniques in speaker separation, identification, enhancement, and language classification, the system is well-suited for tasks such as speaker diarization, transcription, multilingual voice analytics, and audio scene understanding. The pipeline leverages deep learning models and signal processing methods to achieve high-quality results in diverse and noisy environments.

This project integrates:
Speaker Separation: Using SepFormer

Speaker Identification: via WavLM (pre-trained and fine-tuned with LoRA & ArcFace)

Speech Enhancement: Through joint training

Language Classification: Using MFCCs and Random Forest

Key Features
Speaker Separation & Enhancement
Model: speechbrain/sepformer-wsj02mix

Metrics:

SDR (Signal-to-Distortion Ratio): 9.80

SIR (Signal-to-Interference Ratio): 10.50

SAR (Signal-to-Artifacts Ratio): 11.20

PESQ (Perceptual Evaluation of Speech Quality): 1.95

Speaker Identification
Model: microsoft/wavlm-base-plus

Pre-trained Performance:

EER (Equal Error Rate): 34.00%

TAR@1%FAR: 12.00%

Accuracy: 66.10%

Fine-tuned Performance:

EER: 52.48%

TAR@1%FAR: 0.29%

Accuracy: 47.40%

Rank-1 Accuracy (on separated speech):

Pre-trained: 58.00%

Fine-tuned: 62.00%

Language Classification
Dataset: 10 Indian Languages (from Kaggle)

Features: MFCCs

Classifier: Random Forest

Accuracy: 76.57%

Setup
Datasets
VoxCeleb2

VoxCeleb1

Indian Languages Dataset (Available on Kaggle)

Requirements
Python 3.8+

Install dependencies:

bash
Copy
pip install torch torchaudio transformers speechbrain pesq numpy tqdm peft librosa soundfile scik

## GitHub
https://github.com/M23CSA536-Codes/Speech_Understanding_Assignment_PA2

