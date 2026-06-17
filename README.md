# ecg-arrhythmia-neurosim

# Real-Time Arrhythmia Detection using DNN and NeuroSim

## Overview

This project investigates arrhythmia classification from ECG signals using deep neural networks and neuromorphic computing principles. The objective was to evaluate the feasibility of deploying convolutional neural networks for real-time cardiac rhythm analysis under hardware-constrained neuromorphic environments.

## Methodology

The workflow consisted of:

* ECG signal preprocessing using the MIT-BIH Arrhythmia Database
* Heartbeat segmentation and signal filtering
* Transformation of 1D ECG signals into image-compatible representations
* CNN-based arrhythmia classification using a modified VGG8 architecture
* WAGE quantization to simulate low-precision neuromorphic hardware behavior
* Hardware-aware evaluation using DNN+NeuroSim

## Dataset

* MIT-BIH Arrhythmia Database
* Five heartbeat classes: N, L, R, V, and A

## Results

The full-precision model achieved over 99% classification accuracy within four training epochs. After introducing WAGE quantization to emulate hardware-constrained neuromorphic deployment, the model maintained approximately 91% accuracy while operating under low-precision assumptions. These results demonstrate the trade-off between classification performance and hardware feasibility.

## Technologies Used

* Python
* PyTorch
* DNN+NeuroSim
* WAGE Quantization
* NumPy
* SciPy

## Repository Contents


## Author

Satya Sujan Chinamilli
M.Eng. Artificial Intelligence, University of Cincinnati
