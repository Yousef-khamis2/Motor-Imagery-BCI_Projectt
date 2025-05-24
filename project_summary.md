# Motor Imagery BCI Project Summary

## Overview

This project implements a subject-dependent Brain-Computer Interface (BCI) based on motor imagery mental strategy. It processes EEG data from standard datasets and classifies them into four motor imagery classes: feet, left hand, right hand, and tongue.

## Implemented Components

1. **Data Processing**
   - CSV data loading and parsing
   - EEG preprocessing with bandpass filtering (8-30 Hz)
   - Data standardization
   - Class-balanced train/test splitting (60 training, 12 testing samples per class)

2. **Feature Extraction**
   - Common Spatial Patterns (CSP) implementation
   - Configurable number of CSP components (2-10)

3. **Classification**
   - Support Vector Machine (SVM) classifier
   - Random Forest classifier
   - Model comparison and evaluation

4. **User Interface**
   - Interactive GUI for data loading and model training
   - Visualization of motor imagery with directional arrows
   - Real-time classification testing
   - Detailed performance reporting

## Project Structure

- `bci_interface.py` - Main application with GUI and all processing components
- `README.md` - Usage instructions and project information
- `requirements.txt` - Required Python packages

## How to Use

1. Load your BCICIV_2a_1.csv dataset
2. Train both SVM and Random Forest models
3. Test the interface with real-time classification
4. Compare the performance of both models

## Requirements Checklist

The implementation satisfies all the specified requirements:

✅ Subject-dependent interface  
✅ Uses non-invasive freely published BCI Competition data  
✅ Includes all four motor imagery classes  
✅ Implements EEG preprocessing  
✅ Uses CSP for feature extraction  
✅ Implements two classifiers (SVM and Random Forest)  
✅ Compares classifiers based on accuracy  
✅ Provides a UI with directional arrows for visualization

## Next Steps

Potential improvements to consider:
- Additional feature extraction methods beyond CSP
- Deep learning-based approaches
- Online/real-time BCI implementation with live EEG data
- More sophisticated preprocessing techniques
- Hyperparameter optimization for the classifiers 