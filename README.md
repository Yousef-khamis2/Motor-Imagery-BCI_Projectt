# Motor Imagery BCI Interface

A Brain-Computer Interface (BCI) that utilizes motor imagery for control. This project allows for the training and testing of BCI classifiers on standard motor imagery datasets.

## Features

- Load and preprocess EEG data from CSV files
- Train and compare two classifiers (SVM and Random Forest)
- Visualize motor imagery classifications in real-time
- Subject-dependent interface that can be trained on individual subjects

## Requirements

- Python 3.6 or later
- Required libraries:
  - numpy
  - scikit-learn
  - mne
  - pandas
  - matplotlib
  - tkinter

## Installation

1. Clone this repository or download the source code
2. Install required packages:
```
pip install numpy scikit-learn mne pandas matplotlib
```

## Usage

1. Run the application:
```
python bci_interface.py
```

2. Load a CSV dataset:
   - Click "Browse" to select your BCICIV_2a_1.csv file
   - Click "Load Data" to process the data
   - The data will be automatically split into training (60 samples per class) and testing (12 samples per class)

3. Train the classifiers:
   - Adjust CSP components if desired (default is 4)
   - Click "Train Models" to train both SVM and Random Forest classifiers
   - Choose which model to use for testing by selecting either "SVM" or "Random Forest"

4. Test the interface:
   - Click "Start Testing" to begin real-time testing
   - The corresponding arrow will flash red when a motor imagery class is detected
   - Click "Stop Testing" to end the testing session

5. Compare model performance:
   - Click "Compare Models" to see a detailed comparison of both classifiers

## Input Data Format

The application expects a CSV file with the following columns:
- `epoch`: Identifier for each trial
- `label`: Class labels (feet, left_hand, right_hand, tongue)
- `EEG-*`: EEG channel data (multiple columns)

## Project Requirements Fulfilled

1. ✓ Subject-dependent interface (trained and tested on the same subject)
2. ✓ Uses freely available BCI Competition dataset
3. ✓ Includes all four motor imagery classes (feet, left hand, right hand, tongue)
4. ✓ Implements appropriate EEG preprocessing (bandpass filtering, standardization)
5. ✓ Uses CSP for feature extraction with configurable number of components
6. ✓ Uses two classifiers (SVM and Random Forest) with parameter optimization
7. ✓ Compares the classifiers based on accuracy
8. ✓ Provides a UI with arrow visualization that responds to detected movements

## Troubleshooting

- If the application fails to load data, check if your CSV file has the expected format
- If you encounter any issues with training, try adjusting the number of CSP components
- For any other issues, check the console output for error messages 