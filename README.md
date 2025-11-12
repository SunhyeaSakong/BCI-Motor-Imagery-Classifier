📄 README.md 

🧠 BCI-Motor-Imagery-Classifier: EEG Signal Processing using Custom CSP/LDA

This project implements a complete pipeline for Brain-Computer Interface (BCI) classification, specifically targeting Motor Imagery (MI) tasks from the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

The core of the project is an scikit-learn pipeline that uses Common Spatial Patterns (CSP) for feature extraction and Linear Discriminant Analysis (LDA) for classification. It features a custom, from-scratch implementation of the CSP algorithm alongside MNE's robust data handling.

✨ Key Features

    Custom CSP Implementation: Includes a robust CustomCSP class, built from numpy and scipy.linalg, integrated directly into the scikit-learn pipeline.

    MNE Integration: Leverages the MNE-Python library for efficient data loading, band-pass filtering (8-30 Hz), epoching, and visualization of spatial patterns (topomaps).

    Comprehensive Evaluation: Supports stratified cross-validation (CV), evaluation on a never-learned test set, and a simulated real-time prediction latency check (< 2s delay requirement).

    Flexible CLI: A command-line interface (argparse) allows for multiple modes:

        Full Analysis: Runs cross-validation across multiple subjects and experiments.

        Train Mode: Trains and performs CV on specified runs and visualizes CSP patterns.

        Predict Mode: Evaluates the trained model on a hold-out test set and simulates real-time performance.

🛠️ Dependencies

This project requires the following Python libraries:

    numpy

    scipy

    mne

    scikit-learn

    matplotlib

You can install them using pip:
Bash

pip install numpy scipy mne scikit-learn matplotlib

🚀 Usage

The script BCI_Classifier.py (assuming you name your file this) can be run in three main modes:

1. Full Experiment Analysis (Default Mode)

This mode runs the pipeline's cross-validation logic across a defined subset of subjects for all four common MI experiments.
Bash

python BCI_Classifier.py

2. Training Mode (Cross-Validation & Plotting)

Trains the model on the specified ACTION_NUMBER and RUNS, performs 10-fold cross-validation, and displays the CSP spatial patterns (topomaps).

    ACTION_NUMBER: Corresponds to the experiment (e.g., 4 for Imagine Both Fists/Feet).

    RUNS: A space-separated list of run numbers to use (e.g., 6 10 14).

Bash

# Example: Train on Subject 4, Runs 6, 10, and 14
python BCI_Classifier.py 4 6 10 14 train

3. Prediction Mode (Test Accuracy & Real-time Simulation)

Trains the final model on 70% of the data and reports epoch-by-epoch predictions, final test accuracy, and latency for a real-time prediction simulation on the remaining 30% test set.
Bash

# Example: Predict on Subject 4, Runs 6, 10, and 14
python BCI_Classifier.py 4 6 10 14 predict

⚙️ Customization (Optional Arguments)

You can customize the number of CSP components and the local data path using optional arguments:
Argument	Type	Default	Description
--csp_components	int	6	Number of spatial filters to select (n_components).
--local_path	str	physionet.org/files/eegmmidb/1.0.0	Base directory for local EEGBCI data files.
Bash

# Example with custom CSP components
python BCI_Classifier.py 4 6 10 14 train --csp_components 8