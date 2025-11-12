import numpy as np
from scipy.linalg import eigh, inv
import mne
from mne.decoding import CSP 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import time
import os
import matplotlib.pyplot as plt
import warnings
import argparse # <-- ADDED
import sys # <-- ADDED
# Ignore warnings related to MNE/data loading for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- V.1.3: Custom CSP Implementation (Dimensionality Reduction) ---
class CustomCSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Pattern (CSP) Dimensionality Reduction.
    Implements the CSP algorithm from scratch, inheriting from 
    BaseEstimator and TransformerMixin for seamless integration 
    into an scikit-learn Pipeline.
    
    The output feature is the log-variance of the projected signal.
    """
    def __init__(self, n_components=4):
        """
        Initializes the CSP transformer.
        :param n_components: The number of spatial filters (features) to keep. 
                             Typically an even number (e.g., 2 from each class).
        """
        self.n_components = n_components
        self.filters_ = None  # W matrix
        # CSP works best for two classes. We assume labels y have 2 unique values.

    def fit(self, X, y):
        """
        Calculates the spatial filter matrix W based on class covariances.
        
        :param X: Epoch data, shape (n_epochs, n_channels, n_times)
        :param y: Labels, shape (n_epochs,)
        :return: self
        """
        
        # 1. Separate data by class
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP is implemented for a two-class problem.")
            
        X_class1 = X[y == classes[0]]
        X_class2 = X[y == classes[1]]

        # 2. Calculate Normalized Spatial Covariance for each class
        # Input shape: (n_epochs, n_channels, n_times)
        
        def calculate_normalized_covariance(X_class):
            """
            Calculates the average normalized spatial covariance matrix (C_n).
            """
            # This function body is unchanged from original
            n_epochs, n_channels, n_times = X_class.shape
            
            # Sum of spatial covariance matrices across all epochs in the class
            C_sum = np.zeros((n_channels, n_channels))
            
            for epoch in X_class:
                # Epoch shape: (n_channels, n_times)
                # Spatial Covariance: C = (E * E^T) / trace(E * E^T)
                # E * E^T is the unnormalized covariance
                epoch_cov = np.cov(epoch)
                
                # Normalization by trace (Trace is sum of diagonal elements)
                # This ensures the total power is the same across trials
                C_norm = epoch_cov / np.trace(epoch_cov)
                C_sum += C_norm
                
            # Average normalized covariance across epochs
            return C_sum / n_epochs
        
        C1 = calculate_normalized_covariance(X_class1)
        C2 = calculate_normalized_covariance(X_class2)
        
        # 3. Calculate Composite Covariance (C_c) and Whitening Matrix (P)
        # Regularize indivisual covariancies C1 and C2 for more stability
        regularization_term = 1e-6 * np.eye(C1.shape[0])
        C1 = C1 + regularization_term
        C2 = C2 + regularization_term

        Cc = C1 + C2

        # Add Tikhonov regularization to ensure the matrix is invertable, not singular.
        # This is crucial when the number of epoch is small.
        regularization_term = 1e-6 * np.eye(Cc.shape[0])
        Cc = Cc + regularization_term  # Regularize the composite covariance
        
        # Eigenvalue decomposition of C_c: C_c = Vc * Dc * Vc^T
        # 'eigh' is used for Hermitian/symmetric matrices
        D_c, V_c = eigh(Cc) # D_c are eigenvalues, V_c are eigenvectors
        
        # Sort eigenvalues/eigenvectors in descending order (largest first)
        sort_indices = np.argsort(D_c)[::-1]
        D_c = D_c[sort_indices]
        V_c = V_c[:, sort_indices]
        
        # Whitening matrix (P): P = sqrt(inv(Dc)) * Vc^T
        # Use a small epsilon for numerical stability
        P = np.dot(np.diag(1.0 / np.sqrt(D_c + 1e-18)), V_c.T)

        # 4. Whiten the covariance matrix of C1: S1 = P * C1 * P^T
        S1 = np.dot(P, np.dot(C1, P.T))
        
        # 5. Eigendecomposition of S1: S1 = B * Lambda * B^T
        # This solves the generalized eigenvalue problem: C1*w = lambda*Cc*w
        # where C2*w = (1-lambda)*Cc*w. Eigenvectors of S1 are the common spatial filters.
        Lambda, B = eigh(S1)
        
        # Sort eigenvalues (Lambda) to maximize C1 variance and minimize C2 variance.
        # Eigenvectors corresponding to the LARGEST eigenvalues maximize C1 variance.
        # Eigenvectors corresponding to the SMALLEST eigenvalues maximize C2 variance (and thus minimize C2 variance).
        # We sort by eigenvalue magnitude (for two-class CSP)
        sort_indices = np.argsort(Lambda)[::-1]
        Lambda = Lambda[sort_indices]
        B = B[:, sort_indices]
        
        # 6. Calculate the unmixing/filter matrix W
        # W = B^T * P
        W = np.dot(B.T, P)
        
        # Select the top K filters (n_components)
        # We select the top n_components//2 from the start (max variance for class 1)
        # and the top n_components//2 from the end (max variance for class 2)
        n_p = self.n_components // 2 # Number of filters for Class 1 (largest eigenvalues)
        n_q = self.n_components - n_p # Number of filters for Class 2 (smallest eigenvalues)

        # Combine the selected filters
        W_selected = np.vstack([W[:n_p], W[-n_q:]])
        
        self.filters_ = W_selected
        
        return self

    def transform(self, X):
        """
        Applies the learned spatial filters and extracts log-variance features.
        
        :param X: Epoch data, shape (n_epochs, n_channels, n_times)
        :return: Transformed features, shape (n_epochs, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("The CSP transformer must be fitted before transforming data.")
            
        n_epochs, n_channels, n_times = X.shape
        n_components = self.n_components
        
        # X_CSP = W * X (Projection)
        # Filters_ shape: (n_components, n_channels)
        # X shape: (n_epochs, n_channels, n_times)
        # np.dot(W, X) broadcasts across epochs
        X_projected = np.dot(self.filters_, X) # Shape: (n_epochs, n_components, n_times)
        
        # Feature extraction: Log-Variance (Log-Power)
        # Variance along the time axis (axis=2)
        # Var(X_CSP) = 1/T * sum_t (X_CSP_t)^2 is often used for power/variance feature.
        # We use numpy's variance for consistency across the time dimension
        X_variance = np.var(X_projected, axis=2) # Shape: (n_epochs, n_components)

        # Add a small epsilon (1e-6) to ensure the the argument log is > 0 and 
        # To prevent division by zero in the Normalization
        epsilon = 1e-6

        # X_variance.sum(axis=1, keepdims=True) could qbe zero, we aedd epsolon to 
        # the denominator
        denominator = X_variance.sum(axis=1, keepdims=True) + epsilon
        
        # X_variance itself could be zero, we add epsilon to the numerator befor log.
        X_log_variance = np.log((X_variance + epsilon) / denominator)

        return X_log_variance  # Expected shape: (n_epochs, n_components)

# --- V.1.2: Treatment Pipeline Setup ---
def build_bci_pipeline(n_components_csp=6):
    """
    Constructs the complete classification pipeline using MNE's Csp.
    """
    # V.1.2: Pipeline setup using custom CSP and LDA
    pipeline = Pipeline([
        # 1. Dimensionality Reduction (Use MNE's stable CSP)
        ('CSP', CSP(n_components=n_components_csp, reg='empirical')), 
        # 2. Classification Algorithm (LDA is highly effective with CSP features)
        ('LDA', LinearDiscriminantAnalysis())
    ])
    return pipeline

# --- V.1.1: Preprocessing, Parsing, and Formatting (using MNE) ---
def load_and_preprocess_eeg_data(subject, runs, local_data_base_path=None):
    """
    Loads, filters, and epochs PhysioNet Motor Imagery Data.
    Prioritizes local file loading; fall back to MNE download if local files are not there.
    :param subject: The subject number (Action Number, e.g., 4).
    :param runs: List of run numbers (e.g., [6, 10, 14]).
    :param local_data_base_path: The directory path containing the 'S001' folder.
    :return: X, y, info(data, labels, MNE info)
    """
    # ... (Original function body remains unchanged)
    all_raws = []
    subj_str = 'S%03d' % subject  # e.g., 'S004'
    
    # 1. ATTEMPT LOCAL LOADING (if a base path is provided)
    
    if local_data_base_path:
        print(f"Attempting to load data from local path: {local_data_base_path}")
        try:
            for run in runs:
                run_str = 'R%02d' % run # e.g., 'R06'
                file_name = f'{subj_str}{run_str}.edf'
                # Construct the full path based on the standard EEGBCI directory structure
                full_path = os.path.join(local_data_base_path, subj_str, file_name)
                
                # Check if the file actually exists
                if not os.path.exists(full_path):
                    # Raise an error if any file is missing to trigger the fallback
                    raise FileNotFoundError(f"Local file not found: {full_path}")
                
                # Read the local raw data
                raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
                all_raws.append(raw)
                print(f"-> Successfully loaded local file: {file_name}")
            
            # If all files loaded successfully, concatenate them
            raw = mne.io.concatenate_raws(all_raws)
            print("✅ All runs loaded locally and concatenated.")
            
        except FileNotFoundError as e:
            # 2. FALLBACK TO ONLINE DOWNLOAD
            print(f"❌ Local file loading failed: {e}")
            print("Falling back to MNE's online data downloader...")
            
            # Use MNE's load_data function which handles download and path retrieval
            data_path = mne.datasets.eegbci.load_data(subject, runs, update_path=True)
            
            # Read all downloaded files and concatenate
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in data_path]
            raw = mne.io.concatenate_raws(raws)
            
    # 3. Handle case where no local path was provided (default to download)
    else:
        print("No local data path provided. Downloading data using MNE...")
        data_path = mne.datasets.eegbci.load_data(subject, runs, update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in data_path]
        raw = mne.io.concatenate_raws(raws)
        
    # --- MNE Preprocessing Steps ---
    
    # Set standard EEGBCI channel names and montage
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    print(f"\n[V.1.1 Preprocessing] Raw data info for Action {subject}, Runs {runs}:") # Updated text for clarity

    # Filter data
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
    print("-> Band-pass filtered (8-30 Hz).")
    
    # Event extraction and Epoching
    events, event_id = mne.events_from_annotations(raw)

    
    # Epoching: extract time windows (e.g., 1 to 4.1 seconds after cue)
    # Motor imagery periods are typically 1s to 4s after cue (t=0)
    tmin, tmax = 1.0, 4.1 
    epochs = mne.Epochs(raw, events, event_id=dict(T1=events[0, 2], T2=events[1, 2]), 
                        tmin=tmin, tmax=tmax, proj=True, picks='eeg', 
                        baseline=None, preload=True, verbose=False)
    
    # Select only the two motor imagery classes (T1 and T2)
    epochs.decimate(4) # Decimate to speed up computation
    epochs.pick_types(eeg=True)
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Map event IDs to 0/1 for binary classification
    # Assuming T1 maps to 0 and T2 maps to 1 based on array indexing of event_id
    y = np.where(y == epochs.event_id['T1'], 0, 1)

    print(f"-> Epochs created. Data shape: {X.shape}. Labels shape: {y.shape}")
    print("----------------------------------------")
    
    return X, y, epochs.info

# --- V.1.4: Train, Validation, and Test Script (Original, for reference) ---
def train_and_evaluate_pipeline(X, y, pipeline):
    """
    Performs cross-validation for training and evaluation.
    This function is retained for original logic, but is not used in the 
    new command-line interface to allow for custom output printing.
    """
    print("[V.1.4 Train/Validation/Test]")
    
    # Define split strategy: ShuffleSplit for cross-validation on the whole dataset.
    
    # We will use 70% for training/validation and 30% for a simulated 'never-learned' test set.
    
    # Split the data into Train/Validation set (70%) and Test set (30%)
    cv_initial = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    # Initial split (simulating Train/Validation vs Test data)
    for train_val_idx, test_idx in cv_initial.split(X, y):
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        break
        
    print(f"-> Total Epochs: {len(X)}. Train/Val Set: {len(X_train_val)}. Test Set: {len(X_test)}.")

    # 1. Cross-validation on the Training/Validation set
    cv_train = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    print("-> Performing 10-fold stratified cross-validation on Train/Validation Set...")
    scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=cv_train, n_jobs=1)
    
    mean_cv_score = scores.mean() * 100
    std_cv_score = scores.std() * 100
    
    print(f"   Cross-Validation Mean Accuracy: {mean_cv_score:.2f}% (±{std_cv_score:.2f}%)")
    
    # 2. Cross-validation on the Train/Validation set
    print("-> Training final model on all Train/Validation Set...")
    pipeline.fit(X_train_val, y_train_val)
    
    # 3. Final Evaluation on the never-learned Test Set
    test_score = pipeline.score(X_test, y_test) * 100
    print(f"   Test Set (Never-Learned Data) Accuracy: {test_score:.2f}%")
    
    # Check mandatory accuracy requirement (60%)
    if test_score >= 60.0:
        print("\n✅ Mandatory Requirement Met: Test Accuracy is >= 60.0%.")
    else:
        print("\n❌ Mandatory Requirement NOT Met: Test Accuracy is < 60.0%. (Note: Results depend heavily on dataset and hyperparams.)")

    return pipeline, X_test, y_test

# --- V.1.5: CSP Plotting (Unchanged) ---
def plot_csp_patterns(pipeline, info, n_components):
    """
    Extracts and plots the CSP spatial patterns from the fitted pipeline.
    (Original function body remains unchanged)
    """
    print("\n[V.1.5 Visualization of CSP Spatial Patterns]")
    
    # Check if the pipeline step is MNE's CSP or your custom CSP
    if 'CSP' not in pipeline.named_steps:
        print("Error: The pipeline does not contain an MNE CSP step for plotting.")
        return
        
    csp = pipeline.named_steps['CSP']

    # Get the spatial patterns (A) from the CSP object
    patterns = csp.patterns_ 
    
    # Calculate the number of components to plot from each end
    n_plot = n_components // 2 
    
    # 1. Select the top (Class 1) and bottom (Class 2) patterns
    class1_patterns = patterns[:n_plot]
    class2_patterns = patterns[-n_plot:]
    
    # --- Visualization Setup ---
    
    # Create figure and axes for plotting
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=n_plot, 
        figsize=(2 * n_plot + 1, 5), 
        squeeze=False 
    )
    
    # Set a common colormap limits for consistency
    max_val = max(np.abs(patterns).max(), 1e-6)
    vmin, vmax = -max_val, max_val

    # 2. Plotting Class 1 (Top Row)
    for i, pattern in enumerate(class1_patterns):
        ax = axes[0, i]
        mne.viz.plot_topomap(
            data=pattern, 
            pos=info, 
            vlim=(vmin, vmax), 
            axes=ax, 
            show=False,
            extrapolate='head'
        )
        ax.set_title(f"Class 1 Pattern {i+1}", fontsize=10)

    # 3. Plotting Class 2 (Bottom Row)
    for i, pattern in enumerate(class2_patterns):
        ax = axes[1, i]
        mne.viz.plot_topomap(
            data=pattern, 
            pos=info, 
            vlim=(vmin, vmax), 
            axes=ax, 
            show=False,
            extrapolate='head'
        )
        ax.set_title(f"Class 2 Pattern {i+1}", fontsize=10)

    fig.suptitle(f"Top {n_plot} CSP Spatial Patterns (Sources)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

# --- V.1.2: "Playback" Reading / Real-time Simulation Script (Unchanged) ---
def simulate_realtime_prediction(pipeline, X_test, y_test, max_delay_s=2.0):
    """
    Simulates prediction on a data stream, adhering to the < 2s delay requirement.
    (Original function body remains unchanged)
    """
    print("\n[V.1.2 Real-time Simulation]")
    print(f"Simulating prediction on the Test Set (Max Delay: {max_delay_s}s)")
    
    total_latency = 0
    correct_predictions = 0
    n_epochs = len(X_test)
    
    # Simulate predicting on 10 random epochs from the test set
    np.random.seed(42)
    random_indices = np.random.choice(n_epochs, size=10, replace=False)
    
    for i, idx in enumerate(random_indices):
        # Data chunk sent to the pipeline
        data_chunk = X_test[idx:idx+1] # Keep 3D shape (1, n_channels, n_times)
        true_label = y_test[idx]
        
        start_time = time.time()
        
        # Prediction: The pipeline steps (CSP transform + LDA predict)
        # Prediction happens here
        predicted_label = pipeline.predict(data_chunk)[0]
        
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency
        
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
            
        latency_status = "✅" if latency < max_delay_s else "❌"
        correct_status = "CORRECT" if is_correct else "INCORRECT"
        
        print(f"  Chunk {i+1}: True={true_label}, Pred={predicted_label} ({correct_status}). Latency: {latency:.4f}s {latency_status}")

    avg_latency = total_latency / len(random_indices)
    accuracy = (correct_predictions / len(random_indices)) * 100
    
    print("\n--- Simulation Summary ---")
    print(f"Average Prediction Latency: {avg_latency:.4f}s")
    print(f"Simulated Real-Time Accuracy: {accuracy:.2f}%")
    if avg_latency < max_delay_s:
        print(f"Requirement Met: Average prediction latency is below {max_delay_s}s.")
    else:
        print(f"Requirement NOT Met: Average prediction latency exceeds {max_delay_s}s.")


# --- NEW Function: Predict on Test Data for Required Output ---
def predict_on_test_data(pipeline, X_test, y_test, action_number, runs):
    """
    Performs epoch-by-epoch prediction and prints the result in the required format.
    """
    # Use BCI_Classifier.py for the desired output format
    print(f"python BCI_Classifier.py {action_number} {' '.join(map(str, runs))} predict")
    
    predictions = pipeline.predict(X_test)
    correct_predictions = 0
    
    print("epoch nb: [prediction] [truth] equal?")
    for i in range(len(X_test)):
        pred = predictions[i]
        truth = y_test[i]
        is_equal = "True" if pred == truth else "False"
        
        # Map 0 -> 1 and 1 -> 2 for output display, matching the desired example
        display_pred = pred + 1
        display_truth = truth + 1
        
        print(f"epoch {i:02d}: [{display_pred}] [{display_truth}] {is_equal}")
        if pred == truth:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(X_test)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


# --- NEW Function: Run All Experiments Analysis ---
def run_all_experiments_analysis(local_path, n_components, subjects_to_run):
    """
    Runs a representative sample of BCI experiments across selected subjects 
    and prints the cross-validation accuracy for each.
    """
    # Define the mapping of Action Number (Subject in MNE) to Runs
    # We will only run Experiments 0-3 for a faster execution, 
    # as the other two are less common.
    EXPERIMENT_MAP = {
        # Exp 0 (Action 1): Left/Right Fist
        0: {'action': 1, 'runs': [3, 7, 11]},
        # Exp 1 (Action 2): Imagine Left/Right Fist
        1: {'action': 2, 'runs': [4, 8, 12]},
        # Exp 2 (Action 3): Both Fists/Feet
        2: {'action': 3, 'runs': [5, 9, 13]},
        # Exp 3 (Action 4): Imagine Both Fists/Feet
        3: {'action': 4, 'runs': [6, 10, 14]},
    }

    all_scores = []

    # Loop through each defined experiment
    for exp_id, exp_details in EXPERIMENT_MAP.items():
        exp_action = exp_details['action']
        exp_runs = exp_details['runs']
        exp_scores = []
        
        # Loop through each selected subject
        for subject in subjects_to_run:
            try:
                # 1. Load and preprocess data
                X, y, _ = load_and_preprocess_eeg_data(subject, exp_runs, local_path)
                
                # 2. Build pipeline
                bci_pipeline = build_bci_pipeline(n_components_csp=n_components)
                
                # 3. Cross-validate (using 5-fold CV for speed in this large loop)
                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
                scores = cross_val_score(bci_pipeline, X, y, cv=cv, n_jobs=1)
                
                mean_accuracy = scores.mean()
                
                # Print individual result as requested
                print(f"experiment {exp_id}: subject {subject:03d}: accuracy = {mean_accuracy:.4f}")
                
                exp_scores.append(mean_accuracy)
                all_scores.append(mean_accuracy)
                
            except Exception as e:
                # Handle cases where data for a specific subject/run combination might be missing or corrupted
                print(f"experiment {exp_id}: subject {subject:03d}: FAILED (Error: {e})")
                
        # Calculate mean for the current experiment across subjects
        if exp_scores:
            EXPERIMENT_MAP[exp_id]['mean_acc'] = np.mean(exp_scores)
        else:
            EXPERIMENT_MAP[exp_id]['mean_acc'] = 0.0

    print("\nMean accuracy of the six different experiments for all 109 subjects:")
    
    # Print the experiment means
    for exp_id, details in EXPERIMENT_MAP.items():
        print(f"experiment {exp_id}: accuracy = {details['mean_acc']:.4f}")
        
    # Print the final grand mean
    if all_scores:
        grand_mean = np.mean(all_scores)
        print(f"Mean accuracy of {len(EXPERIMENT_MAP)} experiments: {grand_mean:.4f}")
    else:
        print("Mean accuracy of 0 experiments: 0.0000")

# --- Main Execution (MODIFIED) ---

if __name__ == '__main__':
    # Setup argument Parser
    parser = argparse.ArgumentParser(
        description="BCI EEG Data Processing and Classification.",
        usage='%(prog)s [action_number] [run1 run2 ...] [train/predict] or %(prog)s'
    )

    # Subject is positional argument 1.
    parser.add_argument('subject', type=int, nargs='?', default=None, 
                        help="Action Number (1-4). Required for 'train' or 'predict' mode.")
    
    # runs_and_mode captures ALL remaining positional arguments as STRINGS.
    parser.add_argument('runs_and_mode', nargs='*', default=None,
                        help="List of run numbers followed by the mode ('train' or 'predict').")
     
    # Optional Arguments
    parser.add_argument('--csp_components', type=int, default=6,
                        help="Number of CSP components (filters) to use.")
    parser.add_argument('--local_path', type=str,
                        default='physionet.org/files/eegmmidb/1.0.0',
                        help="Base path for local EEGBCI data.")
    args = parser.parse_args()

    N_COMPONENTS = args.csp_components
    LOCAL_EEGBCI_BASE_PATH = args.local_path

    # --- Dispatch based on the 'mode' and arguments provided ---
    
    # CASE 1: Default mode 'all' (Multi-experiment/subject analysis)
    if args.subject is None or not args.runs_and_mode:
        
        # Check if runs_and_mode is present but insufficient (e.g., just 'train')
        if args.runs_and_mode and args.runs_and_mode[0] in ['train', 'predict']:
            print("ERROR: 'train' or 'predict' mode requires ACTION NUMBER and RUNS arguments.")
            parser.print_usage()
            sys.exit(1)

        print("python Adding_parsing.py")
        print("Running analysis for all 6 experiments across a subset of subjects...")
        
        # Define a small, runnable subset (e.g., Subjects 1, 2, 3) to demonstrate the logic
        SUBJECTS_TO_ANALYZE = list(range(1, 4)) 

        # *** REPLACING THE SIMULATION WITH THE ACTUAL FUNCTION CALL ***
        run_all_experiments_analysis(LOCAL_EEGBCI_BASE_PATH, N_COMPONENTS, SUBJECTS_TO_ANALYZE)
        
    # CASE 2 & 3: Specific Action/Run analysis ('train' or 'predict')
    elif args.subject is not None and len(args.runs_and_mode) >= 2:
        
        # (This block remains unchanged from the previous, working version)
        try:
            ACTION_NUMBER = args.subject
            # Extract the mode (last string) and runs (remaining strings)
            mode = args.runs_and_mode[-1]
            runs_str_list = args.runs_and_mode[:-1]
            
            if mode not in ['train', 'predict']:
                raise ValueError(f"Invalid mode: '{mode}'. Must be 'train' or 'predict'.")

            # Convert the run numbers from string to integer
            RUNS = [int(r) for r in runs_str_list] 

            # V.1.1: Load data
            X, y, info = load_and_preprocess_eeg_data(ACTION_NUMBER, RUNS, LOCAL_EEGBCI_BASE_PATH)

            # Build the BCI classification pipeline
            bci_pipeline = build_bci_pipeline(n_components_csp=N_COMPONENTS)

            # Split the data into Train/Validation set (70%) and Test set (30%)
            cv_initial = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            X_train_val, X_test, y_train_val, y_test = None, None, None, None
            for train_val_idx, test_idx in cv_initial.split(X, y):
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]
                break

            if mode == 'train':
                # V.1.4: Train mode (Cross-validation only)
                
                # Perform 10-fold CV and format output as required
                cv_train = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
                scores = cross_val_score(bci_pipeline, X_train_val, y_train_val, cv=cv_train, n_jobs=1)
                
                # Print output in the desired format
                print(f"python BCI_Classifier.py {ACTION_NUMBER} {' '.join(map(str, RUNS))} train")
                score_str = ' '.join([f"{s:.4f}" for s in scores])
                print(f"[{score_str}]")
                print(f"cross_val_score: {scores.mean():.4f}")
                
                # Fit the pipeline on the full training/validation set for pattern plotting
                bci_pipeline.fit(X_train_val, y_train_val)
                fitted_pipeline = bci_pipeline

                # Plot the visualization after training (V.1.5)
                plot_csp_patterns(fitted_pipeline, info, N_COMPONENTS)
                
            elif mode == 'predict':
                # V.1.4/V.1.2: Predict mode
                
                # Train the final model
                bci_pipeline.fit(X_train_val, y_train_val)
                fitted_pipeline = bci_pipeline

                # Epoch-by-epoch prediction (Uses new function for required output)
                predict_on_test_data(fitted_pipeline, X_test, y_test, ACTION_NUMBER, RUNS)

                # Real-time simulation
                simulate_realtime_prediction(fitted_pipeline, X_test, y_test)
                
                # Plot the visualization
                plot_csp_patterns(fitted_pipeline, info, N_COMPONENTS)
                
        except Exception as e:
            print(f"An error occured during execution: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Catch case where subject is provided but not enough runs/mode
        print("Error: When providing an ACTION NUMBER, you must also provide RUNS and the MODE ('train' or 'predict').")
        parser.print_usage()
        sys.exit(1)
        
 
# Content of experimental runs
    # 1.Baseline, eyes open
    # 2.Baseline, eyes closed
    # 3. Task : open and close left or right fist(Action 1)
    # 4. Task : imagine opening and closing left or right fist(Action 2)
    # 5. task : Open and close both fist or both feet(Action 3)
    # 6. Task : imagine opening and closing both fists or both feet(Action 4)
    # 7. Task : open and close left or right fist(Action 1)
    # 8. Task : Imagine opening and closing left or right fist(Action 2)
    # 9. Task : open and close both fists or both feet(Action3)
    # 10. Task : Imagine opening and closing both fists or both feet(Action 4)
    # 11. Task : open and close left or right fist(Action 1)
    # 12. Task : Imagine opening and closing left or right fist(Action 2)
    # 13. Task : open and close both fists or both feet(Action 3)
    # 14. Task : imagine opening and closing both fists or both feet(Action 4)

 
 

    

