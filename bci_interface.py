import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from mne.decoding import CSP
import os
import logging
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotorImageryBCI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motor Imagery BCI Interface")
        self.root.geometry("900x800")  # Increased height to accommodate all sections
        self.root.resizable(True, True)
        
        # State variables
        self.csv_file = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.svm_model = None
        self.rf_model = None
        self.csp = None
        self.current_model = "SVM"  # Default model
        self.is_training = False
        self.is_testing = False
        self.test_thread = None
        
        # Define class mappings for the actual dataset labels
        self.class_labels = ['foot', 'left', 'right', 'tongue']
        # Map class names to indices for visualization
        self.class_to_idx = {
            'foot': 0,   # Bottom arrow
            'left': 1,   # Left arrow
            'right': 2,  # Right arrow
            'tongue': 3  # Top arrow
        }
        
        # Initialize visualization components
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off')
        
        # Main frame with scrollbar
        main_canvas = tk.Canvas(self.root)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=main_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create main frame inside canvas
        self.main_frame = ttk.Frame(main_canvas, padding=10)
        main_canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)
        
        # Create UI elements
        self._create_data_section()
        self._create_model_section()
        self._create_visualization_section()
        self._create_test_section()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure scrolling
        self.main_frame.bind('<Configure>', lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        
        # Initial arrow drawing
        self._draw_arrows()
        
    def _create_data_section(self):
        data_frame = ttk.LabelFrame(self.main_frame, text="Data Settings", padding=10)
        data_frame.pack(fill=tk.X, pady=5)
        
        # CSV file selection
        ttk.Label(data_frame, text="CSV Data File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.file_var, width=40).grid(row=0, column=1, pady=5)
        ttk.Button(data_frame, text="Browse", command=self._browse_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(data_frame, text="Load Data", command=self._load_data).grid(row=0, column=3, padx=5, pady=5)
        
        # Data info
        self.data_info_var = tk.StringVar()
        self.data_info_var.set("No data loaded")
        ttk.Label(data_frame, textvariable=self.data_info_var, wraplength=600).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)
        
    def _create_model_section(self):
        model_frame = ttk.LabelFrame(self.main_frame, text="Model Settings", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # CSP components frame
        csp_frame = ttk.Frame(model_frame)
        csp_frame.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(csp_frame, text="CSP Components:").pack(side=tk.LEFT, padx=5)
        self.csp_var = tk.IntVar(value=4)
        csp_spinbox = ttk.Spinbox(csp_frame, from_=2, to=20, textvariable=self.csp_var, width=5)
        csp_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Filter bank option
        self.use_filter_bank_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(csp_frame, text="Use Filter Bank", variable=self.use_filter_bank_var).pack(side=tk.LEFT, padx=20)
        
        # Model selection frame
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)
        
        ttk.Label(model_select_frame, text="Active Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="SVM")
        ttk.Radiobutton(model_select_frame, text="SVM", variable=self.model_var, value="SVM").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_select_frame, text="Random Forest", variable=self.model_var, value="RF").pack(side=tk.LEFT, padx=5)
        
        # Hyperparameter tuning option
        self.use_hyperopt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Use Hyperparameter Optimization", variable=self.use_hyperopt_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(model_frame, variable=self.progress_var, length=400, mode='determinate')
        self.progress_bar.grid(row=1, column=1, pady=5, padx=10, sticky=tk.W)
        
        # Training buttons
        button_frame = ttk.Frame(model_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(button_frame, text="Train Models", command=self._train_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Compare Models", command=self._compare_models).pack(side=tk.LEFT, padx=5)
        
        # Model info
        self.model_info_var = tk.StringVar()
        self.model_info_var.set("No models trained")
        ttk.Label(model_frame, textvariable=self.model_info_var, wraplength=800).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
    def _create_visualization_section(self):
        # Create a frame for the visualization section
        viz_frame = ttk.LabelFrame(self.main_frame, text="Motor Imagery Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Force update to ensure proper layout
        self.root.update_idletasks()
        
    def _create_test_section(self):
        # Create test frame with distinct border and padding
        test_frame = ttk.LabelFrame(self.main_frame, text="Testing Controls", padding=10)
        test_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create button frame with distinct background and style
        style = ttk.Style()
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
        button_frame = ttk.Frame(test_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Add Start and Stop buttons with improved visibility
        start_btn = ttk.Button(button_frame, text="▶ Start Testing", 
                             command=self._start_testing,
                             style='Action.TButton')
        start_btn.pack(side=tk.LEFT, padx=5)
        
        stop_btn = ttk.Button(button_frame, text="⬛ Stop Testing", 
                            command=self._stop_testing,
                            style='Action.TButton')
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Add separator
        ttk.Separator(test_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Add test results label with border
        result_frame = ttk.Frame(test_frame, relief=tk.GROOVE, borderwidth=1)
        result_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.test_result_var = tk.StringVar(value="Click 'Start Testing' to begin")
        result_label = ttk.Label(result_frame, textvariable=self.test_result_var, 
                               wraplength=600, padding=5)
        result_label.pack(fill=tk.X)
        
        # Force update to ensure proper layout
        self.root.update_idletasks()
        
    def _browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.file_var.set(filename)
            self.csv_file = filename
            
    def _load_data(self):
        try:
            if not self.csv_file:
                messagebox.showerror("Error", "Please select a CSV file first")
                return
                
            self.status_var.set("Loading data...")
            self.root.update_idletasks()
            
            # Load the CSV file
            logger.info(f"Loading data from {self.csv_file}")
            df = pd.read_csv(self.csv_file)
            
            # Extract unique epochs and classes
            unique_epochs = sorted(df['epoch'].unique())
            unique_labels = sorted(df['label'].unique())
            
            # Display the unique labels found in the dataset (for debugging)
            logger.info(f"Unique labels found in dataset: {unique_labels}")
            
            # Prepare data arrays
            eeg_columns = [col for col in df.columns if col.startswith('EEG')]
            n_channels = len(eeg_columns)
            n_times = 201  # As per the project requirements
            n_trials = len(unique_epochs)
            
            X = np.zeros((n_trials, n_channels, n_times))
            y = []
            
            # Process each epoch
            for i, epoch_num in enumerate(unique_epochs):
                epoch_data = df[df['epoch'] == epoch_num]
                
                # Get label
                label = epoch_data['label'].iloc[0]
                y.append(label)
                
                # Extract EEG data
                for j, channel in enumerate(eeg_columns):
                    X[i, j, :] = epoch_data[channel].values[:n_times]
            
            # Enhanced preprocessing pipeline
            logger.info("Applying preprocessing pipeline...")
            
            # 1. Apply notch filter to remove power line interference (50Hz)
            X_notch = self._apply_notch_filter(X, 50)
            
            # 2. Apply bandpass filter (8-30 Hz for motor imagery)
            X_filtered = self._apply_bandpass_filter(X_notch, [8, 30], fs=250)
            
            # 3. Remove artifacts
            X_clean = self._remove_artifacts(X_filtered)
            
            # 4. Standardize the data
            X_processed = self._standardize_data(X_clean)
            
            # Keep labels as strings
            y_strings = np.array(y)
            
            # Display class distribution before splitting
            label_counts = {}
            for label in y_strings:
                label_counts[label] = label_counts.get(label, 0) + 1
            logger.info(f"Class distribution: {label_counts}")
            
            # Split into train/test with cross-validation
            self.X_train, self.X_test, self.y_train, self.y_test = self._split_train_test_by_class(
                X_processed, y_strings, train_per_class=60, test_per_class=12
            )
            
            # Update info with actual class labels from dataset
            self.data_info_var.set(
                f"Data loaded successfully.\n"
                f"Total samples: {n_trials}, Features: {n_channels} channels x {n_times} time points\n"
                f"Class distribution: {label_counts}\n"
                f"Training samples: {len(self.y_train)}, Testing samples: {len(self.y_test)}"
            )
            
            self.status_var.set("Data loaded successfully")
            logger.info("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
            self.status_var.set("Error loading data")
            
    def _apply_bandpass_filter(self, X, bandpass, fs=250, order=5):
        """Apply bandpass filter to the data."""
        nyq = 0.5 * fs
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        b, a = butter(order, [low, high], btype='band')
        
        n_trials, n_channels, n_times = X.shape
        X_filtered = np.zeros_like(X)
        
        for i in range(n_trials):
            for j in range(n_channels):
                X_filtered[i, j] = lfilter(b, a, X[i, j])
                
        return X_filtered
        
    def _apply_notch_filter(self, X, f0, fs=250, Q=30):
        """Apply notch filter to remove power line interference."""
        nyq = 0.5 * fs
        w0 = f0 / nyq
        b, a = butter(2, [w0-0.1, w0+0.1], btype='bandstop')
        
        n_trials, n_channels, n_times = X.shape
        X_filtered = np.zeros_like(X)
        
        for i in range(n_trials):
            for j in range(n_channels):
                X_filtered[i, j] = lfilter(b, a, X[i, j])
                
        return X_filtered
    
    def _remove_artifacts(self, X, threshold=100):
        """Simple artifact removal by amplitude thresholding."""
        X_clean = X.copy()
        
        # Get trial-wise standard deviation
        trial_std = np.std(X_clean, axis=(1, 2))
        outlier_trials = np.where(trial_std > np.mean(trial_std) + 2 * np.std(trial_std))[0]
        
        logger.info(f"Removing {len(outlier_trials)} outlier trials out of {X.shape[0]}")
        
        # Replace outlier trials with mean of non-outlier trials
        if len(outlier_trials) > 0 and len(outlier_trials) < X.shape[0]:
            good_trials = np.setdiff1d(np.arange(X.shape[0]), outlier_trials)
            mean_trial = np.mean(X[good_trials], axis=0)
            for idx in outlier_trials:
                X_clean[idx] = mean_trial
                
        # Channel-wise artifact removal
        for i in range(X_clean.shape[0]):
            for j in range(X_clean.shape[1]):
                # Find outlier points in the channel
                mean_val = np.mean(X_clean[i, j])
                std_val = np.std(X_clean[i, j])
                threshold_val = threshold * std_val
                
                # Replace outliers with interpolated values
                outliers = np.abs(X_clean[i, j] - mean_val) > threshold_val
                if np.any(outliers):
                    indices = np.arange(X_clean.shape[2])
                    X_clean[i, j, outliers] = np.interp(indices[outliers], indices[~outliers], X_clean[i, j, ~outliers])
        
        return X_clean
        
    def _standardize_data(self, X):
        """Standardize data by trial."""
        n_trials, n_channels, n_times = X.shape
        X_std = np.zeros_like(X)
        
        for i in range(n_trials):
            for j in range(n_channels):
                # Standardize each channel in each trial
                X_std[i, j] = (X[i, j] - np.mean(X[i, j])) / np.std(X[i, j])
                
        return X_std
        
    def _split_train_test_by_class(self, X, y, train_per_class=60, test_per_class=12):
        """Split data into train/test ensuring equal class distribution with stratified sampling."""
        classes = np.unique(y)
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        
        for c in classes:
            # Get indices for this class
            idx = np.where(y == c)[0]
            
            # Shuffle the indices
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(idx)
            
            if len(idx) < (train_per_class + test_per_class):
                logger.warning(f"Not enough samples for class {c}: {len(idx)} < {train_per_class + test_per_class}")
                train_size = int(0.8 * len(idx))
                train_idx = idx[:train_size]
                test_idx = idx[train_size:]
            else:
                train_idx = idx[:train_per_class]
                test_idx = idx[train_per_class:train_per_class+test_per_class]
                
            X_train_list.append(X[train_idx])
            y_train_list.append(y[train_idx])
            X_test_list.append(X[test_idx])
            y_test_list.append(y[test_idx])
            
            logger.info(f"Class {c}: {len(train_idx)} training samples, {len(test_idx)} testing samples")
            
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Shuffle training data
        train_shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        
        return X_train, X_test, y_train, y_test
        
    def _train_models(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        if self.is_training:
            messagebox.showinfo("Info", "Training already in progress")
            return
            
        self.is_training = True
        
        # Start a training thread
        training_thread = threading.Thread(target=self._train_models_thread)
        training_thread.daemon = True
        training_thread.start()
    
    def _train_models_thread(self):
        """Worker thread for model training."""
        try:
            self.root.after(0, lambda: self.status_var.set("Training models..."))
            self.root.after(0, lambda: self.progress_var.set(0.0))
            
            # Update progress
            def update_progress(value, message=""):
                self.root.after(0, lambda: self.progress_var.set(value))
                if message:
                    self.root.after(0, lambda: self.status_var.set(message))
            
            # Get CSP components
            n_components = self.csp_var.get()
            
            # Apply CSP transformation
            logger.info(f"Applying CSP with {n_components} components")
            update_progress(10, f"Applying CSP with {n_components} components...")
            
            self.csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
            X_train_csp = self.csp.fit_transform(self.X_train, self.y_train)
            X_test_csp = self.csp.transform(self.X_test)
            
            update_progress(20, "CSP transformation complete")
            
            # Hyperparameter tuning
            use_hyperopt = self.use_hyperopt_var.get()
            
            if use_hyperopt:
                # Setup cross-validation
                cv = 5
                logger.info(f"Using {cv}-fold cross-validation for hyperparameter tuning")
                
                # Define parameter grids
                svm_param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
                
                rf_param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Train SVM with GridSearch
                logger.info("Optimizing SVM hyperparameters...")
                update_progress(30, "Optimizing SVM hyperparameters...")
                
                svm_base = SVC(probability=True, random_state=42)
                svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=cv, n_jobs=-1, verbose=0)
                svm_grid.fit(X_train_csp, self.y_train)
                
                # Get best SVM model
                self.svm_model = svm_grid.best_estimator_
                logger.info(f"Best SVM parameters: {svm_grid.best_params_}")
                
                update_progress(50, "SVM optimization complete")
                
                # Train RF with GridSearch
                logger.info("Optimizing Random Forest hyperparameters...")
                update_progress(60, "Optimizing Random Forest hyperparameters...")
                
                rf_base = RandomForestClassifier(random_state=42)
                rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=cv, n_jobs=-1, verbose=0)
                rf_grid.fit(X_train_csp, self.y_train)
                
                # Get best RF model
                self.rf_model = rf_grid.best_estimator_
                logger.info(f"Best RF parameters: {rf_grid.best_params_}")
                
                update_progress(80, "Random Forest optimization complete")
                
                # Evaluate models
                svm_train_pred = self.svm_model.predict(X_train_csp)
                svm_test_pred = self.svm_model.predict(X_test_csp)
                svm_train_acc = accuracy_score(self.y_train, svm_train_pred)
                svm_test_acc = accuracy_score(self.y_test, svm_test_pred)
                
                rf_train_pred = self.rf_model.predict(X_train_csp)
                rf_test_pred = self.rf_model.predict(X_test_csp)
                rf_train_acc = accuracy_score(self.y_train, rf_train_pred)
                rf_test_acc = accuracy_score(self.y_test, rf_test_pred)
                
                # Update model info
                model_info = (
                    f"Models trained with {n_components} CSP components and hyperparameter tuning.\n"
                    f"SVM - Best params: {svm_grid.best_params_}\n"
                    f"SVM - Training accuracy: {svm_train_acc:.3f}, Testing accuracy: {svm_test_acc:.3f}\n"
                    f"Random Forest - Best params: {rf_grid.best_params_}\n"
                    f"Random Forest - Training accuracy: {rf_train_acc:.3f}, Testing accuracy: {rf_test_acc:.3f}"
                )
            else:
                # Train SVM model without hyperparameter tuning
                logger.info("Training SVM model")
                update_progress(40, "Training SVM model...")
                
                self.svm_model = SVC(kernel='rbf', gamma='scale', C=10, probability=True, random_state=42)
                self.svm_model.fit(X_train_csp, self.y_train)
                
                # Train RF model without hyperparameter tuning
                logger.info("Training Random Forest model")
                update_progress(70, "Training Random Forest model...")
                
                self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
                self.rf_model.fit(X_train_csp, self.y_train)
                
                # Evaluate models
                svm_train_pred = self.svm_model.predict(X_train_csp)
                svm_test_pred = self.svm_model.predict(X_test_csp)
                svm_train_acc = accuracy_score(self.y_train, svm_train_pred)
                svm_test_acc = accuracy_score(self.y_test, svm_test_pred)
                
                rf_train_pred = self.rf_model.predict(X_train_csp)
                rf_test_pred = self.rf_model.predict(X_test_csp)
                rf_train_acc = accuracy_score(self.y_train, rf_train_pred)
                rf_test_acc = accuracy_score(self.y_test, rf_test_pred)
                
                # Update model info
                model_info = (
                    f"Models trained with {n_components} CSP components.\n"
                    f"SVM - Training accuracy: {svm_train_acc:.3f}, Testing accuracy: {svm_test_acc:.3f}\n"
                    f"Random Forest - Training accuracy: {rf_train_acc:.3f}, Testing accuracy: {rf_test_acc:.3f}"
                )
            
            # Set current model
            self.current_model = self.model_var.get()
            
            # Update UI
            self.root.after(0, lambda: self.model_info_var.set(model_info))
            update_progress(100, "Models trained successfully")
            
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error training models: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error training models"))
            
        finally:
            self.is_training = False
            
    def _compare_models(self):
        if self.svm_model is None or self.rf_model is None or self.csp is None:
            messagebox.showerror("Error", "Please train models first")
            return
            
        # Create comparison window
        compare_win = tk.Toplevel(self.root)
        compare_win.title("Model Comparison")
        compare_win.geometry("600x500")
        
        # Apply CSP transformation
        X_test_csp = self.csp.transform(self.X_test)
        
        # Get predictions
        svm_preds = self.svm_model.predict(X_test_csp)
        rf_preds = self.rf_model.predict(X_test_csp)
        
        # Calculate accuracy
        svm_acc = accuracy_score(self.y_test, svm_preds)
        rf_acc = accuracy_score(self.y_test, rf_preds)
        
        # Create report text using class labels
        svm_report = classification_report(self.y_test, svm_preds, 
                                        target_names=self.class_labels)
        rf_report = classification_report(self.y_test, rf_preds, 
                                       target_names=self.class_labels)
        
        # Create labels and text areas
        ttk.Label(compare_win, text="SVM Model", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(compare_win, text=f"Accuracy: {svm_acc:.3f}").pack()
        
        svm_text = tk.Text(compare_win, height=10, width=70)
        svm_text.pack(padx=10, pady=5)
        svm_text.insert(tk.END, svm_report)
        svm_text.config(state=tk.DISABLED)  # Make read-only
        
        ttk.Label(compare_win, text="Random Forest Model", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(compare_win, text=f"Accuracy: {rf_acc:.3f}").pack()
        
        rf_text = tk.Text(compare_win, height=10, width=70)
        rf_text.pack(padx=10, pady=5)
        rf_text.insert(tk.END, rf_report)
        rf_text.config(state=tk.DISABLED)  # Make read-only
        
        # Create close button
        ttk.Button(compare_win, text="Close", command=compare_win.destroy).pack(pady=10)
        
    def _start_testing(self):
        if self.X_test is None or self.y_test is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        if self.svm_model is None or self.rf_model is None or self.csp is None:
            messagebox.showerror("Error", "Please train models first")
            return
            
        if self.is_testing:
            messagebox.showinfo("Info", "Testing already in progress")
            return
            
        self.is_testing = True
        self.status_var.set("Testing in progress...")
        
        # Start testing in a separate thread
        self.test_thread = threading.Thread(target=self._run_testing)
        self.test_thread.daemon = True
        self.test_thread.start()
        
    def _stop_testing(self):
        if not self.is_testing:
            messagebox.showinfo("Info", "No testing in progress")
            return
            
        self.is_testing = False
        self.status_var.set("Testing stopped")
        
    def _run_testing(self):
        try:
            # Apply CSP transformation
            X_test_csp = self.csp.transform(self.X_test)
            
            # Get current model
            model = self.svm_model if self.current_model == "SVM" else self.rf_model
            
            # Randomly test samples
            correct = 0
            total = 0
            
            for i in range(len(self.X_test)):
                if not self.is_testing:
                    break
                    
                # Get random sample
                sample_idx = np.random.randint(0, len(self.X_test))
                sample = X_test_csp[sample_idx:sample_idx+1]
                true_class = self.y_test[sample_idx]  # String label
                
                # Predict
                pred_class = model.predict(sample)[0]  # String label
                
                # Log prediction details for debugging
                logger.info(f"Sample {i+1}: True class = {true_class}, Predicted class = {pred_class}")
                
                # Update visualization with prediction
                self.root.after(0, self._draw_arrows, pred_class)
                
                # Update result text
                result_text = (
                    f"Testing sample {i+1}\n"
                    f"True class: {true_class}\n"
                    f"Predicted class: {pred_class}\n"
                    f"Correct: {true_class == pred_class}"
                )
                self.root.after(0, self.test_result_var.set, result_text)
                
                # Update counters
                total += 1
                if true_class == pred_class:
                    correct += 1
                    
                # Update display
                self.root.update_idletasks()
                time.sleep(2)  # Pause between tests
                
            # Final results
            if total > 0:
                accuracy = correct / total
                final_result = (
                    f"Testing completed\n"
                    f"Accuracy: {accuracy:.3f} ({correct}/{total})"
                )
                self.root.after(0, self.test_result_var.set, final_result)
                
            self.root.after(0, self.status_var.set, "Testing completed")
            
        except Exception as e:
            logger.error(f"Error during testing: {str(e)}", exc_info=True)
            self.root.after(0, self.test_result_var.set, f"Error during testing: {str(e)}")
            self.root.after(0, self.status_var.set, "Error during testing")
            
        finally:
            self.is_testing = False
            self.root.after(0, self._draw_arrows)  # Reset arrows
            
    def _draw_arrows(self, highlight=None):
        """Draw the motor imagery arrows with optional highlighting."""
        self.ax.clear()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off')
        
        # Define arrow configurations
        arrow_configs = [
            {'pos': (0, -0.6), 'dir': (0, 0.4), 'label': 'Foot', 'class_name': 'foot'},
            {'pos': (-0.6, 0), 'dir': (-0.4, 0), 'label': 'Left Hand', 'class_name': 'left'},
            {'pos': (0.6, 0), 'dir': (0.4, 0), 'label': 'Right Hand', 'class_name': 'right'},
            {'pos': (0, 0.6), 'dir': (0, -0.4), 'label': 'Tongue', 'class_name': 'tongue'}
        ]
        
        # Draw each arrow
        for config in arrow_configs:
            # Determine color based on highlight
            color = 'red' if highlight is not None and highlight == config['class_name'] else 'blue'
            
            # Draw arrow
            self.ax.arrow(config['pos'][0], config['pos'][1],
                         config['dir'][0], config['dir'][1],
                         head_width=0.1, head_length=0.1,
                         fc=color, ec=color, linewidth=3)
            
            # Position label
            if config['label'] in ['Left Hand', 'Right Hand']:
                y_offset = -0.2
                x_pos = config['pos'][0]
            else:
                y_offset = -0.8 if config['label'] == 'Foot' else 0.8
                x_pos = 0
            
            self.ax.text(x_pos, y_offset, config['label'], ha='center')
        
        # Update the canvas
        try:
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating canvas: {str(e)}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MotorImageryBCI(root)
    root.mainloop() 