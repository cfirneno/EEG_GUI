# Modular EEG Analysis Dashboard
# This script creates a graphical user interface (GUI) to run and verify 
# each step of the EEG analysis pipeline independently.

# --- Installation ---
# You'll need to install the necessary libraries. Open your terminal or command prompt and run:
# pip install mne numpy pywavelets nolds matplotlib

import os
import shutil
import numpy as np
import mne
import pywt
import nolds
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# --- Core Analysis Functions (from the original script) ---
# These functions contain the main logic for each step of the analysis.

def load_and_validate_eeg(directory_path, file_extension, progress_callback):
    """Loads and validates EEG files, providing progress updates."""
    valid_raw_objects = []
    invalid_files = {}
    filenames = [f for f in os.listdir(directory_path) if f.endswith(file_extension)]
    
    for i, filename in enumerate(filenames):
        file_path = os.path.join(directory_path, filename)
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValueError("File contains NaN or infinite values.")
            
            progress_callback(f"  [SUCCESS] Validated: {filename}\n")
            raw.info['filename'] = filename
            valid_raw_objects.append(raw)
            
        except Exception as e:
            progress_callback(f"  [SKIPPED] Invalid file {filename}. Reason: {e}\n")
            invalid_files[filename] = str(e)
            
    return valid_raw_objects, invalid_files

def perform_wpd(raw, wavelet, level):
    """Performs WPD on a single MNE Raw object."""
    wpd_results = {}
    data = raw.get_data()
    for i, channel_data in enumerate(data):
        ch_name = raw.ch_names[i]
        wp = pywt.WaveletPacket(data=channel_data, wavelet=wavelet, mode='symmetric', maxlevel=level)
        wpd_results[ch_name] = wp
    return wpd_results

def plot_wpd_scalogram(wp, sfreq, ch_name, filename):
    """Generates and displays a scalogram-like plot for WPD results."""
    nodes = wp.get_level(wp.maxlevel, order='freq')
    labels = [n.path for n in nodes]
    coeffs = [n.data for n in nodes]
    max_len = max(len(c) for c in coeffs)
    coeffs_padded = np.array([np.pad(c, (0, max_len - len(c)), 'constant') for c in coeffs])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.abs(coeffs_padded), interpolation='nearest', cmap=plt.cm.viridis,
                   aspect='auto', origin='lower')
    ax.set_title(f"WPD Scalogram\n(File: {filename}, Channel: {ch_name})")
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Frequency Bands (Packets)")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label='Coefficient Magnitude')
    plt.tight_layout()
    plt.show() # Display the plot instead of saving it

def calculate_lyapunov_for_wp(wp, emb_dim, lag):
    """Calculates Lyapunov exponents for a single WaveletPacket object."""
    lyapunov_results = {}
    terminal_nodes = wp.get_level(wp.maxlevel, order='freq')
    for node in terminal_nodes:
        reconstructed_signal = node.reconstruct(update=False)
        if len(reconstructed_signal) > (emb_dim - 1) * lag + 10:
            try:
                lle = nolds.lyap_r(reconstructed_signal, emb_dim=emb_dim, lag=lag)
                lyapunov_results[node.path] = lle
            except Exception:
                lyapunov_results[node.path] = "Error"
        else:
            lyapunov_results[node.path] = "Signal too short"
    return lyapunov_results

# --- GUI Application Class ---
class EEGAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modular EEG Analysis Dashboard")
        self.geometry("800x700")

        # --- Style ---
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TNotebook.Tab", padding=[10, 5], font=('Helvetica', 11, 'bold'))
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=('Helvetica', 10))
        style.configure("TButton", padding=6, font=('Helvetica', 10, 'bold'))
        style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
        
        # --- Data Storage ---
        self.eeg_directory = ""
        self.valid_raw_data = []
        self.wpd_results = {}
        self.lyapunov_results = {}

        # --- Configuration ---
        self.config = {
            'file_extension': tk.StringVar(value='.edf'),
            'mother_wavelet': tk.StringVar(value='sym2'),
            'decomp_level': tk.IntVar(value=4),
            'embedding_dim': tk.IntVar(value=10),
            'time_delay': tk.IntVar(value=1)
        }

        # --- Main Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill="both")

        # Create a notebook for the three steps
        notebook = ttk.Notebook(main_frame)
        notebook.pack(expand=True, fill="both")

        # Create frames for each step
        self.step1_frame = self.create_step1_ui(notebook)
        self.step2_frame = self.create_step2_ui(notebook)
        self.step3_frame = self.create_step3_ui(notebook)

        notebook.add(self.step1_frame, text="Step 1: Load & Validate")
        notebook.add(self.step2_frame, text="Step 2: Wavelet Analysis")
        notebook.add(self.step3_frame, text="Step 3: Lyapunov & Summary")
        
        # Disable tabs 2 and 3 initially
        notebook.tab(1, state="disabled")
        notebook.tab(2, state="disabled")
        self.notebook = notebook

    # --- UI for Step 1: Data Loading ---
    def create_step1_ui(self, parent):
        frame = ttk.Frame(parent, padding="10")
        
        ttk.Label(frame, text="Step 1: Load and Validate EEG Files", style="Header.TLabel").pack(pady=5, anchor="w")
        
        # Directory Selection
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill="x", pady=5)
        ttk.Button(dir_frame, text="Select EEG Data Directory", command=self.select_directory).pack(side="left", padx=(0, 10))
        self.dir_label = ttk.Label(dir_frame, text="No directory selected.")
        self.dir_label.pack(side="left")

        # Run Button
        ttk.Button(frame, text="▶ Run Step 1: Validate Files", command=self.run_step1).pack(pady=10, anchor="w")

        # Dashboard / Results
        ttk.Label(frame, text="Dashboard: Validation Results", style="Header.TLabel").pack(pady=5, anchor="w")
        self.step1_results_text = scrolledtext.ScrolledText(frame, height=20, width=80, wrap=tk.WORD, state="disabled")
        self.step1_results_text.pack(expand=True, fill="both")
        
        return frame

    # --- UI for Step 2: Wavelet Analysis ---
    def create_step2_ui(self, parent):
        frame = ttk.Frame(parent, padding="10")
        
        ttk.Label(frame, text="Step 2: Wavelet Packet Decomposition", style="Header.TLabel").pack(pady=5, anchor="w")
        
        # File and Channel Selection for Visualization
        select_frame = ttk.Frame(frame)
        select_frame.pack(fill="x", pady=5, anchor="w")
        ttk.Label(select_frame, text="File to Analyze:").pack(side="left", padx=(0, 5))
        self.step2_file_combo = ttk.Combobox(select_frame, state="readonly", width=30)
        self.step2_file_combo.pack(side="left", padx=(0, 20))
        
        ttk.Button(frame, text="▶ Run Step 2 & Visualize", command=self.run_step2).pack(pady=10, anchor="w")

        # Dashboard / Results
        ttk.Label(frame, text="Dashboard: WPD Status", style="Header.TLabel").pack(pady=5, anchor="w")
        self.step2_results_text = scrolledtext.ScrolledText(frame, height=10, width=80, wrap=tk.WORD, state="disabled")
        self.step2_results_text.pack(expand=True, fill="both")
        
        return frame

    # --- UI for Step 3: Lyapunov Analysis ---
    def create_step3_ui(self, parent):
        frame = ttk.Frame(parent, padding="10")
        
        ttk.Label(frame, text="Step 3: Lyapunov Exponent Analysis & Summary", style="Header.TLabel").pack(pady=5, anchor="w")
        
        # File and Channel Selection
        select_frame = ttk.Frame(frame)
        select_frame.pack(fill="x", pady=5, anchor="w")
        ttk.Label(select_frame, text="File to Analyze:").pack(side="left", padx=(0, 5))
        self.step3_file_combo = ttk.Combobox(select_frame, state="readonly", width=30)
        self.step3_file_combo.pack(side="left", padx=(0, 20))

        ttk.Button(frame, text="▶ Run Step 3 & Generate Summary", command=self.run_step3).pack(pady=10, anchor="w")

        # Dashboard / Results
        ttk.Label(frame, text="Dashboard: Final Summary", style="Header.TLabel").pack(pady=5, anchor="w")
        self.step3_results_text = scrolledtext.ScrolledText(frame, height=20, width=80, wrap=tk.WORD, state="disabled")
        self.step3_results_text.pack(expand=True, fill="both")
        
        return frame
        
    # --- UI Logic and Actions ---
    def select_directory(self):
        directory = filedialog.askdirectory(title="Select EEG Data Folder")
        if directory:
            self.eeg_directory = directory
            self.dir_label.config(text=self.eeg_directory)

    def update_text_widget(self, widget, text, clear=False):
        widget.config(state="normal")
        if clear:
            widget.delete('1.0', tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def run_step1(self):
        if not self.eeg_directory:
            messagebox.showerror("Error", "Please select an EEG data directory first.")
            return

        self.update_text_widget(self.step1_results_text, "Starting validation...\n", clear=True)
        self.update() # Force UI update

        self.valid_raw_data, invalid_files = load_and_validate_eeg(
            self.eeg_directory, 
            self.config['file_extension'].get(),
            lambda msg: self.update_text_widget(self.step1_results_text, msg)
        )
        
        summary = f"\n--- Validation Complete ---\n"
        summary += f"Found {len(self.valid_raw_data)} valid files.\n"
        summary += f"Found {len(invalid_files)} invalid files.\n"
        self.update_text_widget(self.step1_results_text, summary)

        if self.valid_raw_data:
            self.notebook.tab(1, state="normal") # Enable Step 2
            filenames = [d.info['filename'] for d in self.valid_raw_data]
            self.step2_file_combo['values'] = filenames
            self.step2_file_combo.current(0)
        else:
            self.notebook.tab(1, state="disabled")
            self.notebook.tab(2, state="disabled")

    def run_step2(self):
        selected_filename = self.step2_file_combo.get()
        if not selected_filename:
            messagebox.showerror("Error", "No valid file selected.")
            return

        self.update_text_widget(self.step2_results_text, f"Running WPD for {selected_filename}...\n", clear=True)
        self.update()

        # Find the corresponding raw object
        raw_to_process = next((r for r in self.valid_raw_data if r.info['filename'] == selected_filename), None)
        
        if raw_to_process:
            self.wpd_results[selected_filename] = perform_wpd(
                raw_to_process, 
                self.config['mother_wavelet'].get(), 
                self.config['decomp_level'].get()
            )
            self.update_text_widget(self.step2_results_text, f"WPD complete for {selected_filename}.\n")
            
            # Visualization for the first channel
            first_ch_name = raw_to_process.ch_names[0]
            wp_to_plot = self.wpd_results[selected_filename][first_ch_name]
            self.update_text_widget(self.step2_results_text, f"Generating verification plot for channel '{first_ch_name}'...")
            plot_wpd_scalogram(wp_to_plot, raw_to_process.info['sfreq'], first_ch_name, selected_filename)

            # Enable Step 3
            self.notebook.tab(2, state="normal")
            self.step3_file_combo['values'] = list(self.wpd_results.keys())
            self.step3_file_combo.current(0)
        else:
            self.update_text_widget(self.step2_results_text, "Error: Could not find selected file data.", clear=True)

    def run_step3(self):
        selected_filename = self.step3_file_combo.get()
        if not selected_filename or selected_filename not in self.wpd_results:
            messagebox.showerror("Error", "Please run Step 2 for this file first.")
            return

        self.update_text_widget(self.step3_results_text, f"Running Lyapunov analysis for {selected_filename}...\n", clear=True)
        self.update()

        wpd_data_for_file = self.wpd_results[selected_filename]
        self.lyapunov_results[selected_filename] = {}
        
        chaotic_channels = 0
        for ch_name, wp in wpd_data_for_file.items():
            lle_results = calculate_lyapunov_for_wp(
                wp, 
                self.config['embedding_dim'].get(), 
                self.config['time_delay'].get()
            )
            self.lyapunov_results[selected_filename][ch_name] = lle_results
            # Check if any packet in the channel has a positive LLE
            if any(isinstance(v, (float, int)) and v > 0 for v in lle_results.values()):
                chaotic_channels += 1

        self.update_text_widget(self.step3_results_text, "Lyapunov analysis complete.\n\n")
        self.generate_final_summary(selected_filename, chaotic_channels)

    def generate_final_summary(self, filename, chaotic_count):
        summary = "--- FINAL SUMMARY DASHBOARD ---\n\n"
        
        # Step 1 KPI
        summary += "--- Step 1: Validation ---\n"
        summary += f"Total Valid Files Processed: {len(self.valid_raw_data)}\n\n"
        
        # Step 2 KPI
        summary += "--- Step 2: Wavelet Analysis ---\n"
        summary += f"File Analyzed: {filename}\n"
        summary += f"WPD Status: Complete. Verification plot was generated.\n\n"
        
        # Step 3 KPI
        summary += f"--- Step 3: Lyapunov Analysis ({filename}) ---\n"
        num_channels = len(self.lyapunov_results[filename])
        summary += f"Channels Analyzed: {num_channels}\n"
        summary += f"Channels with Chaotic Signature (LLE > 0): {chaotic_count}\n"
        
        # Detailed sample
        summary += "\nSample LLE Results:\n"
        sample_ch = list(self.lyapunov_results[filename].keys())[0]
        summary += f"  Channel: {sample_ch}\n"
        for i, (packet, lle) in enumerate(self.lyapunov_results[filename][sample_ch].items()):
            if i >= 5: break # show first 5 packets
            if isinstance(lle, float):
                summary += f"    - Packet '{packet}': LLE = {lle:.4f}\n"
            else:
                summary += f"    - Packet '{packet}': {lle}\n"
                
        self.update_text_widget(self.step3_results_text, summary, clear=True)

# --- Helper Function to Create Mock Data (for demonstration) ---
def create_mock_eeg_data(directory="eeg_temp_data_gui", num_files=5, num_bad_files=1):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    ch_names = [f'EEG {i:03}' for i in range(64)]
    sfreq = 256
    n_times = sfreq * 10
    for i in range(num_files):
        is_bad = i < num_bad_files
        file_name = f"test_eeg_{'bad' if is_bad else 'good'}_{i+1}.edf"
        file_path = os.path.join(directory, file_name)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        noise = np.random.randn(len(ch_names), n_times)
        if is_bad:
            noise[10, 100:200] = np.nan
        raw = mne.io.RawArray(noise, info)
        mne.export.export_raw(file_path, raw, fmt='edf', overwrite=True, verbose=False)
    return directory


# --- Main Execution Block ---
if __name__ == "__main__":
    # For demonstration, create mock data and inform the user.
    mock_dir = create_mock_eeg_data()
    
    app = EEGAnalysisApp()
    
    # Pre-populate the directory for the user for this demo run
    app.eeg_directory = mock_dir
    app.dir_label.config(text=f"Demo data loaded from: ./{mock_dir}")
    
    app.mainloop()

    # Clean up the dummy directory after the app closes
    if os.path.exists(mock_dir):
        shutil.rmtree(mock_dir)
