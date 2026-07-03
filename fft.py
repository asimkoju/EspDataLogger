import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
input_file = 'currentdata.json'
fs = 5000  # Expected 5 kHz sampling rate

# --- Load and Parse JSON Data ---
if not os.path.exists(input_file):
    raise FileNotFoundError(f'Target file "{input_file}" not found. Please run your collection script first.')

print(f'Reading and parsing "{input_file}" into memory...')
with open(input_file, 'r') as file:
    raw_data = json.load(file)

# Extract raw vectors from JSON structural elements
# Handles both a dictionary of lists or an array of objects
if isinstance(raw_data, dict):
    timestamps_us = np.array(raw_data['Timestamp_us'], dtype=np.float64)
    current_A = np.array(raw_data['Current_A'], dtype=np.float64)
else:
    # Fallback if structural layout is an array of objects
    timestamps_us = np.array([row['Timestamp_us'] for row in raw_data], dtype=np.float64)
    current_A = np.array([row['Current_A'] for row in raw_data], dtype=np.float64)

N = len(current_A)

# Construct time domain vector in seconds starting from 0
time_seconds = (timestamps_us - timestamps_us[0]) / 1000000.0

print(f'Loaded {N} data points successfully (~{time_seconds[-1]:.2f} seconds).')

# --- Plot 1: Full Waveform Reconstruction ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.canvas.manager.set_window_title('Post-Processing Analysis Dashboard')

ax1.plot(time_seconds, current_A, color='#0072B4', linewidth=1)
ax1.grid(True)
ax1.set_xlabel('Time (Seconds)')
ax1.set_ylabel('Current (Amperes)')
ax1.set_title('Reconstructed 10-Second Continuous AC Signal')
ax1.set_xlim(0, time_seconds[-1])
ax1.set_ylim(np.min(current_A) - 5, np.max(current_A) + 5)

# --- Process Global FFT Calculation ---
print('Computing high-resolution Fast Fourier Transform...')

# Calculate the double-sided spectrum Y, then extract single-sided spectrum P1
Y = np.fft.fft(current_A)
P2 = np.abs(Y / N)

# In Python, floor(N/2)+1 slice syntax includes the index natively
half_N = int(np.floor(N / 2) + 1)
P1 = P2[:half_N].copy()
P1[1:-1] = 2 * P1[1:-1]  # Scale to map true peak amplitudes correctly (excluding DC and Nyquist)

# Generate accurate frequency grid coordinates
frequencies = np.fft.fftfreq(N, d=1/fs)[:half_N]
# Fix negative frequency components due to rfft parity symmetry
frequencies = np.abs(frequencies) 

# --- Plot 2: Detailed Frequency Spectrum ---
ax2.plot(frequencies, P1, color='#D95318', linewidth=1.5)
ax2.grid(True)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (Amperes Peak)')
ax2.set_title('Full Spectrum Frequency Domain Analysis (FFT)')
ax2.set_xlim(0, 500)  # Focus view from 0 to 500Hz to catch grid harmonics cleanly
ax2.set_ylim(0, np.max(P1) * 1.2)

plt.tight_layout()

# Locate the dominant fundamental frequency peak (ignoring the 0Hz DC component)
max_idx = np.argmax(P1[1:]) + 1  # Offset by 1 to skip DC
max_peak_mag = P1[max_idx]
fundamental_freq = frequencies[max_idx]

print('\n--- Signal Diagnostics ---')
print(f'Dominant Base Frequency detected at: {fundamental_freq:.2f} Hz')
print(f'Peak Current Amplitude at Base:      {max_peak_mag:.2f} A Peak')

# Render the window display panel
plt.show()