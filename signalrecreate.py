import time
import json
import serial
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
PORT_NAME = '/dev/cu.usbserial-0001' # Your Mac's verified USB port string
BAUD_RATE = 921600
OUTPUT_FILE = 'currentdata.json'
CAPTURE_DURATION_SECONDS = 10.0       # Strict 10-second run time
WINDOW_SIZE_SECONDS = 0.04           # 40ms scrolling window (2 full 50Hz cycles)

# --- Pre-allocate Memory Buffer ---
# 5000 Hz * 12 seconds safety margin = 60,000 slots
ESTIMATED_SAMPLES = int(5000 * (CAPTURE_DURATION_SECONDS + 2))
data_timestamps = np.zeros(ESTIMATED_SAMPLES, dtype=np.float64)
data_currents = np.zeros(ESTIMATED_SAMPLES, dtype=np.float32)

sample_counter = 0
start_time_us = None
plot_update_counter = 0

# --- Setup Live Plot Window ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(num='Live AC Current Capture (10s Mode)')
line, = ax.plot([], [], color='#0072B2', linewidth=1.5)

ax.set_xlabel('Time (Seconds)')
ax.set_ylabel('Current (Amperes)')
ax.set_title('Real-Time Waveform Oscilloscope')
ax.set_ylim(-35, 35)
ax.set_xlim(0, WINDOW_SIZE_SECONDS)

# Turn on interactive mode for live canvas updates
plt.ion()
plt.show(block=False)

# --- Initialize Serial Port ---
print(f"Connecting to ESP32 on {PORT_NAME}...")
try:
    # timeout=1 prevents hanging if the hardware is unplugged
    device = serial.Serial(PORT_NAME, BAUD_RATE, timeout=1)
    device.reset_input_buffer()
except Exception as e:
    print(f"Could not open serial port. Details: {e}")
    exit(1)

print("Recording started. Gathering exactly 10 seconds of data...")

# --- Data Acquisition Loop ---
try:
    while plt.fignum_exists(fig.number):
        if device.in_waiting > 0:
            try:
                # Read line and decode from bytes to string
                raw_data = device.readline().decode('utf-8', errors='ignore').strip()
                parsed_data = raw_data.split(',')
                
                if len(parsed_data) == 2:
                    t_us = float(parsed_data[0])
                    current_A = float(parsed_data[1])
                    
                    if start_time_us is None:
                        start_time_us = t_us
                        
                    t_seconds = (t_us - start_time_us) / 1000000.0
                    
                    # Break loop immediately if we hit the 10-second limit
                    if t_seconds >= CAPTURE_DURATION_SECONDS:
                        break
                    
                    # Save to RAM buffers safely
                    if sample_counter < ESTIMATED_SAMPLES:
                        data_timestamps[sample_counter] = t_us
                        data_currents[sample_counter] = current_A
                        sample_counter += 1
                    
                    # Compute rolling live display window
                    if t_seconds > WINDOW_SIZE_SECONDS:
                        ax.set_xlim(t_seconds - WINDOW_SIZE_SECONDS, t_seconds)
                    else:
                        ax.set_xlim(0, WINDOW_SIZE_SECONDS)
                        
                    # Update line plot coordinates
                    # Note: slicing dynamically matching the current live time frame
                    current_times = (data_timestamps[:sample_counter] - start_time_us) / 1000000.0
                    line.set_data(current_times, data_currents[:sample_counter])
                    
                    # Throttle UI rendering to maintain high data safety margin (every 50 points)
                    plot_update_counter += 1
                    if plot_update_counter >= 50:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plot_update_counter = 0
                        
            except (ValueError, IndexError):
                # Skip corrupt or incomplete serial lines cleanly
                continue
except KeyboardInterrupt:
    print("\nCapture interrupted manually by user.")

# --- Save Data to JSON File ---
print(f"\nCapture complete. Compiling dataset into {OUTPUT_FILE}...")

# Trim down unallocated buffer blocks
final_timestamps = data_timestamps[:sample_counter].tolist()
final_currents = data_currents[:sample_counter].tolist()

if sample_counter > 0:
    data_structure = {
        "Timestamp_us": final_timestamps,
        "Current_A": final_currents
    }
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data_structure, f, indent=4)
        print("Success! 10s dataset saved completely.")
    except Exception as e:
        print(f"Error saving file: {e}")
else:
    print("No samples were successfully saved.")

# Release physical serial line back to macOS
if 'device' in locals() and device.is_open:
    device.close()
    print("Serial port released safely.")
