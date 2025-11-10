import numpy as np
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import os

try:
    # Try new moviepy structure first
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
    except ImportError:
        # Fall back to old moviepy structure
        from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Please install with: pip install moviepy")


def extract_audio_from_mp4(mp4_path, output_wav=None):
    """
    Extract audio from an MP4, M4A, or other video/audio file and save as WAV.
    
    Parameters:
    - mp4_path (str or Path): Path to the MP4/M4A file
    - output_wav (str or Path, optional): Path to save WAV file. If None, uses temp file.
    
    Returns:
    - tuple: (sample_rate, audio_data) where audio_data is numpy array
    """
    if not MOVIEPY_AVAILABLE:
        raise ImportError("moviepy is required. Install with: pip install moviepy")
    
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {mp4_path}")
    
    # Use temp file if output not specified
    if output_wav is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_wav = temp_file.name
        temp_file.close()
        cleanup_temp = True
    else:
        output_wav = Path(output_wav)
        cleanup_temp = False
    
    try:
        # Extract audio using moviepy
        # Try VideoFileClip first (for video files), then AudioFileClip (for audio-only files)
        audio = None
        video = None
        try:
            video = VideoFileClip(str(mp4_path))
            audio = video.audio
            if audio is None:
                # If no audio track in video, try as audio-only file
                video.close()
                video = None
                audio = AudioFileClip(str(mp4_path))
        except Exception:
            # If VideoFileClip fails, try as audio-only file
            if video is not None:
                video.close()
            audio = AudioFileClip(str(mp4_path))
        
        # Write audio to WAV file
        # Try with verbose parameter, fall back without it for newer moviepy versions
        try:
            audio.write_audiofile(str(output_wav), verbose=False, logger=None)
        except TypeError:
            # Newer moviepy versions don't accept verbose/logger parameters
            audio.write_audiofile(str(output_wav))
        audio.close()
        if video is not None:
            video.close()
        
        # Read the WAV file
        sample_rate, audio_data = wavfile.read(str(output_wav))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return sample_rate, audio_data
    
    finally:
        # Clean up temp file if created
        if cleanup_temp and os.path.exists(output_wav):
            os.unlink(output_wav)


def filter_audio_bandpass(audio_data, sample_rate, low_freq=5, high_freq=23):
    """
    Apply bandpass filter to audio signal, keeping only frequencies between low_freq and high_freq.
    
    Parameters:
    - audio_data (np.ndarray): Audio signal data
    - sample_rate (float): Sample rate in Hz
    - low_freq (float): Lower cutoff frequency in Hz (default: 5)
    - high_freq (float): Upper cutoff frequency in Hz (default: 23)
    
    Returns:
    - np.ndarray: Filtered audio signal
    """
    # Normalize audio data to float [-1, 1] range
    if audio_data.dtype == np.int16:
        audio_normalized = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_normalized = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_normalized = audio_data.astype(np.float32)
        if np.abs(audio_normalized).max() > 1.0:
            audio_normalized = audio_normalized / np.abs(audio_normalized).max()
    
    # Design bandpass filter
    # Nyquist frequency
    nyquist = sample_rate / 2.0
    
    # Normalize frequencies
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range [0, 1]
    low = max(0.0, min(low, 1.0))
    high = max(0.0, min(high, 1.0))
    
    if low >= high:
        raise ValueError(f"Invalid frequency range: low_freq ({low_freq} Hz) must be less than high_freq ({high_freq} Hz)")
    
    # Design Butterworth bandpass filter
    # Using 4th order filter for good frequency response
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio_normalized)
    
    # Convert back to original dtype if needed
    if audio_data.dtype == np.int16:
        filtered_audio = (filtered_audio * 32768.0).astype(np.int16)
    elif audio_data.dtype == np.int32:
        filtered_audio = (filtered_audio * 2147483648.0).astype(np.int32)
    
    return filtered_audio


def analyze_amplitude(audio_data, sample_rate):
    """
    Analyze amplitude characteristics of audio signal.
    
    Parameters:
    - audio_data (np.ndarray): Audio signal data
    - sample_rate (float): Sample rate in Hz
    
    Returns:
    - dict: Dictionary containing amplitude analysis results
    """
    # Normalize audio data to float [-1, 1] range
    if audio_data.dtype == np.int16:
        audio_normalized = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_normalized = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_normalized = audio_data.astype(np.float32)
        if np.abs(audio_normalized).max() > 1.0:
            audio_normalized = audio_normalized / np.abs(audio_normalized).max()
    
    # Time array
    duration = len(audio_normalized) / sample_rate
    time = np.linspace(0, duration, len(audio_normalized))
    
    # Check for NaN or Inf values
    if np.any(np.isnan(audio_normalized)) or np.any(np.isinf(audio_normalized)):
        # Replace NaN/Inf with zeros
        audio_normalized = np.nan_to_num(audio_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Amplitude statistics
    rms_amplitude = np.sqrt(np.mean(audio_normalized**2))
    peak_amplitude = np.abs(audio_normalized).max()
    mean_amplitude = np.mean(np.abs(audio_normalized))
    
    # Amplitude envelope (smoothed)
    # Calculate window size, ensuring it's valid
    window_size = int(sample_rate * 0.01)  # 10ms window
    if window_size % 2 == 0:
        window_size += 1
    
    # Ensure window_size is valid (must be less than signal length and >= polyorder + 1)
    min_window = 5  # Minimum window size for polyorder 3
    max_window = len(audio_normalized) - 1 if len(audio_normalized) > 1 else 1
    window_size = max(min_window, min(window_size, max_window))
    if window_size % 2 == 0:
        window_size -= 1  # Ensure odd
    window_size = max(5, window_size)  # Ensure at least 5 for polyorder 3
    
    # Try Savitzky-Golay filter, fall back to moving average if it fails
    try:
        if len(audio_normalized) >= window_size:
            envelope = signal.savgol_filter(np.abs(audio_normalized), window_size, 3)
        else:
            # Signal too short, use simple moving average
            envelope = np.convolve(np.abs(audio_normalized), np.ones(window_size)/window_size, mode='same')
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fall back to moving average if savgol fails
        print(f"Warning: Savitzky-Golay filter failed ({e}), using moving average instead")
        envelope = np.convolve(np.abs(audio_normalized), np.ones(window_size)/window_size, mode='same')
    
    return {
        'time': time,
        'amplitude': audio_normalized,
        'envelope': envelope,
        'rms': rms_amplitude,
        'peak': peak_amplitude,
        'mean': mean_amplitude,
        'duration': duration,
        'sample_rate': sample_rate
    }


def analyze_spectrum(audio_data, sample_rate, time_bin=0.1, overlap=0.5):
    """
    Analyze frequency spectrum of audio signal.
    
    Parameters:
    - audio_data (np.ndarray): Audio signal data
    - sample_rate (float): Sample rate in Hz
    - time_bin (float): Time bin size in seconds (default: 0.1 = 100 ms)
    - overlap (float): Overlap fraction (0-1)
    
    Returns:
    - dict: Dictionary containing spectrum analysis results
    """
    # Normalize audio data
    if audio_data.dtype == np.int16:
        audio_normalized = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_normalized = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_normalized = audio_data.astype(np.float32)
        if np.abs(audio_normalized).max() > 1.0:
            audio_normalized = audio_normalized / np.abs(audio_normalized).max()
    
    # Calculate nperseg based on time_bin (100 ms = 0.1 seconds)
    nperseg = int(sample_rate * time_bin)
    # Ensure nperseg is a power of 2 for efficient FFT (optional, but can improve performance)
    # Round to nearest power of 2
    nperseg = int(2 ** np.ceil(np.log2(nperseg)))
    
    noverlap = int(nperseg * overlap)
    
    # Compute power spectral density using Welch's method
    frequencies, power_spectrum = signal.welch(
        audio_normalized,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    
    # Convert to dB
    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-10)  # Add small value to avoid log(0)
    
    # Compute spectrogram with 100 ms time bins
    f, t, Sxx = signal.spectrogram(
        audio_normalized,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Find dominant frequencies
    peak_freq_idx = np.argmax(power_spectrum)
    peak_frequency = frequencies[peak_freq_idx]
    
    # Calculate spectral centroid (weighted mean frequency)
    spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    
    # Calculate bandwidth
    spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid)**2) * power_spectrum) / np.sum(power_spectrum))
    
    # Find dominant frequency at each time point
    # For each time point, find the frequency with maximum power
    dominant_freq_indices = np.argmax(Sxx_db, axis=0)  # Find max along frequency axis for each time
    dominant_frequencies = f[dominant_freq_indices]  # Get the actual frequencies
    dominant_powers_db = np.max(Sxx_db, axis=0)  # Get the maximum power in dB for each time
    
    return {
        'frequencies': frequencies,
        'power_spectrum': power_spectrum,
        'power_spectrum_db': power_spectrum_db,
        'spectrogram_freq': f,
        'spectrogram_time': t,
        'spectrogram': Sxx,
        'spectrogram_db': Sxx_db,
        'peak_frequency': peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'dominant_frequencies': dominant_frequencies,
        'dominant_powers_db': dominant_powers_db,
        'time_frequency_data': np.column_stack([t, dominant_frequencies, dominant_powers_db])
    }


def plot_amplitude_analysis(amp_results, title="Amplitude Analysis"):
    """
    Plot amplitude analysis results.
    
    Parameters:
    - amp_results (dict): Results from analyze_amplitude()
    - title (str): Plot title
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Amplitude Waveform', 'Amplitude Envelope'),
        vertical_spacing=0.1
    )
    
    # Waveform
    fig.add_trace(
        go.Scatter(
            x=amp_results['time'],
            y=amp_results['amplitude'],
            mode='lines',
            name='Waveform',
            line=dict(width=1)
        ),
        row=1, col=1
    )
    
    # Envelope
    fig.add_trace(
        go.Scatter(
            x=amp_results['time'],
            y=amp_results['envelope'],
            mode='lines',
            name='Envelope',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Add RMS and peak lines
    fig.add_hline(
        y=amp_results['rms'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"RMS: {amp_results['rms']:.4f}",
        row=2, col=1
    )
    fig.add_hline(
        y=amp_results['peak'],
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Peak: {amp_results['peak']:.4f}",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=True
    )
    
    return fig


def plot_spectrum_analysis(spec_results, title="Spectrum Analysis"):
    """
    Plot spectrum analysis results.
    
    Parameters:
    - spec_results (dict): Results from analyze_spectrum()
    - title (str): Plot title
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Power Spectral Density', 'Spectrogram'),
        vertical_spacing=0.1
    )
    
    # Power spectral density
    fig.add_trace(
        go.Scatter(
            x=spec_results['frequencies'],
            y=spec_results['power_spectrum_db'],
            mode='lines',
            name='PSD',
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    # Mark peak frequency
    peak_idx = np.argmax(spec_results['power_spectrum'])
    fig.add_vline(
        x=spec_results['peak_frequency'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Peak: {spec_results['peak_frequency']:.2f} Hz",
        row=1, col=1
    )
    
    # Spectrogram
    fig.add_trace(
        go.Heatmap(
            x=spec_results['spectrogram_time'],
            y=spec_results['spectrogram_freq'],
            z=spec_results['spectrogram_db'],
            colorscale='Viridis',
            name='Spectrogram'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text=title,
        showlegend=False
    )
    
    return fig


def analyze_mp4_audio(mp4_path, plot=True, time_bin=0.1, overlap=0.5, filter_freq_range=None):
    """
    Complete analysis of audio from an MP4, M4A, or other video/audio file.
    
    Parameters:
    - mp4_path (str or Path): Path to the MP4/M4A file
    - plot (bool): Whether to create plots
    - time_bin (float): Time bin size in seconds for spectrogram (default: 0.1 = 100 ms)
    - overlap (float): Overlap fraction for spectrum analysis (0-1)
    - filter_freq_range (tuple, optional): (low_freq, high_freq) in Hz to filter audio. 
                                           If None, no filtering is applied. Default: (5, 23)
    
    Returns:
    - dict: Dictionary containing all analysis results
    """
    print(f"Extracting audio from: {mp4_path}")
    sample_rate, audio_data = extract_audio_from_mp4(mp4_path)
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Audio shape: {audio_data.shape}")
    
    # Apply bandpass filter if requested
    if filter_freq_range is not None:
        low_freq, high_freq = filter_freq_range
        print(f"Applying bandpass filter: {low_freq}-{high_freq} Hz")
        audio_data = filter_audio_bandpass(audio_data, sample_rate, low_freq=low_freq, high_freq=high_freq)
        print("Filter applied successfully")
    else:
        # Default filter: 5-23 Hz
        audio_data = filter_audio_bandpass(audio_data, sample_rate, low_freq=5, high_freq=23)
    
    print("Analyzing amplitude...")
    amp_results = analyze_amplitude(audio_data, sample_rate)
    
    print("Analyzing spectrum...")
    print(f"Using time bin: {time_bin*1000:.0f} ms for spectrogram")
    spec_results = analyze_spectrum(audio_data, sample_rate, time_bin=time_bin, overlap=overlap)
    
    # Print summary statistics
    print("\n=== Amplitude Statistics ===")
    print(f"RMS Amplitude: {amp_results['rms']:.4f}")
    print(f"Peak Amplitude: {amp_results['peak']:.4f}")
    print(f"Mean Amplitude: {amp_results['mean']:.4f}")
    
    print("\n=== Spectrum Statistics ===")
    print(f"Peak Frequency: {spec_results['peak_frequency']:.2f} Hz")
    print(f"Spectral Centroid: {spec_results['spectral_centroid']:.2f} Hz")
    print(f"Spectral Bandwidth: {spec_results['spectral_bandwidth']:.2f} Hz")
    
    # Extract time-frequency data (main frequency and power at each time point)
    time_freq_array = spec_results['time_frequency_data']  # Shape: (n_time_points, 3) - [time, frequency, power_db]
    print(f"\n=== Time-Frequency Data ===")
    print(f"Array shape: {time_freq_array.shape}")
    print(f"Format: [time (s), dominant_frequency (Hz), power (dB)]")
    print(f"First 5 time points:")
    print(time_freq_array[:5])
    
    results = {
        'amplitude': amp_results,
        'spectrum': spec_results,
        'sample_rate': sample_rate,
        'audio_data': audio_data,
        'time_frequency_array': time_freq_array  # Array with [time, dominant_freq, power_db] for each time point
    }
    
    if plot:
        print("\nGenerating plots...")
        amp_fig = plot_amplitude_analysis(amp_results, title=f"Amplitude Analysis: {Path(mp4_path).name}")
        spec_fig = plot_spectrum_analysis(spec_results, title=f"Spectrum Analysis: {Path(mp4_path).name}")
        results['amplitude_plot'] = amp_fig
        results['spectrum_plot'] = spec_fig
    
    return results


# Example usage
if __name__ == "__main__":
    # Analyze the provided M4A file
    mp4_file = r"C:\Users\Owner\Downloads\10 Nov at 16-43.m4a"
    results = analyze_mp4_audio(mp4_file, filter_freq_range=(7*100,22*100), time_bin=0.2, overlap=0.3)
    
    # Access the time-frequency array
    # Each row contains: [time (s), dominant_frequency (Hz), power (dB)]
    time_freq_array = results['time_frequency_array']
    print(f"\nTime-frequency array shape: {time_freq_array.shape}")
    print(f"Access as: results['time_frequency_array']")
    print(f"Columns: [time, dominant_frequency, power_db]")
    
    # Validate stimulus order
    # Expected stimulus frequencies in kHz (in order by time)
    expected_stims_khz = np.array([
        0.7070, 1.2246, 1.2246, 0.7070, 1.2246, 0.7891, 1.0972, 0.8807, 1.3668, 1.7026, 1.9003,
        1.2246, 1.0972, 1.3668, 2.1210, 0.7070, 1.2246, 0.7891, 1.0972, 1.3668, 1.2246, 0.8807,
        1.0972, 1.7026, 1.3668, 1.9003, 2.1210, 1.2246, 0.7070, 0.7891, 1.0972, 0.8807, 1.3668,
        1.7026, 1.2246, 1.0972, 1.9003, 1.3668, 2.1210, 0.7070, 0.7891
    ])
    expected_stims_hz = expected_stims_khz * 1000
    
    # Sort time-frequency array by time
    time_sorted_idx = np.argsort(time_freq_array[:, 0])
    time_sorted_array = time_freq_array[time_sorted_idx]
    
    # Account for 20 second offset at the start of recording
    start_time_offset = 20.0  # seconds
    # Filter to only include data after the offset
    time_mask = time_sorted_array[:, 0] >= start_time_offset
    time_sorted_array = time_sorted_array[time_mask]
    
    # Adjust times to be relative to stimulus start (subtract offset)
    time_sorted_array[:, 0] = time_sorted_array[:, 0] - start_time_offset
    
    # Extract detected frequencies (in Hz) in time order
    detected_freqs_hz = time_sorted_array[:, 1]
    detected_times = time_sorted_array[:, 0]  # Now relative to stimulus start
    
    # Convert to kHz for comparison
    detected_freqs_khz = detected_freqs_hz / 1000
    
    print(f"\n=== Stimulus Order Validation ===")
    print(f"Start time offset: {start_time_offset} seconds (stimuli start after this time)")
    print(f"Expected number of stimuli: {len(expected_stims_khz)}")
    print(f"Detected number of time points (after offset): {len(detected_freqs_khz)}")
    
    # Find closest match for each detected frequency to expected frequencies
    # Allow some tolerance for frequency matching (e.g., within 5% or 50 Hz)
    tolerance_percent = 0.05
    tolerance_hz = 50
    
    matched_indices = []
    matched_freqs = []
    for i, detected_freq_hz in enumerate(detected_freqs_hz):
        # Find closest expected frequency
        freq_diffs = np.abs(expected_stims_hz - detected_freq_hz)
        closest_idx = np.argmin(freq_diffs)
        closest_diff = freq_diffs[closest_idx]
        
        # Check if within tolerance
        tolerance = max(detected_freq_hz * tolerance_percent, tolerance_hz)
        if closest_diff <= tolerance:
            matched_indices.append(closest_idx)
            matched_freqs.append(expected_stims_hz[closest_idx] / 1000)
        else:
            matched_indices.append(-1)  # No match
            matched_freqs.append(detected_freq_hz / 1000)
    
    # Count matches
    valid_matches = [idx for idx in matched_indices if idx >= 0]
    print(f"Valid frequency matches: {len(valid_matches)} / {len(detected_freqs_hz)}")
    
    # Check order of matched frequencies
    if len(valid_matches) > 0:
        matched_order = [expected_stims_khz[idx] for idx in valid_matches]
        print(f"\nExpected order (first 10): {expected_stims_khz[:10]}")
        print(f"Detected order (first 10): {[f'{f:.4f}' for f in detected_freqs_khz[:10]]}")
        print(f"Matched order (first 10): {[f'{f:.4f}' for f in matched_freqs[:10]]}")
        
        # Compare order
        if len(matched_order) == len(expected_stims_khz):
            order_match = np.allclose(matched_order, expected_stims_khz, atol=0.01)
            if order_match:
                print("\n✓ VALIDATION PASSED: Detected frequencies match expected order!")
            else:
                print("\n✗ VALIDATION FAILED: Order does not match expected sequence")
                print(f"First mismatch at position: {np.where(~np.isclose(matched_order, expected_stims_khz, atol=0.01))[0]}")
        else:
            print(f"\n⚠ Partial match: {len(matched_order)} stimuli matched out of {len(expected_stims_khz)} expected")
    
    # Print detailed comparison for first few stimuli
    print(f"\n=== Detailed Comparison (first 15) ===")
    print("Time(s) | Detected(kHz) | Expected(kHz) | Match")
    print("-" * 50)
    for i in range(min(100, len(detected_freqs_khz), len(expected_stims_khz))):
        detected = detected_freqs_khz[i]
        expected = expected_stims_khz[i]
        time = detected_times[i]
        match = "✓" if abs(detected - expected) < 0.1 else "✗"
        print(f"{time:7.3f} | {detected:13.4f} | {expected:13.4f} | {match}")
    
    # Plot frequency vs dB (power)
    frequencies = time_freq_array[:, 1]  # Column 1: dominant frequencies
    powers_db = time_freq_array[:, 2]    # Column 2: power in dB
    
    # Target frequencies in kHz (convert to Hz)
    target_freqs_khz = np.array([0.7070, 0.7891, 0.8807, 1.0972, 1.2246, 1.3668, 
                                 1.7026, 1.9003, 2.1210])
    target_freqs_hz = target_freqs_khz * 1000  # Convert to Hz
    
    # Define bandwidth around each target frequency (as percentage or fixed Hz)
    # Using 5% width on each side, or minimum 50 Hz
    bandwidth_percent = 0.05  # 5% on each side
    min_bandwidth_hz = 50  # Minimum 50 Hz on each side
    
    # Create mask for frequencies within the target ranges
    mask = np.zeros(len(frequencies), dtype=bool)
    for target_freq in target_freqs_hz:
        bandwidth = max(target_freq * bandwidth_percent, min_bandwidth_hz)
        freq_mask = (frequencies >= target_freq - bandwidth) & (frequencies <= target_freq + bandwidth)
        mask = mask | freq_mask
    
    # Filter data to focus on target frequencies
    frequencies_focused = frequencies[mask]
    powers_db_focused = powers_db[mask]
    
    print(f"\nTarget frequencies: {target_freqs_khz} kHz ({target_freqs_hz} Hz)")
    print(f"Data points in target ranges: {len(frequencies_focused)} / {len(frequencies)}")
    
    # Sort by frequency for better visualization
    sort_idx = np.argsort(frequencies_focused)
    frequencies_sorted = frequencies_focused[sort_idx]
    powers_db_sorted = powers_db_focused[sort_idx]
    
    # Find peaks in the data
    # Use scipy's find_peaks to identify local maxima
    # Find peaks - adjust parameters as needed
    # height: minimum peak height, distance: minimum distance between peaks
    peaks, properties = signal.find_peaks(
        powers_db_sorted,
        height=np.percentile(powers_db_sorted, 50),  # At least 50th percentile
        distance=len(frequencies_sorted) // 20,  # Minimum distance between peaks
        prominence=np.std(powers_db_sorted) * 0.5  # Minimum prominence
    )
    
    peak_frequencies = frequencies_sorted[peaks]
    peak_powers = powers_db_sorted[peaks]
    
    print(f"\nFound {len(peaks)} peaks")
    
    # Fit a polynomial (non-linear) curve to the peaks
    # Using log space for frequency since we'll use log scale
    log_freq_peaks = np.log10(peak_frequencies)
    
    # Fit polynomial (degree 2-4 typically works well)
    poly_coeffs = None
    poly_func = None
    if len(peaks) >= 3:
        # Try polynomial fit
        try:
            # Fit polynomial in log space
            poly_degree = min(1, len(peaks)-1)
            poly_coeffs = np.polyfit(log_freq_peaks, peak_powers, poly_degree)
            poly_func = np.poly1d(poly_coeffs)
            
            # Generate smooth curve for plotting
            log_freq_fit = np.linspace(log_freq_peaks.min(), log_freq_peaks.max(), 200)
            freq_fit = 10 ** log_freq_fit
            power_fit = poly_func(log_freq_fit)
            
            print(f"\n=== Fitted Curve Coefficients ===")
            print(f"Polynomial degree: {poly_degree}")
            print(f"Coefficients (highest order first):")
            # Print coefficients in readable format
            for i, coeff in enumerate(poly_coeffs):
                order = len(poly_coeffs) - 1 - i
                if order == 0:
                    print(f"  Constant term: {coeff:.6f}")
                elif order == 1:
                    print(f"  Linear term (log10(freq)): {coeff:.6f}")
                else:
                    print(f"  log10(freq)^{order} term: {coeff:.6f}")
            print(f"\nFull coefficient array: {poly_coeffs}")
            print(f"\nEquation: power_db = ", end="")
            terms = []
            for i, coeff in enumerate(poly_coeffs):
                order = len(poly_coeffs) - 1 - i
                if order == 0:
                    terms.append(f"{coeff:.6f}")
                elif order == 1:
                    terms.append(f"{coeff:.6f} * log10(freq)")
                else:
                    terms.append(f"{coeff:.6f} * log10(freq)^{order}")
            print(" + ".join(terms))
            
        except Exception as e:
            print(f"Polynomial fit failed: {e}, using linear fit instead")
            # Fallback to linear fit
            poly_coeffs = np.polyfit(log_freq_peaks, peak_powers, 1)
            poly_func = np.poly1d(poly_coeffs)
            log_freq_fit = np.linspace(log_freq_peaks.min(), log_freq_peaks.max(), 200)
            freq_fit = 10 ** log_freq_fit
            power_fit = poly_func(log_freq_fit)
            
            print(f"\n=== Fitted Curve Coefficients (Linear) ===")
            print(f"Slope: {poly_coeffs[0]:.6f}")
            print(f"Intercept: {poly_coeffs[1]:.6f}")
            print(f"Equation: power_db = {poly_coeffs[0]:.6f} * log10(freq) + {poly_coeffs[1]:.6f}")
            print(f"Full coefficient array: {poly_coeffs}")
    else:
        print("Not enough peaks for curve fitting")
        freq_fit = np.array([])
        power_fit = np.array([])
    
    # Create plot with log scale x-axis
    fig = go.Figure()
    
    # Scatter plot of all original data (for context)
    sort_idx_all = np.argsort(frequencies)
    fig.add_trace(go.Scatter(
        x=frequencies[sort_idx_all],
        y=powers_db[sort_idx_all],
        mode='markers',
        name='All Data',
        marker=dict(size=1, color='lightgray', opacity=0.3)
    ))
    
    # Scatter plot of focused data (target frequencies)
    fig.add_trace(go.Scatter(
        x=frequencies_sorted,
        y=powers_db_sorted,
        mode='markers',
        name='Target Frequencies',
        marker=dict(size=3, color='lightblue', opacity=0.7)
    ))
    
    # Add vertical lines for target frequencies
    for target_freq in target_freqs_hz:
        fig.add_vline(
            x=target_freq,
            line_dash="dash",
            line_color="green",
            line_width=1,
            opacity=0.5,
            annotation_text=f"{target_freq/1000:.3f} kHz"
        )
    
    # Highlight peaks
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=peak_frequencies,
            y=peak_powers,
            mode='markers',
            name='Peaks',
            marker=dict(size=8, color='red', symbol='diamond')
        ))
    
    # Fitted curve
    if len(freq_fit) > 0:
        fig.add_trace(go.Scatter(
            x=freq_fit,
            y=power_fit,
            mode='lines',
            name='Fitted Curve',
            line=dict(width=3, color='red')
        ))
    
    fig.update_layout(
        title='Frequency vs Power (dB) - Log Scale',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (dB)',
        xaxis_type='log',  # Set x-axis to log scale
        height=600
    )
    fig.show()
    
    # Return coefficients if available
    if poly_coeffs is not None:
        print(f"\n=== Returning Coefficients ===")
        print(f"poly_coeffs = {poly_coeffs}")
        print(f"Access as: poly_coeffs variable in this scope")
        # Store in a way that can be accessed
        fitted_coefficients = {
            'coefficients': poly_coeffs,
            'degree': len(poly_coeffs) - 1,
            'equation_type': 'polynomial in log10(frequency)',
            'poly_func': poly_func
        }
        print(f"Stored in fitted_coefficients dictionary")
    
    # Find peaks in the power spectrum and spectrogram
    print(f"\n=== Finding Peaks in Power Spectrum ===")
    spec_results = results['spectrum']
    
    # Power Spectral Density (1D)
    power_spectrum_db = spec_results['power_spectrum_db']
    frequencies = spec_results['frequencies']
    
    # Find peaks in power spectrum
    power_peaks, power_properties = signal.find_peaks(
        power_spectrum_db,
        height=np.percentile(power_spectrum_db, 75),  # At least 75th percentile
        distance=len(frequencies) // 50,  # Minimum distance between peaks
        prominence=np.std(power_spectrum_db) * 0.5  # Minimum prominence
    )
    
    peak_freqs_psd = frequencies[power_peaks]
    peak_powers_psd = power_spectrum_db[power_peaks]
    
    # Sort by power
    sort_idx_psd = np.argsort(peak_powers_psd)[::-1]
    peak_freqs_psd_sorted = peak_freqs_psd[sort_idx_psd]
    peak_powers_psd_sorted = peak_powers_psd[sort_idx_psd]
    
    print(f"Found {len(power_peaks)} peaks in power spectrum")
    print(f"Peak power range: {peak_powers_psd_sorted.min():.2f} to {peak_powers_psd_sorted.max():.2f} dB")
    
    # Print top peaks
    n_top_peaks_psd = min(20, len(peak_freqs_psd_sorted))
    print(f"\nTop {n_top_peaks_psd} peaks in power spectrum (by power):")
    print("Frequency(Hz) | Power(dB)")
    print("-" * 35)
    for i in range(n_top_peaks_psd):
        print(f"{peak_freqs_psd_sorted[i]:13.2f} | {peak_powers_psd_sorted[i]:9.2f}")
    
    # Store power spectrum peaks
    power_spectrum_peaks = {
        'frequencies': peak_freqs_psd_sorted,
        'powers_db': peak_powers_psd_sorted,
        'all_peaks': {
            'frequencies': peak_freqs_psd,
            'powers_db': peak_powers_psd
        }
    }
    results['power_spectrum_peaks'] = power_spectrum_peaks
    
    # Get the spectrum plot figure for adding traces
    psd_fig = results['spectrum_plot']
    
    # Find peaks at target frequencies and fit curve
    print(f"\n=== Finding Peaks at Target Frequencies and Fitting Curve ===")
    # Target frequencies in kHz (from earlier in the code)
    target_freqs_khz = np.array([0.7070, 0.7891, 0.8807, 1.0972, 1.2246, 1.3668, 
                                 1.7026, 1.9003, 2.1210])
    target_freqs_hz = target_freqs_khz * 1000
    
    # Match power spectrum peaks to target frequencies
    tolerance_percent = 0.05  # 5% tolerance
    tolerance_hz = 50  # Minimum 50 Hz
    
    matched_target_peaks = []
    matched_target_freqs = []
    matched_target_powers = []
    
    for target_freq_hz in target_freqs_hz:
        # Find closest peak in power spectrum
        freq_diffs = np.abs(peak_freqs_psd - target_freq_hz)
        closest_idx = np.argmin(freq_diffs)
        closest_diff = freq_diffs[closest_idx]
        
        # Check if within tolerance
        tolerance = max(target_freq_hz * tolerance_percent, tolerance_hz)
        if closest_diff <= tolerance:
            matched_target_peaks.append(closest_idx)
            matched_target_freqs.append(peak_freqs_psd[closest_idx])
            matched_target_powers.append(peak_powers_psd[closest_idx])
        else:
            # If no peak found, use the power at the target frequency directly
            # Interpolate from power spectrum
            closest_freq_idx = np.argmin(np.abs(frequencies - target_freq_hz))
            matched_target_freqs.append(target_freq_hz)
            matched_target_powers.append(power_spectrum_db[closest_freq_idx])
            print(f"Warning: No peak found near {target_freq_hz/1000:.4f} kHz, using interpolated value")
    
    matched_target_freqs = np.array(matched_target_freqs)
    matched_target_powers = np.array(matched_target_powers)
    
    print(f"Matched {len(matched_target_freqs)} target frequencies to power spectrum peaks")
    print(f"\nTarget frequencies and their power spectrum values:")
    print("Frequency(kHz) | Power(dB)")
    print("-" * 35)
    for i in range(len(matched_target_freqs)):
        print(f"{matched_target_freqs[i]/1000:13.4f} | {matched_target_powers[i]:9.2f}")
    
    # Fit curve to target frequency peaks
    if len(matched_target_freqs) >= 2:
        # Use log space for frequency (since we use log scale)
        log_freq_targets = np.log10(matched_target_freqs)
        
        # Fit polynomial (linear for now, can be adjusted)
        poly_degree_target = min(1, len(matched_target_freqs) - 1)
        try:
            poly_coeffs_target = np.polyfit(log_freq_targets, matched_target_powers, poly_degree_target)
            poly_func_target = np.poly1d(poly_coeffs_target)
            
            # Generate smooth curve for plotting
            log_freq_fit_target = np.linspace(log_freq_targets.min(), log_freq_targets.max(), 200)
            freq_fit_target = 10 ** log_freq_fit_target
            power_fit_target = poly_func_target(log_freq_fit_target)
            
            print(f"\n=== Fitted Curve to Target Frequency Peaks ===")
            print(f"Polynomial degree: {poly_degree_target}")
            print(f"Coefficients (highest order first):")
            for i, coeff in enumerate(poly_coeffs_target):
                order = len(poly_coeffs_target) - 1 - i
                if order == 0:
                    print(f"  Constant term: {coeff:.6f}")
                elif order == 1:
                    print(f"  Linear term (log10(freq)): {coeff:.6f}")
                else:
                    print(f"  log10(freq)^{order} term: {coeff:.6f}")
            print(f"\nFull coefficient array: {poly_coeffs_target}")
            
            # Report the fitted curve equation on target points
            print(f"\n=== Fitted Curve Equation (on Target Points) ===")
            print(f"power_db = ", end="")
            terms = []
            for i, coeff in enumerate(poly_coeffs_target):
                order = len(poly_coeffs_target) - 1 - i
                if order == 0:
                    terms.append(f"{coeff:.6f}")
                elif order == 1:
                    terms.append(f"{coeff:.6f} * log10(freq)")
                else:
                    terms.append(f"{coeff:.6f} * log10(freq)^{order}")
            print(" + ".join(terms))
            
            # Also report in terms of frequency (not log)
            if poly_degree_target == 1:
                print(f"\nIn terms of frequency (Hz):")
                print(f"power_db = {poly_coeffs_target[0]:.6f} * log10(freq) + {poly_coeffs_target[1]:.6f}")
                print(f"Where freq is in Hz")
            
            # Show fitted values at target frequencies
            print(f"\nFitted values at target frequencies:")
            print("Frequency(kHz) | Actual Power(dB) | Fitted Power(dB) | Difference")
            print("-" * 65)
            for i in range(len(matched_target_freqs)):
                fitted_power = poly_func_target(np.log10(matched_target_freqs[i]))
                diff = matched_target_powers[i] - fitted_power
                print(f"{matched_target_freqs[i]/1000:13.4f} | {matched_target_powers[i]:15.2f} | {fitted_power:15.2f} | {diff:10.2f}")
            
            # Store target frequency fit
            target_freq_fit = {
                'coefficients': poly_coeffs_target,
                'degree': poly_degree_target,
                'equation_type': 'polynomial in log10(frequency)',
                'poly_func': poly_func_target,
                'target_frequencies': matched_target_freqs,
                'target_powers': matched_target_powers,
                'fitted_frequencies': freq_fit_target,
                'fitted_powers': power_fit_target
            }
            results['target_frequency_fit'] = target_freq_fit
            
            # Add fitted curve to PSD plot
            psd_fig.add_trace(go.Scatter(
                x=freq_fit_target,
                y=power_fit_target,
                mode='lines',
                name='Target Freq Fit',
                line=dict(width=3, color='orange', dash='dash')
            ), row=1, col=1)
            
            # Highlight target frequency peaks
            psd_fig.add_trace(go.Scatter(
                x=matched_target_freqs,
                y=matched_target_powers,
                mode='markers',
                name='Target Frequencies',
                marker=dict(
                    size=12,
                    color='orange',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                showlegend=True
            ), row=1, col=1)
            
        except Exception as e:
            print(f"Curve fitting failed: {e}")
    else:
        print("Not enough target frequencies matched for curve fitting")
    
    # Removed red diamond markers for PSD peaks (as requested)
    
    # Find peaks in the spectrogram
    print(f"\n=== Finding Peaks in Spectrogram ===")
    spectrogram_db = spec_results['spectrogram_db']  # 2D array: [frequencies x time]
    spectrogram_freq = spec_results['spectrogram_freq']
    spectrogram_time = spec_results['spectrogram_time']
    
    # Find peaks in 2D spectrogram
    # Method 1: Find local maxima in the 2D array
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import generate_binary_structure, binary_erosion
    
    # Define neighborhood for peak detection
    neighborhood_size = 5  # Size of neighborhood to check for peaks
    threshold_percentile = 75  # Only consider peaks above this percentile
    
    # Create a threshold mask
    threshold = np.percentile(spectrogram_db, threshold_percentile)
    
    # Find local maxima
    local_maxima = maximum_filter(spectrogram_db, size=neighborhood_size) == spectrogram_db
    local_maxima = local_maxima & (spectrogram_db > threshold)
    
    # Get peak coordinates
    peak_freq_indices, peak_time_indices = np.where(local_maxima)
    peak_frequencies = spectrogram_freq[peak_freq_indices]
    peak_times = spectrogram_time[peak_time_indices]
    peak_powers = spectrogram_db[peak_freq_indices, peak_time_indices]
    
    print(f"Found {len(peak_frequencies)} peaks in spectrogram")
    print(f"Peak power range: {peak_powers.min():.2f} to {peak_powers.max():.2f} dB")
    
    # Sort peaks by power (descending)
    sort_idx = np.argsort(peak_powers)[::-1]
    peak_frequencies_sorted = peak_frequencies[sort_idx]
    peak_times_sorted = peak_times[sort_idx]
    peak_powers_sorted = peak_powers[sort_idx]
    
    # Print top peaks
    n_top_peaks = min(20, len(peak_frequencies_sorted))
    print(f"\nTop {n_top_peaks} peaks (by power):")
    print("Time(s) | Frequency(Hz) | Power(dB)")
    print("-" * 45)
    for i in range(n_top_peaks):
        print(f"{peak_times_sorted[i]:7.3f} | {peak_frequencies_sorted[i]:13.2f} | {peak_powers_sorted[i]:9.2f}")
    
    # Store peaks in results
    spectrogram_peaks = {
        'frequencies': peak_frequencies_sorted,
        'times': peak_times_sorted,
        'powers_db': peak_powers_sorted,
        'all_peaks': {
            'frequencies': peak_frequencies,
            'times': peak_times,
            'powers_db': peak_powers
        }
    }
    results['spectrogram_peaks'] = spectrogram_peaks
    
    # Add peaks to the spectrum plot
    # Get the figure from results
    spec_fig = results['spectrum_plot']
    
    # Add peak markers to the spectrogram (row 2, col 1 is the spectrogram subplot)
    spec_fig.add_trace(go.Scatter(
        x=peak_times_sorted[:n_top_peaks],  # Top peaks only for visibility
        y=peak_frequencies_sorted[:n_top_peaks],
        mode='markers',
        name='Peaks',
        marker=dict(
            size=8,
            color='red',
            symbol='x',
            line=dict(width=2, color='white')
        ),
        showlegend=True
    ), row=2, col=1)  # Add to spectrogram subplot
    
    results['amplitude_plot'].show()
    results['spectrum_plot'].show()

