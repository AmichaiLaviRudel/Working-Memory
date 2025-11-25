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


def load_wav_file(wav_path):
    """
    Load audio from a WAV file.
    
    Parameters:
    - wav_path (str or Path): Path to the WAV file
    
    Returns:
    - tuple: (sample_rate, audio_data) where audio_data is numpy array
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(str(wav_path))
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    return sample_rate, audio_data


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
    
    # Validate frequency range before normalization
    if low_freq >= high_freq:
        raise ValueError(f"Invalid frequency range: low_freq ({low_freq} Hz) must be less than high_freq ({high_freq} Hz)")
    
    if low_freq >= nyquist:
        raise ValueError(f"Low frequency ({low_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz). "
                        f"Consider resampling audio to at least {low_freq * 2} Hz sample rate.")
    
    if high_freq >= nyquist:
        raise ValueError(f"High frequency ({high_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz). "
                        f"Consider resampling audio to at least {high_freq * 2} Hz sample rate.")
    
    # Normalize frequencies
    low = low_freq / nyquist
    high = high_freq / nyquist
    
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


def plot_spectrum_analysis(spec_results, target_freqs_hz=None, matched_target_freqs=None, 
                          matched_target_powers=None, title="Spectrum Analysis"):
    """
    Plot spectrum analysis results with target frequencies.
    
    Parameters:
    - spec_results (dict): Results from analyze_spectrum()
    - target_freqs_hz (np.ndarray, optional): Array of target frequencies in Hz
    - matched_target_freqs (np.ndarray, optional): Matched target frequencies in Hz
    - matched_target_powers (np.ndarray, optional): Power at matched target frequencies in dB
    - title (str): Plot title
    """
    fig = go.Figure()
    
    # Power spectral density
    fig.add_trace(
        go.Scatter(
            x=spec_results['frequencies'],
            y=spec_results['power_spectrum_db'],
            mode='lines',
            name='PSD',
            line=dict(width=2, color='blue')
        )
    )
    
    # Add vertical lines for target frequencies if provided
    if target_freqs_hz is not None:
        for target_freq in target_freqs_hz:
            fig.add_vline(
                x=target_freq,
                line_dash="dash",
                line_color="green",
                line_width=1,
                opacity=0.5
            )
    
    # Highlight matched target frequency peaks if provided
    if matched_target_freqs is not None and matched_target_powers is not None:
        fig.add_trace(go.Scatter(
            x=matched_target_freqs,
            y=matched_target_powers,
            mode='markers',
            name='Target Frequencies',
            marker=dict(
                size=12,
                color='orange',
                symbol='star',
                line=dict(width=2, color='black')
            )
        ))
    
    fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    fig.update_yaxes(title_text="Power (dB)")
    
    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=True
    )
    
    return fig


def analyze_audio_file(audio_path, plot=True, time_bin=0.1, overlap=0.5, filter_freq_range=None, 
                       target_freqs_khz=None):
    """
    Complete analysis of audio from a WAV, MP4, M4A, or other audio file.
    
    Parameters:
    - audio_path (str or Path): Path to the audio file (WAV/MP4/M4A)
    - plot (bool): Whether to create plots
    - time_bin (float): Time bin size in seconds for spectrogram (default: 0.1 = 100 ms)
    - overlap (float): Overlap fraction for spectrum analysis (0-1)
    - filter_freq_range (tuple, optional): (low_freq, high_freq) in Hz to filter audio. 
                                           If None, applies default filter (5, 23) Hz
    - target_freqs_khz (np.ndarray, optional): Array of target frequencies in kHz to analyze
    
    Returns:
    - dict: Dictionary containing all analysis results
    """
    audio_path = Path(audio_path)
    file_ext = audio_path.suffix.lower()
    
    print(f"Loading audio from: {audio_path}")
    
    # Determine file type and load accordingly
    if file_ext == '.wav':
        print("Detected WAV file, loading directly...")
        sample_rate, audio_data = load_wav_file(audio_path)
    else:
        print(f"Detected {file_ext} file, extracting audio...")
        sample_rate, audio_data = extract_audio_from_mp4(audio_path)
    
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
        print("Applying default bandpass filter: 5-23 Hz")
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
    
    # Analyze target frequencies if provided
    matched_target_freqs = None
    matched_target_powers = None
    target_freqs_hz = None
    
    if target_freqs_khz is not None:
        print(f"\n=== Analyzing Target Frequencies ===")
        target_freqs_hz = target_freqs_khz * 1000  # Convert to Hz
        
        # Find peaks in power spectrum
        power_spectrum_db = spec_results['power_spectrum_db']
        frequencies = spec_results['frequencies']
        
        power_peaks, _ = signal.find_peaks(
            power_spectrum_db,
            height=np.percentile(power_spectrum_db, 75),
            distance=len(frequencies) // 50,
            prominence=np.std(power_spectrum_db) * 0.5
        )
        
        peak_freqs_psd = frequencies[power_peaks]
        peak_powers_psd = power_spectrum_db[power_peaks]
        
        # Match power spectrum peaks to target frequencies
        tolerance_percent = 0.05  # 5% tolerance
        tolerance_hz = 50  # Minimum 50 Hz
        
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
                matched_target_freqs.append(peak_freqs_psd[closest_idx])
                matched_target_powers.append(peak_powers_psd[closest_idx])
            else:
                # If no peak found, use the power at the target frequency directly
                closest_freq_idx = np.argmin(np.abs(frequencies - target_freq_hz))
                matched_target_freqs.append(target_freq_hz)
                matched_target_powers.append(power_spectrum_db[closest_freq_idx])
                print(f"Warning: No peak found near {target_freq_hz/1000:.4f} kHz, using interpolated value")
        
        matched_target_freqs = np.array(matched_target_freqs)
        matched_target_powers = np.array(matched_target_powers)
        
        print(f"\nTarget frequencies and their power spectrum values:")
        print("Frequency(kHz) | Power(dB)")
        print("-" * 35)
        for i in range(len(matched_target_freqs)):
            print(f"{matched_target_freqs[i]/1000:13.4f} | {matched_target_powers[i]:9.2f}")
    
    results = {
        'amplitude': amp_results,
        'spectrum': spec_results,
        'sample_rate': sample_rate,
        'audio_data': audio_data,
        'time_frequency_array': time_freq_array,
        'target_frequencies': matched_target_freqs,
        'target_powers': matched_target_powers
    }
    
    if plot:
        print("\nGenerating plot...")
        spec_fig = plot_spectrum_analysis(
            spec_results, 
            target_freqs_hz=target_freqs_hz,
            matched_target_freqs=matched_target_freqs,
            matched_target_powers=matched_target_powers,
            title=f"Spectrum Analysis: {audio_path.name}"
        )
        results['spectrum_plot'] = spec_fig
    
    return results


def analyze_mp4_audio(mp4_path, plot=True, time_bin=0.1, overlap=0.5, filter_freq_range=None, 
                     target_freqs_khz=None):
    """
    Legacy function name for backwards compatibility.
    Calls analyze_audio_file internally.
    
    Parameters:
    - mp4_path (str or Path): Path to the audio file (WAV/MP4/M4A)
    - plot (bool): Whether to create plots
    - time_bin (float): Time bin size in seconds for spectrogram (default: 0.1 = 100 ms)
    - overlap (float): Overlap fraction for spectrum analysis (0-1)
    - filter_freq_range (tuple, optional): (low_freq, high_freq) in Hz to filter audio
    - target_freqs_khz (np.ndarray, optional): Array of target frequencies in kHz to analyze
    
    Returns:
    - dict: Dictionary containing all analysis results
    """
    return analyze_audio_file(mp4_path, plot=plot, time_bin=time_bin, overlap=overlap, 
                             filter_freq_range=filter_freq_range, target_freqs_khz=target_freqs_khz)


if __name__ == "__main__":
    # Example usage
    audio_file = r"Z:\Shared\Amichai\General\stim_cal\stimuli_power_analysis.wav"

    
    # Define target frequencies in kHz
    target_freqs_khz = np.array([0.7070, 0.7891, 0.8807, 1.0972, 1.2246, 1.3668, 
                                 1.7026, 1.9003, 2.1210])
    
    # Analyze audio file with custom filter range and target frequencies
    results = analyze_audio_file(
        audio_file, 
        filter_freq_range=(700, 2200),  # 700-2200 Hz
        time_bin=0.2,  # 200 ms time bins
        overlap=0.3,   # 30% overlap
        target_freqs_khz=target_freqs_khz
    )
    
    # Access results
    time_freq_array = results['time_frequency_array']
    print(f"\nAnalysis complete!")
    print(f"Time-frequency array shape: {time_freq_array.shape}")
    print(f"Columns: [time (s), dominant_frequency (Hz), power (dB)]")
    
    # Show spectrum plot
    if 'spectrum_plot' in results:
        results['spectrum_plot'].show()
