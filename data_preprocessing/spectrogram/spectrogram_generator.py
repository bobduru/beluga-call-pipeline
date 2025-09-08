import json
import warnings
import os
import matplotlib.pyplot as plt
import librosa
import matplotlib.patches as patches
import matplotlib.ticker as ticker

import librosa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd

class HydrophoneSensitivityManager:
    """
    A class to manage hydrophone sensitivities based on their model numbers.
    Reads sensitivities from a JSON file and provides access methods.
    """
    
    def __init__(self, json_path=None, default_sensitivity=-170.0):
        """
        Initialize the sensitivity manager.
        
        Args:
            json_path (str, optional): Path to the JSON file containing sensitivity mappings.
            default_sensitivity (float, optional): Default sensitivity value in dB re 1V/uPa 
                to use when a model is not found. Defaults to -170.0.
        """
        self.sensitivities = {}
        self.default_sensitivity = default_sensitivity
        
        if json_path:
            # Get the directory of the current script file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_json_path = os.path.join(base_dir, json_path)
            self.load_from_json(absolute_json_path)
    
    def load_from_json(self, json_path):
        """
        Load hydrophone sensitivity mappings from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file.
        
        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
                
            # Convert string values to float
            self.sensitivities = {model: float(sensitivity) for model, sensitivity in data.items()}
        except FileNotFoundError:
            
            print(f"Sensitivity file {json_path} not found.")
        except json.JSONDecodeError:
            print(f"Could not parse {json_path} as valid JSON.")
    
    def get_sensitivity(self, model):
        """
        Get the sensitivity for a specific hydrophone model.
        
        Args:
            model (str): The hydrophone model number.
            
        Returns:
            float: The sensitivity value in dB re 1V/uPa.
        """
        
        if str(model) in self.sensitivities:
            return self.sensitivities[str(model)]
        else:
            warnings.warn(f"Hydrophone model '{model}' not found. Using default sensitivity: {self.default_sensitivity} dB re 1V/uPa.")
            return self.default_sensitivity
        

class SpectrogramGenerator:
    def __init__(
            self,
            n_mels=64,
            n_fft=1024,
            hop_length=128,
            fmin=200,
            sample_rate=192000, 
            power_to_db_ref=1e-12
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.power_to_db_ref = power_to_db_ref

    def compute_mel_power_spect(self, audio):
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio input must be a numpy array.")

        stft_complex = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_magnitude = np.abs(stft_complex)

        # Compute Mel spectrogram
        mel_power_spect = librosa.feature.melspectrogram(
            S=stft_magnitude**2,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
        )
        return mel_power_spect
    
    def apply_hydrophone_sensitivity_power(self, power_spect, hydrophone_sensitivity):
        # Convert system sensitivity from dB to linear scale
        sensitivity_linear = 10 ** (hydrophone_sensitivity / 20)

        # Apply calibration to convert to pressure units (μPa)
        calibrated_power_spect = power_spect / (sensitivity_linear ** 2)

        return calibrated_power_spect
    
    def power_to_db(self, spect, hydrophone_sensitivity=None):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        
        if hydrophone_sensitivity is None:
            return librosa.power_to_db(spect, ref=self.power_to_db_ref)
        else :
            calibrated_power_spect = self.apply_hydrophone_sensitivity_power(spect, hydrophone_sensitivity)
            spect_dB = librosa.power_to_db(calibrated_power_spect, ref=1.0)
            freq_resolution = self.sample_rate / self.n_fft

            spect_dB = spect_dB - 10 * np.log10(freq_resolution)
            
            return spect_dB
    
    def min_max_normalization(self, spect):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if spect.min() == spect.max():
            # self.plot_spect(spect)
            # print("here")
            return np.zeros_like(spect)
        return (spect - spect.min()) / (spect.max() - spect.min())
    
    def resize_spect(self, spect, img_shape):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if len(img_shape) != 2:
            raise ValueError("img_shape must be a tuple of (height, width).")
        
        # return cv2.resize(spect, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resize(spect, img_shape, mode="reflect", anti_aliasing=True)
    
    def get_mel_frequencies(self):
        return librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.fmin, fmax=self.sample_rate // 2)
    
    def compute_psd(self, audio, hydrophone_sensitivity=None):
        stft_complex = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_magnitude = np.abs(stft_complex)
        power_spect = stft_magnitude ** 2

        # Apply calibration if needed
        if hydrophone_sensitivity is not None:
            # print(f"Applying hydrophone sensitivity: {hydrophone_sensitivity} dB re 1V/uPa")
            power_spect = self.apply_hydrophone_sensitivity_power(power_spect, hydrophone_sensitivity)

        # Average power over time (axis=1)
        avg_power_per_freq = np.mean(power_spect, axis=1)

        # Frequency bin width
        freq_resolution = self.sample_rate / self.n_fft

        # Convert to PSD in dB re 1 μPa²/Hz
        psd_db = 10 * np.log10(avg_power_per_freq / freq_resolution)
        psd_db = psd_db - 10 * np.log10(freq_resolution)

        # Frequency axis
        freqs = np.linspace(0, self.sample_rate / 2, len(avg_power_per_freq))

        return freqs, psd_db
    



    
    def denoise_spect(self, spect, noise_profile, method="gating", gating_percentage_threshold=0.1):
        
        # Validate inputs
        if not isinstance(spect, np.ndarray) or not isinstance(noise_profile, np.ndarray):
            raise TypeError("Both spect and noise_profile must be numpy arrays.")
        
        # Ensure noise_profile has the correct shape (64, 1)
        if noise_profile.shape == (spect.shape[0],):
            noise_profile = noise_profile[:, np.newaxis]
        elif noise_profile.shape != (spect.shape[0], 1):
            raise ValueError(f"Noise profile must have shape {(spect.shape[0], 1)}. Got {noise_profile.shape}")

        if method not in ["gating", "subtraction"]:
            raise ValueError("Invalid method. Choose either 'gating' or 'subtraction'.")

        if method == "gating":

            threshold_per_freq_band = noise_profile * gating_percentage_threshold

            #TODO just to test rename if I keep like this
            # threshold_per_freq_band = gating_percentage_threshold
           

            spect = np.where(
                spect > (noise_profile + threshold_per_freq_band),
                spect,
                0
                # spect.min() if spect.min() >= 0 else 0
            )

        else:
            spect = np.maximum(spect - noise_profile, 0)
            
        return spect
    
    def compute_noise_profile(self, spect, method="median", quantile=None):

        if method == "median":
            return np.median(spect, axis=1, keepdims=True)

        elif method == "max":
            return np.max(spect, axis=1, keepdims=True)

        elif method == "quantile":
            if quantile is None or not (0 <= quantile <= 1):
                raise ValueError("For method='quantile', you must provide a quantile between 0 and 1.")
            return np.quantile(spect, q=quantile, axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'median', 'max', or 'quantile'.")

    
    
    def plot_spect(self, spect, title="Mel Spectrogram", figsize=(10, 4), rois_df=None, vmin=None, vmax=None, save_path=None, show=True, ax=None, tick_interval=None, colorbar_range=None, transparent_background=False):
        """
        Plot the Mel spectrogram and optionally save it to disk.

        Args:
            spect (np.ndarray): The spectrogram data.
            title (str, optional): Title for the plot. Defaults to "Mel Spectrogram".
            figsize (tuple, optional): Figure size. Defaults to (10, 4).
            rois_df (pd.DataFrame, optional): DataFrame containing ROIs to plot. Defaults to None.
            vmin (float, optional): Minimum value for the colorbar range. Defaults to None (autoscale).
            vmax (float, optional): Maximum value for the colorbar range. Defaults to None (autoscale).
            colorbar_range (tuple, optional): Tuple (vmin, vmax) for the colorbar range. Overrides vmin and vmax if provided. Defaults to None.
            save_path (str, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new axes. Defaults to None.
            tick_interval (float, optional): Interval between tick labels on the time axis. Defaults to None (autoscale).
            transparent_background (bool, optional): Whether to make the background transparent. Defaults to False.
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")

        if colorbar_range is not None:
            vmin, vmax = colorbar_range

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if transparent_background:
                fig.patch.set_alpha(0.0)  # Make figure background transparent
                ax.patch.set_alpha(0.0)   # Make axes background transparent
        else:
            fig = ax.figure
            if transparent_background:
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
        
        # Pass vmin and vmax to specshow
        img = librosa.display.specshow(
            spect,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            y_axis="mel",
            x_axis='time',
            ax=ax,
            vmin=vmin, # Add vmin
            vmax=vmax,  # Add vmax
            # cmap='winter'

        )
        # The colorbar will now respect the vmin/vmax set in specshow
        fig.colorbar(img, ax=ax, format='%+2.0f dB') # Added format for clarity, adjust as needed

        if spect.shape[0] != self.n_mels:
            print(f"Removing the time and freq axis because of resized spectrogram with shape {spect.shape}")
            ax.axis('off')

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Mel)")
        
        if tick_interval is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

        # Plot ROIs if provided
        if rois_df is not None:
            for _, roi in rois_df.iterrows():
                # color = 'cyan' if roi["source"] == "hf_calls" else 'red'
                color = 'cyan' 

                # Convert Timedelta to seconds if necessary
                x = roi["min_x_s"].total_seconds() if isinstance(roi["min_x_s"], pd.Timedelta) else roi["min_x_s"]
                w = (roi["max_x_s"].total_seconds() if isinstance(roi["max_x_s"], pd.Timedelta) else roi["max_x_s"]) - x

                y = roi["min_freq_hz"]
                h = roi["max_freq_hz"] - roi["min_freq_hz"]

                rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=transparent_background)
            print(f"Figure saved to {save_path}")

        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)


    def load_audio(self, audio_path):
        if not isinstance(audio_path, str):
            raise TypeError("Audio path must be a string.")
        try:
            # Get the original sample rate without loading the audio
            original_sr = librosa.get_samplerate(audio_path)
            
            # Load and resample the audio to the target sample rate
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Return the audio data, target sample rate, and original sample rate
            return audio, self.sample_rate, original_sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
    def plot_mel_band_intensity(self, spect, target_freq_hz, title=None, noise_profile=None, gating_percentage_threshold=0):
        """
        Plot the intensity over time of a specific frequency (in Hz) by converting it to the closest Mel band.
        Optionally overlays the noise profile value for that frequency band.
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if not (0 < target_freq_hz < self.sample_rate // 2):
            raise ValueError(f"Frequency must be between 0 and Nyquist frequency ({self.sample_rate // 2} Hz).")

        mel_frequencies = self.get_mel_frequencies()
        mel_band_idx = np.argmin(np.abs(mel_frequencies - target_freq_hz))
        closest_freq = mel_frequencies[mel_band_idx]

        band_intensity = spect[mel_band_idx, :]

        plt.figure(figsize=(5, 2.5))
        plt.plot(band_intensity, label=f"Mel Band {mel_band_idx} ~ {closest_freq:.1f} Hz")
        
        if noise_profile is not None:
            if not isinstance(noise_profile, np.ndarray):
                raise TypeError("Noise profile must be a numpy array.")
            if noise_profile.shape == (spect.shape[0],):
                noise_value = noise_profile[mel_band_idx]
            elif noise_profile.shape == (spect.shape[0], 1):
                noise_value = noise_profile[mel_band_idx, 0]
            else:
                raise ValueError(f"Noise profile must have shape ({spect.shape[0]},) or ({spect.shape[0]}, 1). Got {noise_profile.shape}")

            plt.axhline(y=noise_value + noise_value*gating_percentage_threshold, color='red', linestyle='--', label="Noise Level")

        plt.xlabel("Time Frame")
        plt.ylabel("Intensity (dB or Linear)")
        plt.title(title or f"Intensity Over Time for ~{closest_freq:.1f} Hz")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _filter_by_frequency_range(self, spect, min_freq_hz=None, max_freq_hz=None):
        """
        Helper method to filter a spectrogram by frequency range.
        
        Args:
            spect (np.ndarray): Input spectrogram
            min_freq_hz (float, optional): Minimum frequency in Hz. Defaults to None.
            max_freq_hz (float, optional): Maximum frequency in Hz. Defaults to None.
            
        Returns:
            tuple: (filtered_spect, freq_indices) - The filtered spectrogram and the indices used for filtering
        """
        # If no frequency bounds are specified, use the entire spectrogram
        if min_freq_hz is None and max_freq_hz is None:
            return spect, np.arange(spect.shape[0])
        
        # Select frequency bands based on min_freq_hz and max_freq_hz
        mel_frequencies = self.get_mel_frequencies()
        freq_mask = np.ones_like(mel_frequencies, dtype=bool)
        if min_freq_hz is not None:
            freq_mask &= (mel_frequencies >= min_freq_hz)
        if max_freq_hz is not None:
            freq_mask &= (mel_frequencies <= max_freq_hz)
            
        freq_indices = np.where(freq_mask)[0]
        
        if len(freq_indices) == 0:
            raise ValueError(f"No frequency bands found between {min_freq_hz}Hz and {max_freq_hz}Hz")
            
        # Filter spectrogram to the selected frequency bands
        filtered_spect = spect[freq_indices, :]
        
        return filtered_spect, freq_indices
    
    def compute_db_in_freq_range(self, spect, min_freq_hz=None, max_freq_hz=None):
        """
        Compute dB statistics (mean, max, min, median) between specified frequency bounds.
        If no bounds are specified, computes statistics for the entire spectrogram.
        
        Args:
            spect (np.ndarray): Input spectrogram in dB
            min_freq_hz (float, optional): Minimum frequency in Hz
            max_freq_hz (float, optional): Maximum frequency in Hz
            
        Returns:
            dict: Dictionary containing mean, max, min, and median dB values
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
            
        # Filter spectrogram by frequency range
        freq_band_data, _ = self._filter_by_frequency_range(spect, min_freq_hz, max_freq_hz)
        
        # Compute statistics
        stats = {
            'mean': round(float(np.mean(freq_band_data)), 1),
            'max': round(float(np.max(freq_band_data)), 1),
            'min': round(float(np.min(freq_band_data)), 1),
            'median': round(float(np.median(freq_band_data)), 1)
        }
        
        return stats
    
    def compute_peak_snr_with_time_compressed(self, signal_spect_db, noise_spect_db, min_freq_hz=None, max_freq_hz=None):
        """
        Since we are predicting calls in a fixed window, the signal can be across the whole window, or only in a part of it.
        So we compress the time axis taking the 95th percentile to make sure we get the signal intensity.

        SNR_dB = Signal_dB - Noise_dB

        Args:
            signal_spect_db (np.ndarray): Spectrogram with signal + noise (in dB units).
            noise_spect_db (np.ndarray): Spectrogram with only noise (reference, in dB units).
            min_freq_hz (float, optional): Minimum frequency in Hz for SNR calculation. Defaults to None (use all frequencies).
            max_freq_hz (float, optional): Maximum frequency in Hz for SNR calculation. Defaults to None (use all frequencies).

        Returns:
            snr_db_per_band (np.ndarray): SNR per mel band within the specified range (in dB, array of shape (n_selected_mels,)).
            snr_stats (dict): Dictionary containing mean, max, min, and median SNR (in dB) values over the selected bands.
        """
        if not isinstance(signal_spect_db, np.ndarray) or not isinstance(noise_spect_db, np.ndarray):
            raise TypeError("Both signal_spect_db and noise_spect_db must be numpy arrays.")
        
        if signal_spect_db.shape[0] != noise_spect_db.shape[0]:
            print(f"signal_spect_db.shape: {signal_spect_db.shape}")
            print(f"noise_spect_db.shape: {noise_spect_db.shape}")
            raise ValueError(f"Spectrograms must have the same number of frequency bands. Got {signal_spect_db.shape[0]} and {noise_spect_db.shape[0]} bands.")

        # Filter spectrograms by frequency range
        signal_spect_db_filtered, _ = self._filter_by_frequency_range(signal_spect_db, min_freq_hz, max_freq_hz)
        noise_spect_db_filtered, _ = self._filter_by_frequency_range(noise_spect_db, min_freq_hz, max_freq_hz)


        # noise_db_per_band = np.median(noise_spect_db_filtered, axis=1)
        noise_db_per_band = np.quantile(noise_spect_db_filtered, q=0.95, axis=1)    # Shape: (n_selected_mels,)

        # Calculate 95th percentile and each 10th percentile
        signal_db_per_band_95th = np.quantile(signal_spect_db_filtered, q=0.95, axis=1)  # Shape: (n_selected_mels,)
        snr_db_per_band_95th = signal_db_per_band_95th - noise_db_per_band

        # Calculate for each 10th percentile (0.1 to 0.9)
        snr_stats = {'95th': round(float(np.max(snr_db_per_band_95th)), 1)}
        
        for i in range(1, 10):
            q = i / 10
            signal_db_per_band = np.quantile(signal_spect_db_filtered, q=q, axis=1)
            snr_db_per_band = signal_db_per_band - noise_db_per_band
            
            percentile_name = f'{int(q*100)}th'
            snr_stats[percentile_name] = round(float(np.max(snr_db_per_band)), 1)




        # binary_mask = signal_spect_db_filtered > (noise_db_per_band[:, np.newaxis] + 5)
        # active_time_fraction = np.mean(np.any(binary_mask, axis=0))

        # snr_stats['active_time_fraction_1'] = round(active_time_fraction, 2)

        snr_db = signal_spect_db_filtered - noise_db_per_band[:, np.newaxis]
        # snr_db_per_time = np.quantile(snr_db, q=0.95, axis=0)
        snr_db_per_time = np.max(snr_db, axis=0)
        snr_time_mask = snr_db_per_time > 5
        active_time_fraction = np.mean(snr_time_mask)
        # Calculate average SNR across all time points
        avg_snr = np.mean(snr_db_per_time)
        snr_stats['avg_snr'] = round(avg_snr, 2)
        # Calculate average SNR only for time points where SNR > 5
        avg_snr_above_5 = np.mean(snr_db_per_time[snr_time_mask]) if np.any(snr_time_mask) else 0
        snr_stats['avg_snr_above_5'] = round(avg_snr_above_5, 2)


        snr_stats['active_time_fraction_1'] = active_time_fraction



        # # Calculate coverage ratio and estimated duration
        # percentiles = np.arange(10, 100, 10)  # 10, 20, 30, ..., 90
        # snr_above_thresh = []

        # for p in percentiles:
        #     q = p / 100
        #     signal_db_per_band = np.quantile(signal_spect_db_filtered, q=q, axis=1)
        #     snr_db_per_band = signal_db_per_band - noise_db_per_band
        #     max_snr = float(np.max(snr_db_per_band))
        #     snr_above_thresh.append(max_snr > 5)

        # coverage_ratio = np.mean(snr_above_thresh)  # Ratio of time with SNR > 5
        # # estimated_duration = round(coverage_ratio * 1, 3)

        
        # snr_stats['active_time_fraction_2'] = round(coverage_ratio, 2)

       


        return snr_stats

    
    def compute_snr_through_time(
        self,
        signal_spect_db,
        noise_spect_db,
        min_freq_hz=None,
        max_freq_hz=None,
        snr_threshold=5,
        time_threshold=0.05
    ):
        # Validate input types
        if not isinstance(signal_spect_db, np.ndarray) or not isinstance(noise_spect_db, np.ndarray):
            raise TypeError("Both signal_spect_db and noise_spect_db must be numpy arrays.")

        # Ensure both spectrograms have the same number of frequency bands
        if signal_spect_db.shape[0] != noise_spect_db.shape[0]:
            raise ValueError(
                f"Spectrograms must have the same number of frequency bands. "
                f"Got {signal_spect_db.shape[0]} and {noise_spect_db.shape[0]} bands."
            )

        # Apply frequency filtering
        signal_filtered, _ = self._filter_by_frequency_range(signal_spect_db, min_freq_hz, max_freq_hz)
        noise_filtered, _ = self._filter_by_frequency_range(noise_spect_db, min_freq_hz, max_freq_hz)

        # Estimate noise level using 95th percentile across time
        # To get a single value per frequency band that is representative of the noise level
        noise_db_per_band = np.quantile(noise_filtered, q=0.95, axis=1)[:, np.newaxis]    # Shape: (n_selected_mels, 1)

        # Compute SNR (in dB) over time
        snr_db = signal_filtered - noise_db_per_band    # Shape: (freq, time)

        #Collapse the frequency axis to get a single value per time point
        #To basically get where snr for each time point
        snr_db_per_time = np.max(snr_db, axis=0)                         # Shape: (time,)

        # Determine active time points where SNR exceeds threshold (where there is signal
        snr_time_mask = snr_db_per_time > snr_threshold

        # Calculate the fraction of time points where SNR exceeds threshold
        active_time_fraction = np.mean(snr_time_mask)

        avg_snr_signal = np.mean(snr_db_per_time[snr_time_mask]) if np.any(snr_time_mask) else 0
        
        # Compute statistics
        snr_stats = {
            'avg_window': round(np.mean(snr_db_per_time), 2),
            'signal_active_time': round(active_time_fraction, 2),
            'signal_present': int(active_time_fraction > time_threshold),
            'avg_signal': round(avg_snr_signal, 2)
        }

        return snr_stats

    


HYDROPHONE_SENSITIVITY = HydrophoneSensitivityManager("./hydrophones_sensitivity.json")

SPECT_GENERATOR = SpectrogramGenerator(
        n_fft=2048,
        hop_length=1024,
        n_mels=64,
        fmin=200,
        sample_rate=192000, 
    )
