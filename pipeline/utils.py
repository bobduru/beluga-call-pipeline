
from data_preprocessing.spectrogram.spectrogram_generator import SPECT_GENERATOR, HYDROPHONE_SENSITIVITY
from pipeline.pipeline import get_hydrophone_model
from pathlib import Path
import os

def cut_audio_file(filename, folder_path, start_s, end_s, output_dir=None, plot=False, comment=None):
    """
    Cut an audio file between start and end times and optionally plot spectrogram
    
    Args:
        audio_file_path (str): Path to input audio file
        start_s (float): Start time in seconds 
        end_s (float): End time in seconds
        output_dir (str): Directory to save output files (optional)
        plot (bool): Whether to generate and plot spectrogram
        
    Returns:
        tuple: (audio data array, sample rate)
    """

    audio_file_path = folder_path + filename
    # Read audio file using the same method as elsewhere in the codebase
    audio_data, sr, original_sr = SPECT_GENERATOR.load_audio(audio_file_path)
    
    # Convert times to samples
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    
    # Cut audio
    audio_cut = audio_data[start_sample:end_sample]

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    input_filename = Path(audio_file_path).stem
    output_file_prefix = f"{input_filename}_{start_s:.1f}s-{end_s:.1f}s"
    wav_output_path = Path(output_dir) / (output_file_prefix + ".wav")
    spec_output_path = Path(output_dir) / (output_file_prefix + ".png")
    comment_output_path = Path(output_dir) / (output_file_prefix + ".txt")
        
    if output_dir:
        
        # Save cut audio using librosa or soundfile
        import soundfile as sf
        sf.write(wav_output_path, audio_cut, sr)
    
    if plot:

        hydrophone_model = get_hydrophone_model(filename)
        hydrophone_sensitivity = HYDROPHONE_SENSITIVITY.get_sensitivity(hydrophone_model)
        # Generate spectrogram using your existing SPECT_GENERATOR
        power_spect = SPECT_GENERATOR.compute_mel_power_spect(audio_cut)
       
        dB_spect = SPECT_GENERATOR.power_to_db(power_spect, hydrophone_sensitivity=hydrophone_sensitivity)

        SPECT_GENERATOR.plot_spect(dB_spect, figsize=(6,2.6), save_path=spec_output_path)
        
    if comment:
        with open(comment_output_path, "w") as f:
            f.write(comment)
            
    return audio_cut, sr