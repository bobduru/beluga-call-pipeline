import torch
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from models.quant_mobilenet import load_mobilenet_v3_quant_from_file
from spectrogram.spectrogram_generator import SPECT_GENERATOR, HYDROPHONE_SENSITIVITY

import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import numpy as np
import librosa
import os
import json
import re
import time
from datetime import datetime

from models.utils import get_best_device

import threading
import queue


import os
import platform
import sys

from pathlib import Path


def get_audio_start_time(audio_file):
    """
    Extract and parse the start time from an audio filename.
    
    Expected format: hydrophone_id.YYMMDDHHMMSS.wav
    
    Args:
        audio_file (str): The audio filename
        
    Returns:
        datetime: The parsed start time
        
    Raises:
        ValueError: If the file doesn't end with .wav or doesn't contain a valid date
    """
    # Check if file ends with .wav
    if not audio_file.endswith('.wav'):
        raise ValueError(f"File '{audio_file}' is not a WAV file")
    
    # Split the filename by dots and try to extract the date part
    parts = audio_file.split(".")
    
    # Check if we have enough parts (should have at least 3: hydrophone_id.date.wav)
    if len(parts) < 3:
        raise ValueError(f"Filename '{audio_file}' does not contain a date component")
    
    date_str = parts[1]
    
    # Try to parse the date string
    try:
        audio_start_time = datetime.strptime(date_str, "%y%m%d%H%M%S")
        return audio_start_time
    except ValueError:
        raise ValueError(f"Could not parse date from '{date_str}' in filename '{audio_file}'")
    
def get_hydrophone_model(audio_file):
    """
    Extract the hydrophone model or ID from the audio filename.
    
    Expected format: hydrophone_id.YYMMDDHHMMSS.wav
    
    Args:
        audio_file (str): The audio filename
        
    Returns:
        str: The hydrophone model or identifier
        
    Raises:
        ValueError: If the file doesn't end with .wav or doesn't contain a valid format
    """
    if not audio_file.endswith('.wav'):
        raise ValueError(f"File '{audio_file}' is not a WAV file")
    
    parts = audio_file.split(".")
    
    if len(parts) < 3:
        raise ValueError(f"Filename '{audio_file}' does not contain expected structure")
    
    hydrophone_model = parts[0]
    return hydrophone_model

def classify(
    spectrograms, 
    model, 
    device,
    label_columns=["ECHO", "HFPC", "CC", "Whistle"], 
    batch_size=32, 
    threshold=0.5
):
    """
    Classify spectrograms using a multi-label model in mini-batches.
    Handles both lists of spectrograms and a single spectrogram.

    Args:
        spectrograms (list or np.ndarray): List of spectrograms or a single spectrogram (H, W).
        model (torch.nn.Module): Trained multi-label classification model.
        label_columns (list of str): Names of the output labels.
        batch_size (int): Number of spectrograms to process per batch.
        threshold (float): Threshold for considering a label as present.

    Returns:
        list of tuples: Each tuple is (predicted_labels_dict, probabilities_dict)
    """
    all_results = []

    model.eval()

    # Handle a single spectrogram input
    if isinstance(spectrograms, np.ndarray) and spectrograms.ndim == 2:
        spectrograms = [spectrograms]  


    for i in range(0, len(spectrograms), batch_size):
        batch = spectrograms[i:i + batch_size]
        tensor_batch = torch.from_numpy(np.stack(batch)).float().unsqueeze(1)  # [B, 1, H, W]
        tensor_batch = tensor_batch.to(device)

        with torch.no_grad():
            logits = model(tensor_batch)
            probabilities = torch.sigmoid(logits)

        for probs in probabilities:
            probs = probs.cpu().numpy()
            prob_dict = {label: round(float(prob), 3) for label, prob in zip(label_columns, probs)}
            prob_dict["Call_Detection"] = round(float(max(probs)), 3)

            pred_dict = {
                label: (prob >= threshold) 
                for label, prob in prob_dict.items()
            }

            
            all_results.append((pred_dict, prob_dict))

    
    return all_results 


def compute_initial_noise_spects(audio, sample_rate, call_model,device, spect_generator=SPECT_GENERATOR, hydrophone_sensitivity=None, call_model_window_s=1.5, compute_n_noise_spects=5, debug=False, max_steps=100):
    """
    Computes and returns a list of spectrograms containing no beluga calls based on the classificiation results of our call model.

    Only spectrograms classified as not containing a whistle or high-frequency call
    are stored as 'noise'.

    Parameters:
    - audio: np.ndarray, raw audio signal
    - sample_rate: int, sample rate of the audio
    - call_model: classifier model to detect calls
    - call_model_window_s: float, time window in seconds
    - compute_n_noise_spects: int, number of noise spectrograms to collect
    - debug: bool, if True, prints debug info
    - max_steps: int, maximum number of steps to compute before giving up
    Returns:
    - List of dB spectrograms (length: n_noise_spects)
    """

    if compute_n_noise_spects==0:
        return []

    window_samples = int(call_model_window_s * sample_rate)
    total_samples = len(audio)
    step = window_samples

    noise_spects = []
    current_sample_idx = 0
    i = 0

    while current_sample_idx + window_samples <= total_samples and len(noise_spects) < compute_n_noise_spects:
        snippet = audio[current_sample_idx:current_sample_idx+window_samples]

        power_spect = spect_generator.compute_mel_power_spect(snippet)
        dB_spect = spect_generator.power_to_db(power_spect, hydrophone_sensitivity=hydrophone_sensitivity)
        # dB_spect = SPECT_GENERATOR.power_to_db(power_spect)
        normalized_spect = spect_generator.min_max_normalization(dB_spect)
        resized_spect = SPECT_GENERATOR.resize_spect(normalized_spect, img_shape=(244, 244))

        res = classify(resized_spect, call_model, device=device)
        prediction, probs = res[0]

        # if debug:
        #     print(f"Window {i // step + 1}: Prediction = {prediction}")
        #     SPECT_GENERATOR.plot_spect(dB_spect, title=f"Model prediction : {prediction} => NOISE")


        if probs["Call_Detection"] < 0.1:

            # We remove the first and last bits of the spectrogram to avoid edge effects of spectrogram generation
            noise_spect_more_confident = dB_spect[:, 10:-10]
            noise_spects.append(noise_spect_more_confident)
            # noise_spects.append(dB_spect)
            # SPECT_GENERATOR.plot_spect(dB_spect, title=f"Model prediction : {prediction} => NOISE")
            # SPECT_GENERATOR.plot_spect(noise_spect_more_confident, title=f"Model prediction : {prediction} => NOISE")

        current_sample_idx += step
        i += 1

        if i > max_steps:
            if debug:
                print(f"WARNING max steps reached ({max_steps}) while computing initial noise spectrograms, found {len(noise_spects)} noise spectrograms")
            break


    if debug:
        
        # print(f"Found the noise spects in the first {i/step} computed spectrograms")
        if len(noise_spects) > 0:
            concatenated_noise_spect = np.concatenate(noise_spects, axis=1)
            SPECT_GENERATOR.plot_spect(concatenated_noise_spect, title=f"{compute_n_noise_spects} concatenated initial noise spectrograms")

    return noise_spects


def calculate_window_db_snr(dB_spect, noise_profile_spect, whistles_freq_range, hf_calls_freq_range, spect_generator):
    db_window = {
        "w_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }

    db_noise = {
        "w_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }
    snr = {
        "w_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }
    # peak_snr = {
    #     "w_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
    #     "hf_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
    #     "full_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    # }

    return db_window, db_noise, snr


# Deprecated use run_pipeline_batched_long_spects instead
# Kept for as a reference
def run_pipeline(
    audio,
    audio_start_time,
    sample_rate,
    model,
    debug=False,
    hydrophone_sensitivity=None,
    call_model_window_s=1,
    seconds_to_process=None,
    batch_size=32,
    debug_at_s=None,
    n_noise_spects=5,
    whistles_freq_range=(200, 22000),
    hf_calls_freq_range=(40000, 90000),
    process_index=0,
    spect_generator=SPECT_GENERATOR,
    log_title = ""
):
    """
    Process an audio file using batch inference instead of single snippets.
    """
    
    # Duration calculations
    audio_duration = len(audio) / sample_rate
    end_time = audio_start_time + timedelta(seconds=audio_duration)

    if debug:
        print(f"Audio is {audio_duration:.2f} seconds long")

    if seconds_to_process is not None and seconds_to_process < audio_duration:
        end_time = audio_start_time + timedelta(seconds=seconds_to_process)
        print(f"Processing first {seconds_to_process} seconds.")

    step_duration = call_model_window_s
    current_time = audio_start_time

    results = []

    # Batch setup
    batch_audio_snippets = []
    batch_start_times = []
    batch_db_spects = []
    batch_model_spects = []


    #Here we keep an array and the spect itself to be able to only keep the latest n_noise_spects in memory to keep a fresh noise profile
    noise_spects = []
    noise_profile_spect = None

    spects_generation_time = 0
    spect_initial_gen_time = 0
    spect_db_time = 0
    spect_normalization_time = 0
    spect_resizing_time = 0

    inference_time = 0
    noise_calculation_time = 0


    if debug:
            print(f"Computing {n_noise_spects} spectrograms with only noise, to then be used to create a reliable noise profile")
      
    # noise_spects = compute_initial_noise_spects(
    #     audio, sample_rate, model, hydrophone_sensitivity=hydrophone_sensitivity, call_model_window_s=call_model_window_s, compute_n_noise_spects=n_noise_spects, debug=debug, max_steps=600)
    noise_spects = []

    if len(noise_spects) == 0:
        print("WARNING no noise spect found")
    else :
        noise_profile_spect = np.concatenate(noise_spects, axis=1)

     
    total_steps = int(((end_time - audio_start_time).total_seconds()) // step_duration) + 1

    if debug_at_s is not None:
        debug = False

    with tqdm(total=total_steps, desc=f"[Thread {process_index}] " + log_title, position=process_index, leave=True, mininterval=0.2, maxinterval=1.0) as pbar:
    # with tqdm(total=total_steps, desc="Running pipeline on audio file (batched)") as pbar:
        while current_time <= end_time:
            elapsed_time_seconds = (current_time - audio_start_time).total_seconds()
            if (debug_at_s is not None) and elapsed_time_seconds > debug_at_s:
                debug =True
            

            start_index = int(elapsed_time_seconds * sample_rate)
            end_index = int((elapsed_time_seconds + call_model_window_s) * sample_rate)

            # skip if end_index is greater than the audio length
            if not end_index > len(audio):
                snippet = audio[start_index:end_index]

                time_calc_start = time.time()
                power_spect = spect_generator.compute_mel_power_spect(snippet)
                spect_initial_gen_time += time.time() - time_calc_start


                time_calc_start = time.time()
                dB_spect = spect_generator.power_to_db(power_spect, hydrophone_sensitivity=hydrophone_sensitivity)
                spect_db_time += time.time() - time_calc_start

                batch_db_spects.append(dB_spect)  

                time_calc_start = time.time()
                model_normalized_spect = spect_generator.min_max_normalization(dB_spect)
                spect_normalization_time += time.time() - time_calc_start

                time_calc_start = time.time()
                model_resized_spect = spect_generator.resize_spect(model_normalized_spect, img_shape=(244, 244))
                spect_resizing_time += time.time() - time_calc_start

                batch_model_spects.append(model_resized_spect)   

                # Store the snippet and time
                batch_audio_snippets.append(snippet)
                batch_start_times.append(current_time)

            

            # If batch is full OR end of audio
            if len(batch_model_spects) >= batch_size or current_time + timedelta(seconds=step_duration) >= end_time:
                
                time_calc_start = time.time()
                # Classify the batch
                batch_predictions = classify(batch_model_spects, model, batch_size=batch_size)
                inference_time += time.time() - time_calc_start

                # Process predictions
                for (pred, probs), dB_spect, start_time_single in zip(batch_predictions, batch_db_spects, batch_start_times):
                    time_calc_start = time.time()

                    used_median_noise_profile = False
                    if len(noise_spects) == 0:
                        if debug:
                            print("Computing artificial noise profile")
                        noise_profile_spect = np.median(dB_spect, axis=1)[:, np.newaxis]
                        used_median_noise_profile = True

                    db_window, db_noise, snr, peak_snr = calculate_window_db_snr(dB_spect, noise_profile_spect, whistles_freq_range, hf_calls_freq_range, spect_generator)
                
                    noise_calculation_time += time.time() - time_calc_start

                    if debug:
                        print(f"Seconds since start: {(start_time_single - audio_start_time).total_seconds()}")
                        print(pred)
                        print(probs)
                        spect_generator.plot_spect(dB_spect, title=f"ECHO: {pred['ECHO']}, HFPC: {pred['HFPC']}, CC: {pred['CC']}, Whistle: {pred['Whistle']}, Call: {pred['Call_Detection']}", figsize=(5, 2.5), vmax=160, vmin=70)
                        
                    if probs["Call_Detection"] < 0.1 :
                        
                        # We remove the first and last bits of the spectrogram to avoid edge effects of spectrogram generation
                        new_noise_spect = dB_spect[:, 10:-10]

                        if len(noise_spects) == 0:
                            noise_spects.append(new_noise_spect)
                        else:
                            #Let's check if the new noise spect intensity is lower than the previous ones
                            if np.mean(new_noise_spect) < np.mean(noise_profile_spect):
                                #It is lower so let's rewrite the noise spects to make sure we keep the lowest most recent noise profile
                                if debug:
                                    print("Replacing noise profile because new one is lower")
                                noise_spects = [new_noise_spect]
                            else:
                                #It is higher we want to keep to simply append it to the list, in case it's a false negative and we actually have a bit of call in the snippet
                                if len(noise_spects) >= n_noise_spects:
                                    noise_spects.pop(0)
                                noise_spects.append(new_noise_spect)
                    

                        noise_profile_spect = np.concatenate(noise_spects, axis=1)
                        if debug:
                            print("Updating noise profile")
                            print("This snippet has no calls => UPDATE NOISE PROFILE")
                            SPECT_GENERATOR.plot_spect(noise_profile_spect, title="New concatenated noise spectrograms", vmax=180, vmin=70)
                        

                    noise_calculation_method = "precise_with_model" if not used_median_noise_profile else "default_median_value"

                    result = {
                        "Timestamp": start_time_single,
                        "seconds_since_file_start": (start_time_single - audio_start_time).total_seconds(),
                        "noise_calc_method": noise_calculation_method,
                        "Call_Detection": probs.get("Call_Detection", 0.0),
                    }

                    # Add call presence and probabilities
                    for label in ["ECHO", "HFPC", "CC", "Whistle"]:
                        result[f"{label}"] = pred.get(label, False)
                        result[f"{label}_prob"] = probs.get(label, 0.0)

                    # Add dB window and noise values
                    for band in ["w_range", "hf_range", "full_range"]:
                        for stat in db_window[band].keys():
                            result[f"db_window_{band}_{stat}"] = round(db_window[band][stat], 1)
                            result[f"db_noise_{band}_{stat}"] = round(db_noise[band][stat], 1)
                            # result[f"snr_{band}_{stat}"] = round(snr[band][stat], 1)

                    for band in ["w_range", "hf_range", "full_range"]:
                        for stat in snr[band].keys():
                            result[f"snr_{band}_{stat}"] = round(snr[band][stat], 2)

                    for band in ["w_range", "hf_range", "full_range"]:
                        for stat in peak_snr[band].keys():
                            # result[f"snr_peak_{band}_{stat}"] = round(peak_snr[band][stat], 1)
                            result[f"snr_peak_{band}_{stat}"] = peak_snr[band][stat]
                   
                    # Add the result to the results list
                    results.append(result)

                # Clear batch
                batch_audio_snippets = []
                batch_start_times = []
                batch_db_spects = []
                batch_model_spects = []
            # Move forward
            current_time += timedelta(seconds=step_duration)
            
                
            pbar.update(1)


    # columns = ["Calls", "Probs", "Timestamp", "seconds_since_file_start", "Window_DB", "Noise calculation method", "Noise_DB", "SNR", "Peak_SNR"]
    # results_df = pd.DataFrame(results, columns=columns)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Timestamp")
    spects_generation_time = spect_initial_gen_time + spect_db_time + spect_normalization_time + spect_resizing_time
    print("===============")
    print(f"Spects generation time: {spects_generation_time:.2f} seconds")
    print(f"  Initial generation: {spect_initial_gen_time:.2f} seconds")
    print(f"  dB conversion: {spect_db_time:.2f} seconds") 
    print(f"  Normalization: {spect_normalization_time:.2f} seconds")
    print(f"  Resizing: {spect_resizing_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Noise calculation time: {noise_calculation_time:.2f} seconds")
    print("===============")
    return results_df

def run_pipeline_batched_long_spects(
    audio,
    audio_start_time,
    sample_rate,
    model,
    device,
    debug=False,
    hydrophone_sensitivity=None,
    call_model_window_s=1,
    seconds_to_process=None,
    batch_size=32,
    debug_at_s=None,
    n_noise_spects=5,
    whistles_freq_range=(200, 22000),
    hf_calls_freq_range=(40000, 90000),
    process_index=0,
    spect_generator=SPECT_GENERATOR,
    skip_first_n_seconds=0,
    filter_absences=False,
    log_title=""
):
    """Process audio using batched spectrogram generation and inference.

    We found that generating spectrograms for longer audio segments at once and then individually cutting them into windows was more efficient than generating spectrograms for each window individually.

    We have developed a method to compute the noise profile and snr for each window using past windows with no calls detected by the model.
    This is still in development and can be improved.
    
    
    Args:
        audio: Audio samples array
        audio_start_time: Start datetime of recording
        sample_rate: Audio sample rate
        model: Classification model to use
        device: Device to run model on 
        debug: Enable debug output
        hydrophone_sensitivity: Hydrophone sensitivity in dB re 1V/uPa
        call_model_window_s: Window size in seconds for model input
        seconds_to_process: Optional max seconds to process
        batch_size: Batch size for model inference
        debug_at_s: Debug at specific timestamp
        n_noise_spects: Number of noise spectrograms to maintain
        whistles_freq_range: (min, max) frequency range for whistles in Hz
        hf_calls_freq_range: (min, max) frequency range for HF calls in Hz
        process_index: Thread index when running in parallel
        spect_generator: SpectrogramGenerator instance
        skip_first_n_seconds: Seconds to skip at start # This is due to some audio artifacts at the start of some files, of the hydrophone turning on
        filter_absences: To log or not the absences
        log_title: Title for logging
    """

    
    # Duration calculations
    audio_duration = len(audio) / sample_rate
    end_time = audio_start_time + timedelta(seconds=audio_duration)

    if debug:
        print(f"Audio is {audio_duration:.2f} seconds long")

    if seconds_to_process is not None and seconds_to_process < audio_duration:
        end_time = audio_start_time + timedelta(seconds=seconds_to_process)
        print(f"Processing first {seconds_to_process} seconds.")

    step_duration = call_model_window_s
    current_time = audio_start_time + timedelta(seconds=skip_first_n_seconds)

    # current_time = audio_start_time + timedelta(seconds=4)

    results = []

    # Timing variables
    spects_generation_time = 0
    spect_initial_gen_time = 0
    spect_db_time = 0
    spect_normalization_time = 0
    spect_resizing_time = 0
    inference_time = 0
    noise_calculation_time = 0

    # Noise profile setup
    noise_spects = []
    noise_profile_spect = None

    if debug:
        print(f"Computing {n_noise_spects} spectrograms with only noise, to then be used to create a reliable noise profile")
      
    noise_spects = compute_initial_noise_spects(
        audio, sample_rate, model,device=device, hydrophone_sensitivity=hydrophone_sensitivity, 
        call_model_window_s=call_model_window_s, compute_n_noise_spects=5, 
        debug=False, max_steps=500)

    # noise_spects = []
    
    if len(noise_spects) == 0:
        print("WARNING no noise spect found")
    else:
        noise_profile_spect = np.concatenate(noise_spects, axis=1)
     
    total_steps = int(((end_time - audio_start_time).total_seconds()) // step_duration) + 1

    if debug_at_s is not None:
        debug = False

    # Calculate samples per window and time bins per window for spectrogram slicing
    
    time_bins_per_window = int(call_model_window_s * sample_rate / spect_generator.hop_length)
    with tqdm(total=total_steps, desc=f"[Thread {process_index}] " + log_title, position=process_index, leave=True, mininterval=0.2, maxinterval=1.0) as pbar:
   
        # Process audio in chunks of batch_size * call_model_window_s
        while current_time <= end_time:
            elapsed_time_seconds = (current_time - audio_start_time).total_seconds()
            if (debug_at_s is not None) and elapsed_time_seconds > debug_at_s:
                debug = True
            
            # Calculate how many full windows we can process in this batch
            remaining_seconds = (end_time - current_time).total_seconds()
            windows_in_batch = min(batch_size, int(remaining_seconds / call_model_window_s) + 1)
            
            if windows_in_batch <= 0:
                break
                
            # Calculate audio indices for the entire batch segment
            batch_start_index = int(elapsed_time_seconds * sample_rate)
            batch_duration = windows_in_batch * call_model_window_s
            batch_end_index = min(batch_start_index + int(batch_duration * sample_rate), len(audio))
            
            # Skip if we're at the end of the audio
            if batch_start_index >= len(audio):
                break
                
            # Extract the batch audio segment
            batch_audio = audio[batch_start_index:batch_end_index]
            
            # Generate one long spectrogram for the entire batch
            time_calc_start = time.time()
            long_power_spect = spect_generator.compute_mel_power_spect(batch_audio)
            spect_initial_gen_time += time.time() - time_calc_start
            
            time_calc_start = time.time()
            long_db_spect = spect_generator.power_to_db(long_power_spect, hydrophone_sensitivity=hydrophone_sensitivity)
            # long_db_spect = spect_generator.power_to_db(long_power_spect)
            spect_db_time += time.time() - time_calc_start

            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=160, vmin=70)
            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=80, vmin=0)
            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=160, vmin=70)
            
            # Calculate actual number of windows we can extract from the spectrogram
            actual_windows = min(windows_in_batch, long_db_spect.shape[1] // time_bins_per_window)
            
            # Prepare batch data
            batch_db_spects = []
            batch_model_spects = []
            batch_start_times = []


            used_median_noise_profile = False
            if len(noise_spects) == 0:
                if debug:
                    print("Computing artificial noise profile")
                noise_profile_spect = np.median(long_db_spect, axis=1)[:, np.newaxis]
                used_median_noise_profile = True
            

            # Cut the long spectrogram into windows
            for i in range(actual_windows):
                start_bin = i * time_bins_per_window
                end_bin = (i + 1) * time_bins_per_window
                
                # Skip if we don't have enough bins left
                if end_bin > long_db_spect.shape[1]:
                    break
                    
                window_db_spect = long_db_spect[:, start_bin:end_bin]
                batch_db_spects.append(window_db_spect)
                
                # Normalize and resize for model input
                time_calc_start = time.time()
                model_normalized_spect = spect_generator.min_max_normalization(window_db_spect)
                spect_normalization_time += time.time() - time_calc_start

                # batch_model_spects.append(model_normalized_spect)
                
                time_calc_start = time.time()
                model_resized_spect = spect_generator.resize_spect(model_normalized_spect, img_shape=(244, 244))
                spect_resizing_time += time.time() - time_calc_start
                
                batch_model_spects.append(model_resized_spect)
                
                # Calculate timestamp for this window
                window_start_time = current_time + timedelta(seconds=i * call_model_window_s)
                batch_start_times.append(window_start_time)
            
            # Run inference on the batch
            if batch_model_spects:

                time_calc_start = time.time()
                # Classify the batch
                batch_predictions = classify(batch_model_spects, model, device=device, batch_size=batch_size)
                inference_time += time.time() - time_calc_start

                # Process predictions
                for (pred, probs), dB_spect, start_time_single in zip(batch_predictions, batch_db_spects, batch_start_times):
                    time_calc_start = time.time()


                    db_window, db_noise, snr = calculate_window_db_snr(dB_spect, noise_profile_spect, whistles_freq_range, hf_calls_freq_range, spect_generator)
                
                    noise_calculation_time += time.time() - time_calc_start

                    if debug:
                        print(f"Seconds since start: {(start_time_single - audio_start_time).total_seconds()}")
                        print(pred)
                        print(probs)
                        spect_generator.plot_spect(dB_spect, title=f"ECHO: {pred['ECHO']}, HFPC: {pred['HFPC']}, CC: {pred['CC']}, Whistle: {pred['Whistle']}, Call: {pred['Call_Detection']}", figsize=(5, 2.5), vmax=160, vmin=70)
                        
                    if probs["Call_Detection"] < 0.1 :
                        # We remove the first and last bits of the spectrogram to avoid edge effects of spectrogram generation
                        new_noise_spect = dB_spect[:, 10:-10]

                        if len(noise_spects) == 0:
                            noise_spects.append(new_noise_spect)
                        else:
                            #Let's check if the new noise spect intensity is lower than the previous ones
                            if np.mean(new_noise_spect) < np.mean(noise_profile_spect):
                                #It is lower so let's rewrite the noise spects to make sure we keep the lowest most recent noise profile
                                if debug:
                                    print("Replacing noise profile because new one is lower")
                                noise_spects = [new_noise_spect]
                            else:
                                #It is higher we want to keep to simply append it to the list, in case it's a false negative and we actually have a bit of call in the snippet
                                if len(noise_spects) >= n_noise_spects:
                                    noise_spects.pop(0)
                                noise_spects.append(new_noise_spect)
                    

                        noise_profile_spect = np.concatenate(noise_spects, axis=1)
                        if debug:
                            print("Updating noise profile")
                            print("This snippet has no calls => UPDATE NOISE PROFILE")
                            SPECT_GENERATOR.plot_spect(noise_profile_spect, title="New concatenated noise spectrograms", vmax=180, vmin=70)
                        

                    noise_calculation_method = "precise_with_model" if not used_median_noise_profile else "default_median_value"

                    result = {
                        "Timestamp": start_time_single,
                        "seconds_since_file_start": (start_time_single - audio_start_time).total_seconds(),
                        "noise_calc_method": noise_calculation_method,
                        "Call_Detection": probs.get("Call_Detection", 0.0),
                    }

                    # Add call presence and probabilities
                    for label in ["ECHO", "HFPC", "CC", "Whistle"]:
                        result[f"{label}"] = pred.get(label, False)
                        result[f"{label}_prob"] = probs.get(label, 0.0)

                    # Add dB window and noise values
                    for band in ["w_range", "hf_range"]:
                        for stat in db_window[band].keys():
                            result[f"db_window_{band}_{stat}"] = round(db_window[band][stat], 1)
                            result[f"db_noise_{band}_{stat}"] = round(db_noise[band][stat], 1)
                        
                        for stat in snr[band].keys():
                            result[f"snr_{band}_{stat}"] = round(snr[band][stat], 2)

                    
                    # Add the result to the results list
                    results.append(result)
            
            # Move forward by the number of windows we actually processed
            current_time += timedelta(seconds=batch_duration)
            pbar.update(actual_windows)

    # columns = ["Call", "Type", "Duration", "Timestamp", "seconds_since_file_start", "min_freq_hz", "max_freq_hz", "noise_calculation_method", "db_noise", "SNR_window", "SNR_peak", "probabilities", "detection_confidence"]
    # results_df = pd.DataFrame(results, columns=columns)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Timestamp")

    if filter_absences:
        results_df = results_df[results_df["Call_Detection"] > 0.5]
    
    spects_generation_time = spect_initial_gen_time + spect_db_time + spect_normalization_time + spect_resizing_time
    print("===============")
    print(f"Spects generation time: {spects_generation_time:.2f} seconds")
    print(f"  Initial generation: {spect_initial_gen_time:.2f} seconds")
    print(f"  dB conversion: {spect_db_time:.2f} seconds") 
    print(f"  Normalization: {spect_normalization_time:.2f} seconds")
    print(f"  Resizing: {spect_resizing_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Noise calculation time: {noise_calculation_time:.2f} seconds")
    print("===============")

    summary = create_results_summary(results_df, audio_duration)
    
    return results_df, summary



def create_results_summary(results_df, audio_length_seconds=None):
    """
    Create a summary of call detections and rates from the results DataFrame.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing call detection results
    audio_length_seconds : float, optional
        Length of the audio file in seconds. If provided, call rates will be calculated.
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    # Initialize summary dictionary
    summary = {}
    
    # Check if all noise calculations used precise_with_model
    summary['all_precise_noise_calc'] = bool((results_df['noise_calc_method'] == 'precise_with_model').all())
    
    # Calculate average noise and window levels
    summary['avg_noise_hf_range'] = float(results_df['db_noise_hf_range_mean'].mean())
    summary['avg_noise_w_range'] = float(results_df['db_noise_w_range_mean'].mean())
    summary['avg_window_hf_range'] = float(results_df['db_window_hf_range_mean'].mean())
    summary['avg_window_w_range'] = float(results_df['db_window_w_range_mean'].mean())
    
    # Count total number of calls detected
    total_calls = results_df['Call_Detection'].sum()
    summary['total_calls_detected'] = int(total_calls)
    
    # Count each type of call
    call_types = ['ECHO', 'HFPC', 'CC', 'Whistle']
    for call_type in call_types:
        call_count = results_df[call_type].sum()
        summary[f'{call_type.lower()}_count'] = int(call_count)
    
    # Calculate call rates if audio length is provided
    if audio_length_seconds is not None:
        # Calculate overall call rate (calls per minute)
        summary['overall_call_rate'] = float((total_calls / audio_length_seconds) * 60)
        
        # Calculate call rates for each type (calls per minute)
        for call_type in call_types:
            call_count = results_df[call_type].sum()
            summary[f'{call_type.lower()}_rate'] = float((call_count / audio_length_seconds) * 60)
    
    return summary




def clear_output_screen():
    try:
        # Detect if running in IPython (Jupyter or IPython shell)
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            from IPython.display import clear_output
            clear_output(wait=True)
        else:
            # Fallback to terminal clearing
            os.system('cls' if platform.system() == 'Windows' else 'clear')
    except Exception as e:
        print(f"Unable to clear output: {e}")

def run_multi_file_pipeline_with_prefetching(
    raw_wav_files,
    raw_wav_files_dir,
    output_dir,
    # model,
    spect_generator,
    process_index,
    debug=False,
    seconds_to_process_if_debugging=100,
    filter_absences=False
   
):
    """
    Run the pipeline on multiple WAV files and save results as JSON, with audio prefetching.
    """


    seconds_to_process = None
    if debug:
        output_dir = os.path.join(output_dir, "debug")
        seconds_to_process = seconds_to_process_if_debugging

    device = get_best_device()
    # model = load_mobilenet_v3_quant_from_file("./model_weights/mobile_net_8_layers.pt", n_layers=8)
    model = load_mobilenet_v3_quant_from_file("C:/Users/Admin/Desktop/Emmanuel/save-the-belugas/pipeline/model_weights/mobile_net_8_layers.pt", n_layers=8)


    model.to(device)
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a queue for prefetching
    audio_queue = queue.Queue(maxsize=1)  # Preload only 1 file ahead to avoid memory issues

    # Start preloading in the background
    loader_thread = threading.Thread(target=preload_audio, args=(raw_wav_files, raw_wav_files_dir, audio_queue, process_index))
    loader_thread.start()

    processed_count = 0
    try:
        while True:
        
            try:
                item = audio_queue.get(timeout=10)  # Prevent indefinite blocking
            except queue.Empty:
                # print("Queue is empty, waiting for audio...")
                continue
            if item is None:
                break  # No more files

            filename, audio, sample_rate, original_sr = item
            if audio is None:
                print(f"Skipping {filename} due to loading error")
                continue

            # print(f"Processing file {processed_count+1}/{len(raw_wav_files)}: {filename}")

            try:
                # Get start time
                audio_start_time = get_audio_start_time(filename)
                hydrophone_model = get_hydrophone_model(filename)
                hydrophone_sensitivity = HYDROPHONE_SENSITIVITY.get_sensitivity(hydrophone_model)

                # Time the processing
                start_processing_time = time.time()

                # Calculate audio duration
                audio_duration_seconds = len(audio) / sample_rate
                audio_end_time = audio_start_time + timedelta(seconds=audio_duration_seconds)

        

                results_df, results_summary = run_pipeline_batched_long_spects(
                    audio,
                    audio_start_time,
                    sample_rate,
                    model,
                    device,
                    spect_generator=spect_generator,
                    debug=debug,
                    batch_size=128,
                    call_model_window_s=1,
                    hydrophone_sensitivity=hydrophone_sensitivity,
                    skip_first_n_seconds=4,
                    filter_absences=filter_absences,
                    process_index=process_index,
                    log_title= f"Processing {filename} : {processed_count+1}/{len(raw_wav_files)}"
                )

                # Calculate processing time
                processing_time = time.time() - start_processing_time
        
                # Save results DataFrame to CSV 
                csv_filename = os.path.splitext(filename)[0] + ".csv"
                # Create output data structure
                metadata = {
                        "raw_wav_filename": filename,
                        "start_time": audio_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": audio_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": round(audio_duration_seconds),
                        "decimated_sample_rate": sample_rate,
                        "original_sample_rate": original_sr,
                        "processing_time": f"{processing_time:.2f}",
                        "hydrophone_sensitivity": hydrophone_sensitivity,
                        "summary": results_summary,
                        "results_file": csv_filename
                }

                # Save metadata to JSON
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                csv_path = os.path.join(output_dir, csv_filename)
                results_df.to_csv(csv_path, index=False)



                # print(f"Results saved to {output_path}")
                processed_count += 1
                # if(process_index == 0):
                clear_output_screen() 

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    except KeyboardInterrupt:
        print("Interrupted — exiting processing loop early.")

    finally:
        loader_thread.join()
        print(f"Completed processing {processed_count}/{len(raw_wav_files)} files")
    
    return processed_count



def find_wav_files_in_dir(directory, log=False):
    """
    Find all WAV files in a directory, sort them, and optionally log info.
    
    Args:
        directory (str): Path to the directory to search
        log (bool): Whether to print logging information
        
    Returns:
        list: Sorted list of WAV filenames
    """
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    wav_files = [filename for filename in filenames if filename.endswith('.wav')]
    
    if log:
        num_files = len(wav_files)
        print(f"Found {num_files} WAV files in {directory}")
        
        # Print the first few files (up to 5)
        num_to_show = min(5, num_files)
        if num_to_show > 0:
            print(f"First {num_to_show} files:")
            for i in range(num_to_show):
                print(f"  {i+1}. {wav_files[i]}")
    
    return wav_files



def preload_audio(file_list, base_dir, queue, process_index=0):
    # Create a dedicated tqdm line for loading status, right below the main progress bar
    # load_bar = tqdm(
    #     total=0,
    #     position=process_index * 2,
    #     bar_format='{desc}',
    #     leave=True
    # )

    try:
        for filename in file_list:

            # load_bar.set_description_str(f"[Proc {process_index}] Loading {filename}")
            full_path = os.path.join(base_dir, filename)

            try:
                audio, sample_rate, original_sr = SPECT_GENERATOR.load_audio(full_path)
                queue.put((filename, audio, sample_rate, original_sr))
                # load_bar.set_description_str(f"[Proc {process_index}] Loaded {filename}")
            except Exception as e:
                # load_bar.set_description_str(f"[Proc {process_index}] Failed {filename}")
                queue.put((filename, None, None, None))
    finally:
        queue.put(None)
        # load_bar.set_description_str(f"[Proc {process_index}] Preload done.")


def filter_already_processed_files(wav_files_list, output_dir):
    """
    Remove WAV files that already have a matching JSON in output_dir.

    Parameters
    ----------
    wav_files_list : list[str]
        WAV file paths you plan to process.
    output_dir     : str or Path
        Folder containing previously generated *.json outputs.

    Returns
    -------
    list[str]
        WAV files still needing processing (as plain strings).
    """
    output_dir = Path(output_dir)

    # Stems (filenames without extensions) for existing JSON files
    processed_stems = {p.stem for p in output_dir.glob("*.json")}

    original_len = len(wav_files_list)

    # Keep only WAVs whose stem is NOT in processed_stems
    remaining_wavs = [
        str(w) for w in wav_files_list
        if Path(w).stem not in processed_stems
    ]

    processed_len = len(remaining_wavs)


    return remaining_wavs


def merge_multi_file_pipeline_outputs(directory):
    # Get all CSV files in directory
    csv_files = list(Path(directory).glob('*.csv'))
    
    # Create empty list to store dataframes
    dfs = []
    
    # Read each CSV and add filename column
    for file in csv_files:
        # Skip files that don't match the pattern number.number.csv
        if not all(part.replace('.', '').isdigit() for part in file.stem.split('.')):
            continue
        if "_merged_results.csv" in file.stem:
            continue
        if "_continuous_segments.csv" in file.stem:
            continue


        df = pd.read_csv(file)
        df['filename'] = file.stem
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(directory + "/_merged_results.csv", index=False)
    
    return merged_df


import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from pathlib import Path


def find_continuous_segments(directory_path, margin_seconds= 3):
    """
    Find continuous segments of files processed by the pipeline.
    
    Args:
        directory_path: Path to directory containing JSON files
        margin_seconds: Maximum allowed gap between files (default: 3 seconds)
    
    Returns:
        pandas DataFrame with columns:
        - start_file: First file in the segment
        - end_file: Last file in the segment  
        - start_time: Start time of first file
        - end_time: End time of last file
        - file_count: Number of files in the segment
        - gap_from_previous: Time gap from previous segment (NaN for first segment)
    """
    
    output_csv_path = os.path.join(directory_path, "_continuous_segments.csv")

    # Get all JSON files in the directory
    json_files = []
    for file in os.listdir(directory_path):
        if file.endswith('.json'):
            file_path = os.path.join(directory_path, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract start_time and end_time
                    start_time = datetime.strptime(data['start_time'], '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(data['end_time'], '%Y-%m-%d %H:%M:%S')
                    json_files.append({
                        'filename': file,
                        'start_time': start_time,
                        'end_time': end_time,
                        'data': data
                    })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading {file}: {e}")
                continue
    
    # Sort files by start_time
    json_files.sort(key=lambda x: x['start_time'])
    
    if not json_files:
        print("No JSON files found in directory")
        return pd.DataFrame()
    
    # Find continuous segments
    segments = []
    current_segment = [json_files[0]]
    
    for i in range(1, len(json_files)):
        current_file = json_files[i]
        previous_file = json_files[i-1]
        
        # Calculate time difference between previous end and current start
        time_diff = (current_file['start_time'] - previous_file['end_time']).total_seconds()
        
        # Check if files are continuous (within margin)
        if time_diff <= margin_seconds:
            # Add to current segment
            current_segment.append(current_file)
        else:
            # End current segment and start new one
            if current_segment:
                segments.append(current_segment)
            current_segment = [current_file]
    
    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Create DataFrame
    rows = []
    for i, segment in enumerate(segments):
        if len(segment) > 0:
            # Calculate gap from previous segment
            if i == 0:
                gap_from_previous = None  # First segment has no previous
            else:
                # Gap from end of previous segment to start of current segment
                gap_seconds = (segment[0]['start_time'] - segments[i-1][-1]['end_time']).total_seconds()
                gap_from_previous = gap_seconds
            
            rows.append({
                'start_file': segment[0]['filename'],
                'end_file': segment[-1]['filename'],
                'start_time': segment[0]['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': segment[-1]['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration_hours': round((segment[-1]['end_time'] - segment[0]['start_time']).total_seconds() / 3600, 2),
                'file_count': len(segment),
                'gap_from_previous_seconds': gap_from_previous
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV if output path provided
    if output_csv_path:
        if not df.empty:
            df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")
            print(f"Found {len(df)} continuous segments")
            
            # Display summary
            print("\nSummary:")
            print(f"Total segments: {len(df)}")
            print(f"Total files: {df['file_count'].sum()}")
            
            # Show gaps larger than margin
            gaps = df['gap_from_previous_seconds'].dropna()
            if not gaps.empty:
                print(f"Gaps between segments: {gaps.tolist()}")
        else:
            print("No segments found")
    
    return df

