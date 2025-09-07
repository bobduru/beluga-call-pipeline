

import matplotlib.pyplot as plt
import librosa
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from spectrogram.spectrogram_generator import SPECT_GENERATOR, HYDROPHONE_SENSITIVITY


def plot_pipeline_outputs_on_spects(
    audio,
    sr,
    results_df=None,
    audio_start_time=None,
    chunk_duration=60,
    num_chunks=3,
    labels=None,
    start_from_timestamp=None,
    start_from_sec=None,
    gating_percentage_threshold=0.08,
    noise_profile=None,
    hydrophone_sensitivity=None,
    show_snr_stats=True,
    show_call_predictions=True,
    vmin=None,
    vmax=None,
    spect_generator=SPECT_GENERATOR,
    figsize=(20, 4),
    start_time_ticks_0=False,
    show_title=True,
    show_probabilities=False
):
    """
    Plots the pipeline results on long spectrogram chunks of the processed audio. If we have access to ground truth labels, it can also plot them.

    Parameters:
    ----------
    audio (np.ndarray): The raw audio signal as a 1D numpy array.  
    sr (int): Sampling rate of the audio signal in Hz.  
    results_df (pd.DataFrame, optional): Pipeline results. If None, only spectrograms will be plotted without bounding boxes.
    audio_start_time (datetime or str): Start timestamp of the audio signal.  
    chunk_duration (int): Duration (in seconds) of each plotted chunk. Default is 60.  
    num_chunks (int): Number of chunks to plot. Default is 3.  
    labels (pd.DataFrame or None): Optional DataFrame of ground-truth labels to plot alongside results.  
    start_from_timestamp (datetime or str or None): Optional timestamp to start plotting from; must be after audio_start_time.
    start_time_ticks_0 (bool): If True, x-axis ticks start from 0. If False, use actual timestamps.
    """

    def plot_chunk(start_sec, end_sec, results_df, title, noise_profile=None, hydrophone_sensitivity=None,vmin=None,vmax=None, spect_generator=SPECT_GENERATOR):
        # Slice audio and create spectrogram
        audio_segment = audio[start_sec * sr:end_sec * sr]
        mel_spectrogram = spect_generator.compute_mel_power_spect(audio_segment)
        mel_spectrogram_db = spect_generator.power_to_db(mel_spectrogram, hydrophone_sensitivity)

        if noise_profile is not None:
            mel_spectrogram_db = spect_generator.denoise_spect(mel_spectrogram_db, noise_profile, gating_percentage_threshold=gating_percentage_threshold)

        # Compute time per frame to match extent
        n_frames = mel_spectrogram_db.shape[1]
        if start_time_ticks_0:
            time_axis = np.linspace(0, end_sec - start_sec, n_frames)
        else:
            time_axis = np.linspace(start_sec, end_sec, n_frames)

        fig, ax = plt.subplots(figsize=figsize)  # increase height a bit
        fig.patch.set_alpha(0.0)  # Make figure background transparent
        ax.patch.set_alpha(0.0)
        
        img = librosa.display.specshow(
            mel_spectrogram_db,
            sr=spect_generator.sample_rate,
            hop_length=spect_generator.hop_length,
            y_axis='mel',
            x_axis=None,
            ax=ax,
            x_coords=time_axis,
            vmin=vmin,
            vmax=vmax
        )
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        ax.set_xlim(start_sec, end_sec)
        if start_time_ticks_0:
            xticks = np.arange(0, int(end_sec - start_sec) + 1)
            ax.set_xlim(0, end_sec - start_sec)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x}" for x in xticks])
        else:
            xticks = np.arange(int(start_sec), int(end_sec) + 1)
            ax.set_xlim(start_sec, end_sec)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x}" for x in xticks])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Mel)")
        ax.set_title(title+"\n\n\n")

        #  Extend the y-axis to give space above the spectrogram
        ymin, ymax = ax.get_ylim()
       
        # Annotation Y-positions
        text_y_base = ymax + 75000
        line_spacing = 65000  # vertical spacing between lines



        if results_df is not None:
            subset = results_df[
                (pd.to_datetime(results_df["Timestamp"]) >= audio_start_time + pd.Timedelta(seconds=start_sec)) &
                (pd.to_datetime(results_df["Timestamp"]) < audio_start_time + pd.Timedelta(seconds=end_sec))
            ]

            for _, row in subset.iterrows():
                # if row["Call_Detection"] < 0.5:
                #     continue
                call_detected = row["Call_Detection"] > 0.5
                color = "white"
                start_time = (pd.to_datetime(row['Timestamp']) - audio_start_time).total_seconds()
                if start_time_ticks_0:
                    start_time -= start_sec
                duration = 1

                # Draw box (still inside the plot)
                y_start = 0
                y_height = 350000
                box = patches.Rectangle(
                    (start_time, y_start),
                    duration,
                    y_height,
                    linewidth=2,
                    edgecolor="grey",
                    facecolor='none',
                    alpha=0.6 if call_detected else 0.5,
                    linestyle='--',
                    clip_on=False
                )
                ax.add_patch(box)

                # Compose label string
                hf_calls = ["ECHO", "HFPC", "CC"]
                top_left_string = " ".join([call[0] for call in hf_calls if row.get(f"{call}", False)])
                if show_probabilities:
                    top_left_string = ", ".join([f"{call[0]}: {row.get(f'{call}_prob', 0):.2f}" for call in hf_calls if row.get(f"{call}", False)])

                top_left_string = top_left_string.replace("C", "B")
                # top_right_string = f"{row['snr_peak_hf_range_95th']}"
                top_right_string = f"{row['snr_hf_range_signal_active_time']:.2f}s, {row['snr_hf_range_avg_signal']:.0f}dB"
                top_right_color = "black" if row['snr_hf_range_signal_present'] else "grey"
                

                
                bottom_left_string = f"{"W" if row['Whistle'] else ''} "
                if show_probabilities:
                    bottom_left_string = f"{f"W : {row.get('Whistle_prob', 0):.2f}" if row['Whistle'] else ''} "

                # bottom_right_string = f"{row['snr_peak_w_range_95th']}"
                # bottom_right_string = f"{row['snr_w_range_avg_window']:.0f}dB, {row['snr_w_range_signal_active_time']:.2f}s, {row['snr_w_range_avg_signal']:.0f}dB"
                bottom_right_string = f" {row['snr_w_range_signal_active_time']:.2f}s, {row['snr_w_range_avg_signal']:.0f}dB"

                bottom_right_color = "black" if (row['snr_w_range_signal_present'] and row['Whistle']) else "grey"
                if not call_detected:
                    top_left_string = f"P: {row['Call_Detection']:.2f}"
                    top_left_string = f""

                if show_call_predictions:   
                    # Stack lines above each other with spacing
                    ax.text(start_time + 0.05, text_y_base, top_left_string, color='black',
                            fontsize=12, ha='left', va='bottom', clip_on=False)
                
                
                    ax.text(start_time + 0.05, text_y_base - line_spacing, bottom_left_string, color='black',
                            fontsize=12, ha='left', va='bottom', clip_on=False)
                
                if show_snr_stats:
                    ax.text(start_time + 0.95, text_y_base, top_right_string, color=top_right_color,
                            fontsize=12, ha='right', va='bottom', clip_on=False)
                    
                    ax.text(start_time + 0.95, text_y_base - line_spacing, bottom_right_string, color=bottom_right_color,
                            fontsize=12, ha='right', va='bottom', clip_on=False)
                    
                    # percentiles_string = ", ".join([f"{key}: {row[f'snr_peak_w_range_{key}']}" for key in ["30th", "50th",  "70th",  "90th",]])
                    # ax.text(start_time + 0.95, text_y_base - line_spacing*2, percentiles_string, color='white',
                    #         fontsize=10, ha='right', va='bottom', clip_on=False)
                    
                    # active_time_string = f"avg: {row['snr_w_range_avg_window']:.2f}, time: {row['snr_w_range_signal_active_time']:.2f}, avg>5: {row['snr_w_range_avg_signal']:.2f}\n"
                    # active_time_string += f"Present: {row['snr_w_range_signal_present']}"
                    # ax.text(start_time + 0.95, text_y_base - line_spacing*2.5, active_time_string, color='white',
                    #         fontsize=10, ha='right', va='bottom', clip_on=False)


            # Deduplicate legend
            handles, labels_ = ax.get_legend_handles_labels()
            by_label = dict(zip(labels_, handles))
            # ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # plt.tight_layout()
        plt.subplots_adjust(top=0.7)  # Make room for the suptitle
    
        plt.show()

    

    
    # Convert audio start time
    audio_start_time = pd.to_datetime(audio_start_time)


    # Handle start_from_sec and start_from_timestamp mutual exclusivity
    if start_from_sec is not None and start_from_timestamp is not None:
        raise ValueError("Cannot specify both start_from_sec and start_from_timestamp!")
        
    # Compute offset in seconds 
    start_offset_sec = 0
    # If start_from_sec is provided, use it directly as the offset
    if start_from_sec is not None:
        if start_from_sec < 0:
            raise ValueError("start_from_sec cannot be negative!")
        start_offset_sec = start_from_sec
    
    if start_from_timestamp is not None:
        start_from_timestamp = pd.to_datetime(start_from_timestamp)
        start_offset_sec = (start_from_timestamp - audio_start_time).total_seconds()

        if start_offset_sec < 0:
            raise ValueError("start_from_timestamp is before audio_start_time!")

    # Plot chunks
    for i in range(num_chunks):
        
        start_sec = int(start_offset_sec + i * chunk_duration)
        end_sec = int(start_offset_sec + (i + 1) * chunk_duration)

        # Compute actual datetime range for the title
        chunk_start_dt = (audio_start_time + pd.to_timedelta(start_sec, unit='s')).round('s')
        chunk_end_dt = (audio_start_time + pd.to_timedelta(end_sec, unit='s')).round('s')

        day_str = chunk_start_dt.strftime("%Y-%m-%d")
        start_str = chunk_start_dt.strftime("%H:%M:%S")
        end_str = chunk_end_dt.strftime("%H:%M:%S")
        title = f"from {start_str} to {end_str} on {day_str}" if show_title else ""

        plot_chunk(start_sec, end_sec, results_df, title, hydrophone_sensitivity=hydrophone_sensitivity, vmin=vmin, vmax=vmax, spect_generator=spect_generator)
        if noise_profile is not None:
            plot_chunk(start_sec, end_sec, results_df, f"{title} (denoised)", noise_profile=noise_profile, hydrophone_sensitivity=hydrophone_sensitivity, vmin=vmin, vmax=vmax, spect_generator=spect_generator)
        
        if labels is not None:
            plot_chunk(start_sec, end_sec, labels, "(Ground Truth Labels)")
        
        print("\n========================\n")


def check_pipeline_performance(
    long_audio,
    sample_rate,
    results_df,
    audio_start_time,
    hydrophone_sensitivity,
    n_samples=5,
    chunk_duration=8,
    num_chunks=1,
    vmin=None,
    vmax=None,
    method="uniform"
):
   
    if method == "random_calls":
        # ---- 1. Sample some actual detected calls ----
        if not results_df.empty:
            print(f"\n🎯 Showing {n_samples} detected calls:")
            sampled_calls = results_df.sample(min(n_samples, len(results_df)))
            for _, row in sampled_calls.iterrows():
                ts = row["Timestamp"]
                plot_pipeline_outputs_on_spects(
                    audio=long_audio,
                    sr=sample_rate,
                    results_df=results_df,
                    audio_start_time=audio_start_time,
                    labels=None,
                    start_from_timestamp=ts,
                    chunk_duration=chunk_duration,
                    num_chunks=num_chunks,
                    hydrophone_sensitivity=hydrophone_sensitivity,
                    vmax=vmax,
                    vmin=vmin,
                    show_call_predictions=True,
                    show_snr_stats=True
                )
        else:
            print("⚠️ No calls detected to sample from.")

    if method == "uniform":
        # ---- 3. Sample uniformly across the audio ----
        print(f"\n📊 Showing uniformly sampled audio segments:")

        duration = len(long_audio) / sample_rate

        # Calculate maximum start time to ensure the last chunk fits within the audio
        max_start_time = duration - (chunk_duration * num_chunks)
        if max_start_time > 0:
            # Generate uniform intervals
            uniform_offsets = np.linspace(0, max_start_time, n_samples)
            
            for offset in uniform_offsets:
                # Print progress through audio file
                progress_pct = (offset / duration) * 100
                print(f"Checking at {progress_pct:.1f}% through audio file ({offset:.1f}s / {duration:.1f}s)")
                ts = audio_start_time + timedelta(seconds=float(offset))
                
                plot_pipeline_outputs_on_spects(
                    audio=long_audio,
                    sr=sample_rate,
                    results_df=results_df,
                    audio_start_time=audio_start_time,
                    labels=None,
                    start_from_timestamp=ts.isoformat(),
                    chunk_duration=chunk_duration,
                    num_chunks=num_chunks,
                    hydrophone_sensitivity=hydrophone_sensitivity,
                    vmax=vmax,
                    vmin=vmin,
                    show_call_predictions=True,
                    show_snr_stats=True
                )