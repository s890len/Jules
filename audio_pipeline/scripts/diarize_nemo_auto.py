import os
import json
import argparse
import pathlib
import librosa
import torch
# from nemo.collections.asr.models.msdd_models import NeuralDiarizer # Will use OfflineDiarWithASR
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel # MSDD is a type of EncDecDiarLabelModel
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from omegaconf import OmegaConf # For NeMo configurations

# Helper to convert NeMo's diarization output (if it's RTTM-like) to a more usable format
# NeMo's `diarize()` method or OfflineDiarWithASR usually outputs RTTM files or a list of lines in RTTM format.
# Each RTTM line is: type, file, chnl, tbeg, tdur, ortho, stype, name, conf, Slat
# We need tbeg, tdur, and name (speaker_label)

def rttm_to_segments(rttm_lines):
    segments = []
    for line in rttm_lines:
        parts = line.strip().split()
        if parts[0] == "SPEAKER":
            start_time = float(parts[3])
            duration = float(parts[4])
            raw_speaker_label = parts[7] # e.g., "SPEAKER_00", "SPEAKER_01" from NeMo
            
            # Convert "SPEAKER_XX" to "Speaker_X"
            try:
                speaker_id = int(raw_speaker_label.split('_')[-1])
                speaker_label = f"Speaker_{speaker_id}"
            except ValueError:
                speaker_label = raw_speaker_label # Fallback if parsing fails
                
            segments.append({
                "start": start_time,
                "end": start_time + duration,
                "speaker_label": speaker_label 
            })
    # Sort by start time, then by end time.
    segments.sort(key=lambda x: (x['start'], x['end']))
    return segments

def main():
    parser = argparse.ArgumentParser(description="Perform speaker diarization using NVIDIA NeMo and merge with Whisper JSON.")
    parser.add_argument("audio_filepath", type=str, help="Path to the input audio file (e.g., .wav).")
    parser.add_argument("whisper_json_filepath", type=str, help="Path to the JSON output from Whisper (containing segments with 'start' and 'end' times).")
    parser.add_argument("--max_speakers", type=int, default=5, help="Maximum number of speakers for clustering (default: 5).")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save the tagged JSON output (default: ../output/, relative to script).")
    # model_name is now for the MSDD model component, OfflineDiarWithASR is the framework
    parser.add_argument("--msdd_model_name", type=str, default="diar_msdd_telephonic_speakerdiarization_css_marblenet", help="Name of the NeMo MSDD model for diarization.")
    parser.add_argument("--vad_model_name", type=str, default="vad_multilingual_marblenet", help="Name of the NeMo VAD model.")
    parser.add_argument("--embedding_model_name", type=str, default="titanet_large", help="Name of the NeMo speaker embedding model.")


    args = parser.parse_args()

    # --- 1. Validate inputs ---
    if not os.path.exists(args.audio_filepath):
        print(f"Error: Audio file '{args.audio_filepath}' not found.")
        return
    if not os.path.exists(args.whisper_json_filepath):
        print(f"Error: Whisper JSON file '{args.whisper_json_filepath}' not found.")
        return

    script_dir = pathlib.Path(__file__).parent.resolve()
    resolved_output_dir = script_dir / args.output_dir
    
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {resolved_output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{resolved_output_dir}': {e}")
        return

    # --- 2. Initialize NeMo OfflineDiarWithASR ---
    print("Initializing NeMo OfflineDiarWithASR pipeline...")
    try:
        nemo_temp_dir = resolved_output_dir / "nemo_temp"
        nemo_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a base NeMo configuration for diarization
        # This would typically be loaded from a YAML file, but we can construct it
        diar_config = OmegaConf.create()

        # VAD parameters (using a pre-trained model)
        diar_config.num_workers = 1 # For local processing
        diar_config.batch_size = 32
        diar_config.diarizer = OmegaConf.create()
        diar_config.diarizer.manifest_filepath = None # Will be set per-file
        diar_config.diarizer.out_dir = str(nemo_temp_dir)
        diar_config.diarizer.oracle_vad = False # Use VAD model if True, or use ASR timestamps if available and configured
        diar_config.diarizer.asr_based_vad = True # Utilize ASR timestamps for VAD

        # Speaker Embeddings model
        diar_config.diarizer.speaker_embeddings = OmegaConf.create()
        diar_config.diarizer.speaker_embeddings.model_path = args.embedding_model_name
        diar_config.diarizer.speaker_embeddings.parameters = OmegaConf.create()
        diar_config.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
        diar_config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
        diar_config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1,1,1] # Example, check model defaults
        diar_config.diarizer.speaker_embeddings.parameters.save_embeddings = False


        # Clustering
        diar_config.diarizer.clustering = OmegaConf.create()
        diar_config.diarizer.clustering.parameters = OmegaConf.create()
        diar_config.diarizer.clustering.parameters.oracle_num_speakers = (args.max_speakers if args.max_speakers > 0 else -1) # -1 for auto
        # diar_config.diarizer.clustering.parameters.max_num_speakers = args.max_speakers if args.max_speakers > 0 else 10 # Fallback max if oracle not used
        diar_config.diarizer.clustering.parameters.enhanced_count_thresh = 0.8 # Example, check defaults


        # MSDD (Multi-scale Diarization Decoder) model
        diar_config.diarizer.msdd_model = OmegaConf.create()
        diar_config.diarizer.msdd_model.model_path = args.msdd_model_name
        diar_config.diarizer.msdd_model.parameters = OmegaConf.create()
        diar_config.diarizer.msdd_model.parameters.use_speaker_labels_from_rttm = False # Use clustering output
        diar_config.diarizer.msdd_model.parameters.num_speakers = args.max_speakers if args.max_speakers > 0 else None # Can be None for MSDD to infer
        diar_config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7] # Example
        diar_config.diarizer.msdd_model.parameters.seq_eval_mode = False


        # VAD model for segmentation (if not relying solely on ASR for VAD)
        diar_config.diarizer.vad = OmegaConf.create()
        diar_config.diarizer.vad.model_path = args.vad_model_name
        diar_config.diarizer.vad.parameters = OmegaConf.create()
        diar_config.diarizer.vad.parameters.smoothing = "median"
        diar_config.diarizer.vad.parameters.overlap = 0.5
        diar_config.diarizer.vad.parameters.onset = 0.3
        diar_config.diarizer.vad.parameters.offset = 0.3
        diar_config.diarizer.vad.parameters.pad_offset = -0.05


        # ASR Decoder Timestamps configuration (for using Whisper's output)
        diar_config.diarizer.asr = OmegaConf.create()
        diar_config.diarizer.asr.parameters = OmegaConf.create()
        diar_config.diarizer.asr.parameters.asr_based_vad = diar_config.diarizer.asr_based_vad
        diar_config.diarizer.asr.parameters.asr_model_path = "None" # Not loading a NeMo ASR model
        diar_config.diarizer.asr.parameters.word_ts_anchor_offset = 0.0
        diar_config.diarizer.asr.parameters.word_ts_anchor_type = "start"
        
        # Initialize the OfflineDiarWithASR pipeline
        # This class internally loads the specified models based on the config.
        diar_pipeline = OfflineDiarWithASR(cfg=diar_config)
        diar_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {diar_pipeline.device}")

    except Exception as e:
        print(f"Error initializing NeMo OfflineDiarWithASR pipeline: {e}")
        return

    # --- 3. Prepare Manifest for NeMo and Perform Diarization ---
    print("Performing speaker diarization using OfflineDiarWithASR...")
    try:
        # Create a manifest for NeMo
        # OfflineDiarWithASR expects a manifest that can point to ASR output (e.g., JSON from Whisper)
        # The ASR output format needs to be compatible with ASRDecoderTimeStamps.
        # Let's adapt Whisper JSON to the expected "word-level" timestamp format for NeMo if needed,
        # or see if segment-level is directly usable by configuring ASRDecoderTimeStamps.
        # For simplicity, we'll provide segment start/end to NeMo.
        # The manifest for OfflineDiarWithASR often includes 'asr_hyp_details_filepath'

        # Load Whisper segments to pass to NeMo
        with open(args.whisper_json_filepath, 'r', encoding='utf-8') as f:
            whisper_segments_for_nemo = json.load(f)

        # Convert Whisper segments to a NeMo-compatible ASR output format (list of dicts with 'start_time', 'end_time', 'label')
        # This is a simplified representation. True word-level might be more complex.
        # ASRDecoderTimeStamps can consume various formats. Let's try with segment-level.
        # The `asr_hyp_details_filepath` in the manifest should point to a JSON file where each line is a dict
        # like: {"start_time": float, "end_time": float, "word": str}
        # Or, for segments: {"start_time": float, "end_time": float, "label": "sentence" (or actual text)}

        asr_timestamps_json_path = nemo_temp_dir / "whisper_asr_timestamps.json"
        with open(asr_timestamps_json_path, 'w', encoding='utf-8') as f:
            for i, seg_item in enumerate(whisper_segments_for_nemo):
                # Using a generic label for segments for now.
                # Actual text might be useful if NeMo can leverage it.
                # ASRDecoderTimeStamps primarily uses start/end.
                # The 'word' key is typical for word-level timestamps.
                # We are providing segment-level timestamps. Let's use 'label'.
                # The `ASRDecoderTimeStamps` class will process this.
                # It looks for "start_time", "end_time", and "word" or "text_label".
                # If "word" is not found, it may default to a generic segment.
                # Let's use "word" and just put the segment index or a placeholder.
                f.write(json.dumps({
                    "start_time": seg_item['start'], 
                    "end_time": seg_item['end'], 
                    "word": f"segment_{i}" # Placeholder, as Whisper output is not word-level here
                }) + "\n")
        
        manifest_data = {
            'audio_filepath': args.audio_filepath,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': args.max_speakers if args.max_speakers > 0 else None,
            'rttm_filepath': None,
            'uem_filepath': None,
            'asr_hyp_details_filepath': str(asr_timestamps_json_path) # Path to Whisper segments in NeMo format
        }
        manifest_path = nemo_temp_dir / "input_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f)
            f.write('\n')
        
        # Update manifest path in config (as it's per-file)
        diar_pipeline.cfg.diarizer.manifest_filepath = str(manifest_path)
        
        # Run diarization
        # OfflineDiarWithASR.run_diarization() is the method.
        # It doesn't return RTTM lines directly but operates based on the config.
        # The results are typically stored in the `out_dir` specified in the config.
        diar_pipeline.run_diarization(diar_pipeline.cfg) # Pass the full config

        # Output RTTM file is expected in: nemo_temp_dir / "pred_rttms" / <audio_filename_base>.rttm
        audio_filename_base = pathlib.Path(args.audio_filepath).stem
        predicted_rttm_filepath = nemo_temp_dir / "pred_rttms" / f"{audio_filename_base}.rttm"

        if not predicted_rttm_filepath.exists():
            print(f"Error: Predicted RTTM file not found at {predicted_rttm_filepath}")
            # Additional debugging info:
            if not (nemo_temp_dir / "pred_rttms").exists():
                print(f"Directory {nemo_temp_dir / 'pred_rttms'} does not exist.")
            else:
                print(f"Contents of {nemo_temp_dir / 'pred_rttms'}: {list((nemo_temp_dir / 'pred_rttms').iterdir())}")
            if not (nemo_temp_dir / "speaker_outputs").exists():
                 print(f"Directory {nemo_temp_dir / 'speaker_outputs'} does not exist (expected for OfflineDiarWithASR).")
            else:
                print(f"Contents of {nemo_temp_dir / 'speaker_outputs'}: {list((nemo_temp_dir / 'speaker_outputs').iterdir())}")

            return

        with open(predicted_rttm_filepath, 'r') as f:
            rttm_lines = f.readlines()
        
        diar_segments = rttm_to_segments(rttm_lines) # Existing helper should still work
        if not diar_segments:
            print("Warning: No speaker segments found in RTTM output.")
        else:
        print(f"Diarization complete. Found {len(diar_segments)} speaker segments from RTTM.")

    except Exception as e:
        print(f"Error during OfflineDiarWithASR diarization: {e}")
        return

    # --- 4. Load Whisper Segments (already loaded for NeMo input preparation, reuse if needed or reload) ---
    # whisper_segs was loaded as whisper_segments_for_nemo. If it's modified, reload.
    # For merging, we need the original Whisper JSON structure.
    print(f"Re-loading original Whisper segments for merging: {args.whisper_json_filepath}")
    try:
        with open(args.whisper_json_filepath, 'r', encoding='utf-8') as f:
            original_whisper_segments = json.load(f) # This is a list of dicts
    except Exception as e:
        print(f"Error loading original Whisper JSON for merging: {e}")
        return

    # --- 5. Merge Whisper Segments with Diarization Results ---
    print("Merging Whisper segments with diarization results...")
    tagged_segments = []
    if not diar_segments:
        print("Warning: No diarization segments found. Output will not have speaker tags.")
        # Populate with Unknown speaker if diarization failed but we still want the text
        for seg in original_whisper_segments:
            tagged_segments.append({**seg, "speaker": "Unknown"})
    else:
        for seg in original_whisper_segments:
            # Ensure segment has 'start' and 'end' keys
            if 'start' not in seg or 'end' not in seg:
                print(f"Warning: Skipping Whisper segment due to missing 'start' or 'end' key: {seg}")
                continue

            seg_start = seg['start']
            seg_end = seg['end']
            
            # Find the speaker whose diarized segment has the maximum overlap with the Whisper segment
            best_speaker = "Unknown"
            max_overlap = 0.0

            for diar_seg in diar_segments: # diar_segments comes from rttm_to_segments
                overlap_start = max(seg_start, diar_seg['start'])
                overlap_end = min(seg_end, diar_seg['end'])
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = diar_seg['speaker_label'] 
            
            tagged_segments.append({**seg, "speaker": best_speaker})

    print("Merging complete.")

    # --- 6. Save Tagged JSON ---
    output_filename = f"{pathlib.Path(args.audio_filepath).stem}_tagged.json"
    output_filepath = resolved_output_dir / output_filename
    
    print(f"Saving tagged segments to: {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(tagged_segments, f, ensure_ascii=False, indent=2)
        print("Tagged JSON saved successfully.")
    except Exception as e:
        print(f"Error saving tagged JSON: {e}")

    # --- 7. Cleanup (optional, NeMo might have its own cleanup) ---
    # shutil.rmtree(nemo_temp_dir) # Example: if you want to remove the temp dir
    # print(f"Cleaned up temporary directory: {nemo_temp_dir}")

if __name__ == "__main__":
    main()
