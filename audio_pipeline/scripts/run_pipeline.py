import os
import subprocess
import argparse
from pathlib import Path
import sys

# Determine the directory where this script is located, to find sibling scripts.
# Assumes all scripts (run_pipeline.py, clean_audio.py, etc.) are in the same directory.
SCRIPTS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPTS_DIR.parent # Assuming scripts/ is under audio_pipeline/

def run_script(command_args, step_name):
    """
    Runs a script using subprocess and handles its output and errors.
    Returns True on success, False on failure.
    """
    print(f"\n--- Running Step: {step_name} ---")
    print(f"Executing command: {' '.join(command_args)}")
    try:
        process = subprocess.run(command_args, check=True, text=True, capture_output=True, cwd=PROJECT_ROOT)
        print(f"--- {step_name} STDOUT ---")
        if process.stdout:
            print(process.stdout)
        print(f"--- {step_name} STDERR ---")
        if process.stderr:
            print(process.stderr) # Even if check=True, stderr might have warnings
        print(f"--- {step_name} completed successfully. ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Error during {step_name}: Command failed with exit code {e.returncode} !!!")
        print(f"--- {step_name} STDOUT (on error) ---")
        if e.stdout:
            print(e.stdout)
        print(f"--- {step_name} STDERR (on error) ---")
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"!!! Error during {step_name}: Script or Python interpreter not found. Command: {' '.join(command_args)} !!!")
        print("Please ensure Python is installed and the script paths are correct.")
        return False
    except Exception as e:
        print(f"!!! An unexpected error occurred during {step_name}: {e} !!!")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full audio processing pipeline.")
    parser.add_argument("filepath", help="Path to the input audio file.")
    parser.add_argument("--skip_clean", action="store_true", help="Skip the audio cleaning step.")
    parser.add_argument("--skip_summary", action="store_true", help="Skip the summarization step.")
    parser.add_argument("--transcribe_model_size", default="large-v3", help="Whisper model size for transcribe.py.")
    parser.add_argument("--transcribe_lang", default="ru", help="Language for transcription.")
    parser.add_argument("--transcribe_prompt", default="", help="Initial prompt for Whisper.")
    parser.add_argument("--diarize_max_speakers", default="5", type=str, help="Max speakers for diarize_nemo_auto.py.")
    parser.add_argument("--summarize_model", default="gemma:27b", help="Ollama model for summarize_json.py.")
    parser.add_argument("--summarize_max_chars", default="15000", type=str, help="Max characters for summary input.")
    parser.add_argument("--output_dir_base", default="../output", help="Base directory for all outputs, relative to project root (audio_pipeline/).")

    args = parser.parse_args()

    # --- Path Setup ---
    try:
        input_audio_filepath = Path(args.filepath).resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: Input audio file not found at '{args.filepath}'")
        sys.exit(1)

    # Base output directory, resolved relative to the project root
    # If output_dir_base is "../output", and project_root is /path/to/audio_pipeline,
    # then resolved_output_dir_base will be /path/to/output
    resolved_output_dir_base = (PROJECT_ROOT / args.output_dir_base).resolve()
    
    try:
        resolved_output_dir_base.mkdir(parents=True, exist_ok=True)
        print(f"Using base output directory: {resolved_output_dir_base}")
    except OSError as e:
        print(f"Error creating base output directory '{resolved_output_dir_base}': {e}")
        sys.exit(1)

    current_audio_file_to_process = input_audio_filepath
    current_audio_stem = input_audio_filepath.stem

    # --- Step 1: Audio Cleaning (Optional) ---
    if not args.skip_clean:
        cleaned_audio_filename = f"{input_audio_filepath.stem}_cleaned{input_audio_filepath.suffix}"
        cleaned_audio_filepath = resolved_output_dir_base / cleaned_audio_filename
        
        cmd_clean = [
            "python", str(SCRIPTS_DIR / "clean_audio.py"),
            str(current_audio_file_to_process),
            "--output_path", str(cleaned_audio_filepath)
        ]
        if not run_script(cmd_clean, "Audio Cleaning"):
            sys.exit(1)
        current_audio_file_to_process = cleaned_audio_filepath
        current_audio_stem = cleaned_audio_filepath.stem # Update stem for subsequent filenames
    else:
        print("Skipping audio cleaning step.")

    # --- Step 2: Transcription (Whisper) ---
    # transcribe.py outputs <input_stem>.json and <input_stem>.txt into its --output_dir
    # So, whisper_json_path will be resolved_output_dir_base / <current_audio_stem>.json
    whisper_json_path = resolved_output_dir_base / f"{current_audio_stem}.json" 
    # (transcribe.py also creates a .txt file, which we don't directly use here but it's generated)

    cmd_transcribe = [
        "python", str(SCRIPTS_DIR / "transcribe.py"),
        str(current_audio_file_to_process),
        "--model_size", args.transcribe_model_size,
        "--lang", args.transcribe_lang,
        "--prompt", args.transcribe_prompt,
        "--output_dir", str(resolved_output_dir_base) # transcribe.py will form filenames based on input stem
    ]
    if not run_script(cmd_transcribe, "Transcription"):
        sys.exit(1)
    
    if not whisper_json_path.exists():
        print(f"Error: Expected Whisper JSON output not found at {whisper_json_path}")
        sys.exit(1)

    # --- Step 3: Diarization (NeMo) ---
    # diarize_nemo_auto.py outputs <input_audio_stem>_tagged.json into its --output_dir
    diarized_json_path = resolved_output_dir_base / f"{current_audio_stem}_tagged.json"

    cmd_diarize = [
        "python", str(SCRIPTS_DIR / "diarize_nemo_auto.py"),
        str(current_audio_file_to_process), # Audio input for diarization
        str(whisper_json_path),             # Whisper JSON input
        "--max_speakers", args.diarize_max_speakers,
        "--output_dir", str(resolved_output_dir_base)
    ]
    if not run_script(cmd_diarize, "Diarization"):
        sys.exit(1)

    if not diarized_json_path.exists():
        print(f"Error: Expected Diarization JSON output not found at {diarized_json_path}")
        sys.exit(1)

    # --- Step 4: Convert to TXT/MD ---
    # convert_tagged_json_to_txt_md.py outputs <input_json_stem>.txt and <input_json_stem>.md
    # So, if input is current_audio_stem_tagged.json, output is current_audio_stem_tagged.txt/md
    
    cmd_convert = [
        "python", str(SCRIPTS_DIR / "convert_tagged_json_to_txt_md.py"),
        str(diarized_json_path),
        "--output_dir", str(resolved_output_dir_base)
    ]
    if not run_script(cmd_convert, "Convert to TXT/MD"):
        sys.exit(1)

    # --- Step 5: Summarization (Ollama) (Optional) ---
    if not args.skip_summary:
        # summarize_json.py outputs <input_json_stem>_summary.txt and <input_json_stem>_summary.md
        # Input is diarized_json_path (e.g., ..._tagged.json)
        cmd_summarize = [
            "python", str(SCRIPTS_DIR / "summarize_json.py"),
            str(diarized_json_path),
            "--model", args.summarize_model,
            "--max_chars", args.summarize_max_chars,
            "--output_dir", str(resolved_output_dir_base)
        ]
        if args.transcribe_prompt: # If user provided a prompt for transcription via run_pipeline, use it for summary's system prompt
            cmd_summarize.extend(["--system_prompt", args.transcribe_prompt])
        # If args.transcribe_prompt is empty, summarize_json.py will use its own default system prompt.

        if not run_script(cmd_summarize, "Summarization"):
            # Don't exit pipeline on summarization failure, it's optional/final step
            print("Summarization step failed, but previous steps were successful.")
            pass 
    else:
        print("Skipping summarization step.")

    print("\n--- Pipeline finished. ---")

if __name__ == "__main__":
    main()
