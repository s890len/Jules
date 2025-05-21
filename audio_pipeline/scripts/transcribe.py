import whisper
import torch
import os
import argparse
import time
import json

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("path", help="Path to the audio file.")
    parser.add_argument("--lang", default="ru", help="Language for transcription (default: ru).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling (default: 0.0).")
    parser.add_argument("--beam_size", type=int, help="Beam size for decoding (optional).")
    parser.add_argument("--condition", action="store_true", help="Enable condition_on_previous_text.")
    parser.add_argument("--prompt", default="", help="Initial prompt: topic, participants, terms...")
    parser.add_argument("--model_size", default="large-v3", help="Size of the Whisper model to use (e.g., tiny, base, small, medium, large-v2, large-v3).")
    parser.add_argument("--output_dir", default="../output", help="Directory to save output files (default: ../output, relative to script location).")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Input audio file '{args.path}' not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        print(f"Loading Whisper model '{args.model_size}'...")
        model = whisper.load_model(args.model_size, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return

    print("Транскрибирование...")
    start_time = time.time()

    transcribe_options = {
        "language": args.lang,
        "temperature": args.temperature,
        "condition_on_previous_text": args.condition,
        "initial_prompt": args.prompt
    }
    if args.beam_size is not None:
        transcribe_options["beam_size"] = args.beam_size
    
    try:
        result = model.transcribe(args.path, **transcribe_options)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    end_time = time.time()
    transcription_time = end_time - start_time
    print(f"Transcription completed in {transcription_time:.2f} seconds.")

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_output_dir = os.path.join(script_dir, args.output_dir)

    if not os.path.exists(resolved_output_dir):
        try:
            os.makedirs(resolved_output_dir)
            print(f"Created output directory: {resolved_output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{resolved_output_dir}': {e}")
            return
            
    input_filename_base = os.path.splitext(os.path.basename(args.path))[0]
    
    json_output_path = os.path.join(resolved_output_dir, f"{input_filename_base}_transcription.json")
    txt_output_path = os.path.join(resolved_output_dir, f"{input_filename_base}_transcription.txt")

    try:
        # Save segments to JSON
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(result["segments"], json_file, ensure_ascii=False, indent=2)
        print(f"Transcription segments saved to: {json_output_path}")

        # Save full text to TXT
        with open(txt_output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(result["text"])
        print(f"Full transcription text saved to: {txt_output_path}")
            
    except Exception as e:
        print(f"Error saving transcription files: {e}")

if __name__ == "__main__":
    main()
