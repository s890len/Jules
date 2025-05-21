import subprocess
import os
import argparse

def clean_audio(input_path, output_path):
    """
    Cleans the audio file using ffmpeg.
    Converts to mono, sets sample rate to 16kHz, and removes silence.
    """
    try:
        command = [
            "ffmpeg",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-af", "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:detection=peak",
            output_path
        ]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully cleaned '{input_path}' and saved to '{output_path}'")
        print(f"ffmpeg stdout: {process.stdout}")
        if process.stderr:
            print(f"ffmpeg stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning audio file '{input_path}':")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean an audio file using ffmpeg.")
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("--output_path", help="Optional: Full path for the output cleaned audio file. If not provided, it's derived from input file name and placed in 'output' directory relative to project root.")
    
    args = parser.parse_args()
    
    input_file_path = args.input_file
    
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        return # Exit if input file not found

    final_output_path = ""
    if args.output_path:
        final_output_path = args.output_path
        # Ensure the directory for the custom output path exists
        output_parent_dir = os.path.dirname(final_output_path)
        if output_parent_dir and not os.path.exists(output_parent_dir):
            os.makedirs(output_parent_dir)
    else:
        # Default behavior: place in audio_pipeline/output directory
        input_basename = os.path.basename(input_file_path)
        output_filename_root, output_ext = os.path.splitext(input_basename)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) # Assumes audio_pipeline/scripts/
        output_dir_target = os.path.join(project_root, "output")

        if not os.path.exists(output_dir_target):
            os.makedirs(output_dir_target)

        final_output_path = os.path.join(output_dir_target, f"{output_filename_root}_cleaned{output_ext if output_ext else '.wav'}")

    print(f"Input file: {input_file_path}")
    print(f"Output file: {final_output_path}")
    
    clean_audio(input_file_path, final_output_path)
