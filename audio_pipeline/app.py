import os
import subprocess
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, Response

# Flask App Initialization
app = Flask(__name__)

# Configuration
APP_ROOT = Path(__file__).parent.resolve() # This is audio_pipeline/
UPLOAD_FOLDER = APP_ROOT / 'input_audio'
OUTPUT_FOLDER = APP_ROOT / 'output' # Default output for the pipeline
SCRIPTS_DIR = APP_ROOT / 'scripts'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac'} # Added more common types

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER) # For easy access if needed by app
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Ensure upload and output folders exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Helper Function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for Index Page
@app.route('/')
def index():
    # This will render templates/index.html.
    # We'll create index.html in a subsequent step.
    return render_template('index.html') 

# Route for Running the Pipeline
@app.route('/run', methods=['POST'])
def run_the_pipeline():
    # --- File Handling ---
    if 'audio_file' not in request.files:
        return jsonify(error="No audio_file part in the request"), 400
    
    file = request.files['audio_file']

    if file.filename == '':
        return jsonify(error="No audio file selected"), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_audio_path = UPLOAD_FOLDER / filename
        try:
            file.save(str(saved_audio_path))
        except Exception as e:
            return jsonify(error=f"Failed to save uploaded file: {str(e)}"), 500
    else:
        return jsonify(error="Invalid file type. Allowed types: " + ", ".join(app.config['ALLOWED_EXTENSIONS'])), 400

    # --- Get Parameters from Form ---
    skip_clean = 'skip_clean' in request.form
    skip_summary = 'skip_summary' in request.form
    
    transcribe_model_size = request.form.get('transcribe_model_size', 'large-v3')
    transcribe_lang = request.form.get('transcribe_lang', 'ru')
    transcribe_prompt = request.form.get('transcribe_prompt', '')
    diarize_max_speakers = request.form.get('diarize_max_speakers', '5')
    summarize_model = request.form.get('summarize_model', 'gemma:27b') # Default from run_pipeline.py
    summarize_max_chars = request.form.get('summarize_max_chars', '15000') # Default from run_pipeline.py
    
    # output_dir_base for run_pipeline.py will be the app's OUTPUT_FOLDER
    pipeline_output_dir = str(app.config['OUTPUT_FOLDER'])

    # --- Construct run_pipeline.py Command ---
    pipeline_script_path = str(SCRIPTS_DIR / 'run_pipeline.py')
    command = ["python", pipeline_script_path, str(saved_audio_path)]

    if skip_clean:
        command.append("--skip_clean")
    if skip_summary:
        command.append("--skip_summary")
    
    command.extend(["--transcribe_model_size", transcribe_model_size])
    command.extend(["--transcribe_lang", transcribe_lang])
    if transcribe_prompt: # Only add if not empty, as run_pipeline.py has its own default
        command.extend(["--transcribe_prompt", transcribe_prompt])
    command.extend(["--diarize_max_speakers", diarize_max_speakers])
    command.extend(["--summarize_model", summarize_model])
    command.extend(["--summarize_max_chars", summarize_max_chars])
    command.extend(["--output_dir_base", pipeline_output_dir]) # Pass the absolute path

    # --- Execute Pipeline (Simplified Blocking Version) ---
    try:
        # Using a longer timeout as the pipeline can be lengthy
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1800) # 30 min timeout

        # Determine output file paths based on conventions in run_pipeline.py
        # The stem used for output files depends on whether cleaning was skipped.
        original_audio_stem = Path(saved_audio_path).stem
        
        # If cleaning was done, run_pipeline.py appends "_cleaned" to the stem for subsequent files.
        # If cleaning was skipped, it uses the original stem.
        processed_audio_stem = f"{original_audio_stem}_cleaned" if not skip_clean else original_audio_stem

        # Construct relative paths from APP_ROOT for links
        relative_output_dir = Path(pipeline_output_dir).relative_to(APP_ROOT)

        # Key output files (as produced by convert_tagged_json_to_txt_md.py and summarize_json.py)
        # These are based on the *processed_audio_stem* (which could be original_stem or original_stem_cleaned)
        # and then further suffixed by subsequent scripts.
        
        # From convert_tagged_json_to_txt_md.py (input is <processed_audio_stem>_tagged.json)
        transcript_md_filename = f"{processed_audio_stem}_tagged.md"
        transcript_md_path = str(relative_output_dir / transcript_md_filename)

        summary_md_filename = ""
        summary_md_path = ""
        if not skip_summary:
            # From summarize_json.py (input is <processed_audio_stem>_tagged.json)
            summary_md_filename = f"{processed_audio_stem}_tagged_summary.md"
            summary_md_path = str(relative_output_dir / summary_md_filename)

        return jsonify(
            status="success",
            message="Pipeline completed successfully.",
            stdout_log=process.stdout, # Full stdout for debugging or info
            # Provide relative paths for client-side linking
            transcript_file_path=transcript_md_path,
            summary_file_path=summary_md_path if not skip_summary else None
        )
    except subprocess.CalledProcessError as e:
        return jsonify(status="error", message=f"Pipeline failed with exit code {e.returncode}.", stdout_log=e.stdout, stderr_log=e.stderr), 500
    except subprocess.TimeoutExpired:
        return jsonify(status="error", message="Pipeline execution timed out after 30 minutes."), 500
    except Exception as e: # Catch any other unexpected errors
        return jsonify(status="error", message=f"An unexpected error occurred: {str(e)}"), 500

# Main Execution Block
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) # Running on a different port for clarity if other apps use 5000
```
