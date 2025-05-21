# Local Audio Transcription and Summarization Pipeline

## Overview

This project provides a local pipeline for processing audio files. It performs transcription, speaker diarization, and summarization using locally runnable AI tools. The pipeline is inspired by and based on concepts from a Habr article discussing similar local AI workflows. It's designed to offer a private and cost-effective way to process audio data.

The pipeline takes an audio file as input and sequentially processes it through several stages:
1.  **Audio Cleaning**: Uses `ffmpeg` to convert the audio to a standard format (mono, 16kHz) and remove silence.
2.  **Transcription**: Uses OpenAI Whisper to generate a text transcript of the audio.
3.  **Speaker Diarization**: Uses NVIDIA NeMo to identify different speakers and assign parts of the transcript to them.
4.  **Summarization**: Uses a local Large Language Model (LLM) via Ollama to create a structured summary of the conversation.

## Features

*   **Audio Cleaning**: Pre-processes audio for better transcription quality using `ffmpeg` (mono conversion, 16kHz sample rate, silence removal).
*   **Accurate Transcription**: Leverages OpenAI Whisper for high-quality speech-to-text.
*   **Speaker Diarization**: Identifies and attributes speech segments to different speakers using NVIDIA NeMo.
*   **Local Summarization**: Generates summaries using Ollama with models like Gemma, Llama, etc., ensuring data privacy.
*   **Multiple Output Formats**: Produces results in various formats:
    *   Raw transcription text (`.txt`)
    *   Transcription segments with timestamps (`.json`)
    *   Speaker-tagged transcription (`_tagged.json`, `.txt`, `.md`)
    *   Summaries (`_summary.txt`, `_summary.md`)

## Based On

This project is heavily inspired by the following article:
*   **Habr Article:** [Локальный AI-ассистент для анализа совещаний и лекций: Whisper, NeMo, Ollama + Gemma](https://habr.com/ru/companies/alfa/articles/909498/)

## Directory Structure

*   `scripts/`: Contains all the Python scripts that make up the pipeline.
*   `input_audio/`: (Recommended) A place to store your input audio files. The pipeline can process files from any path.
*   `output/`: The default directory where all processed files (transcripts, diarized text, summaries) are saved.
*   `README.md`: This file.
*   `requirements.txt`: Python dependencies for the project.

## Setup Instructions

Follow these steps to set up and run the audio processing pipeline:

**1. Clone the Repository**
```bash
# If this project is in a Git repository:
# git clone <repository_url>
# cd <repository_name>
```
(Assuming you have already downloaded or cloned the project files.)

**2. Install System Dependencies**

*   **`ffmpeg`**: Required for audio cleaning.
    *   On Debian/Ubuntu:
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
    *   On macOS (using Homebrew):
        ```bash
        brew install ffmpeg
        ```
    *   For other systems, please refer to the official [ffmpeg download page](https://ffmpeg.org/download.html).

**3. Install Python Dependencies**

*   **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    # On Windows, use: .venv\Scripts\activate
    ```
*   **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on `nemo_toolkit`**: The installation of `nemo_toolkit[all]` can be complex and time-consuming. It might have additional system dependencies (like `libsndfile1`). If you encounter issues, please consult the [official NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) for detailed installation instructions and troubleshooting.

**4. Set up Ollama**

*   **Install Ollama:** Download and install Ollama from the [official website](https://ollama.com).
*   **Pull an LLM Model:** You need to have a model available for Ollama to use for summarization. The default model for this pipeline is `gemma:27b`. You can pull it (or another model) using:
    ```bash
    ollama pull gemma:27b 
    # Or, for example, llama3:8b
    # ollama pull llama3:8b 
    ```
    Ensure the Ollama server is running before executing the pipeline if you intend to use the summarization step.

## Running the Pipeline

The main script to run the entire pipeline is `run_pipeline.py` located in the `scripts/` directory.

**Basic Usage Example:**

Assuming your project is in a directory named `audio_pipeline` and you are in that directory:
```bash
python scripts/run_pipeline.py path/to/your/audio.wav
```
Or, if your audio file is in the `input_audio` directory:
```bash
python scripts/run_pipeline.py input_audio/my_meeting.mp3
```

**Command-line Arguments for `run_pipeline.py`:**

*   `filepath`: (Required) Path to the input audio file.
*   `--skip_clean`: Skip the audio cleaning step.
*   `--skip_summary`: Skip the summarization step.
*   `--transcribe_model_size`: Whisper model size for transcription (default: `large-v3`). Other options include `tiny`, `base`, `small`, `medium`.
*   `--transcribe_lang`: Language for transcription (default: `ru`). Use `en` for English, etc.
*   `--diarize_max_speakers`: Maximum number of speakers expected for diarization (default: `5`).
*   `--summarize_model`: Ollama model to use for summarization (default: `gemma:27b`).
*   `--output_dir_base`: Base directory for all output files (default: `../output`, which resolves to `audio_pipeline/output/` if running from the project root).

Example with more options:
```bash
python scripts/run_pipeline.py "input_audio/lecture.ogg" --transcribe_lang en --diarize_max_speakers 2 --summarize_model llama3:8b
```

## Output

The pipeline generates several files in the specified output directory (default is `output/`). Assuming your input file was `meeting_audio.wav` and it was cleaned:

*   `output/meeting_audio_cleaned.wav`: The cleaned audio file.
*   `output/meeting_audio_cleaned.json`: Raw Whisper transcription segments with timestamps.
*   `output/meeting_audio_cleaned.txt`: Plain text from Whisper transcription.
*   `output/meeting_audio_cleaned_tagged.json`: Transcription segments with speaker labels from diarization.
*   `output/meeting_audio_cleaned_tagged.txt`: Formatted text with speaker labels.
*   `output/meeting_audio_cleaned_tagged.md`: Markdown formatted text with speaker labels.
*   `output/meeting_audio_cleaned_tagged_summary.txt`: Text summary from the LLM.
*   `output/meeting_audio_cleaned_tagged_summary.md`: Markdown formatted summary from the LLM.

If audio cleaning is skipped, filenames will be based on the original input audio filename.

## Troubleshooting

*   **NeMo Installation Issues**:
    *   Ensure you have all necessary build tools and libraries (like `libsndfile1` on Linux).
    *   Refer to the [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) for specific guidance.
    *   Installation can take a long time.
*   **Ollama Server Not Running**:
    *   If summarization fails with a connection error, ensure the Ollama application is running in the background.
    *   You can test Ollama with `ollama list` in your terminal.
*   **Ollama Model Not Pulled**:
    *   If summarization reports the model is not found (e.g., HTTP 404 error from Ollama), make sure you have pulled the model using `ollama pull <model_name>`.
*   **`ffmpeg` Not Found**:
    *   Ensure `ffmpeg` is installed and accessible in your system's PATH.
*   **Python Version**:
    *   This project is developed with Python 3. Ensure you are using a compatible Python version (e.g., Python 3.9+).

This README provides a comprehensive guide to get started with the local audio processing pipeline.
