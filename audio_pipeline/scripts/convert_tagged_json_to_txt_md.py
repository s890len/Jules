import json
import argparse
import os
from pathlib import Path

def generate_txt_output(tagged_segments):
    """
    Generates plain text output from tagged segments.
    Format: [Speaker_X]: Text of the segment
    """
    output_lines = []
    for segment in tagged_segments:
        speaker = segment.get("speaker", "Unknown Speaker")
        text = segment.get("text", "")
        output_lines.append(f"[{speaker}]: {text}")
    return "\n".join(output_lines)

def generate_md_output(tagged_segments):
    """
    Generates Markdown output from tagged segments.
    Format: **[Speaker_X]:** Text of the segment
    (with double newline for paragraph breaks)
    """
    output_lines = []
    for segment in tagged_segments:
        speaker = segment.get("speaker", "Unknown Speaker")
        text = segment.get("text", "")
        output_lines.append(f"**[{speaker}]:** {text}")
    return "\n\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(description="Convert speaker-tagged JSON to TXT and Markdown formats.")
    parser.add_argument("tagged_json_filepath", type=str, help="Path to the input tagged JSON file (output of diarize_nemo_auto.py).")
    parser.add_argument("--output_dir", type=str, default="../output/", help="Directory to save output TXT and MD files (default: ../output/, relative to script).")

    args = parser.parse_args()

    input_json_path = Path(args.tagged_json_filepath)

    if not input_json_path.is_file():
        print(f"Error: Input JSON file not found at '{input_json_path}'")
        return

    # Determine output directory
    script_dir = Path(__file__).parent.resolve()
    resolved_output_dir = script_dir / args.output_dir
    
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {resolved_output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{resolved_output_dir}': {e}")
        return

    # Load the tagged JSON file
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            tagged_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'. File might be corrupted or not valid JSON.")
        return
    except Exception as e:
        print(f"Error reading JSON file '{input_json_path}': {e}")
        return

    if not isinstance(tagged_data, list):
        print(f"Error: Expected a list of segments in JSON file, but got type '{type(tagged_data)}'.")
        return

    # Generate TXT output
    txt_content = generate_txt_output(tagged_data)
    txt_output_filename = f"{input_json_path.stem}.txt" # Based on input JSON filename, e.g., input_tagged.txt
    txt_output_filepath = resolved_output_dir / txt_output_filename
    
    try:
        with open(txt_output_filepath, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        print(f"Successfully saved TXT output to: {txt_output_filepath}")
    except Exception as e:
        print(f"Error saving TXT file to '{txt_output_filepath}': {e}")

    # Generate Markdown output
    md_content = generate_md_output(tagged_data)
    md_output_filename = f"{input_json_path.stem}.md" # Based on input JSON filename, e.g., input_tagged.md
    md_output_filepath = resolved_output_dir / md_output_filename

    try:
        with open(md_output_filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Successfully saved Markdown output to: {md_output_filepath}")
    except Exception as e:
        print(f"Error saving Markdown file to '{md_output_filepath}': {e}")

if __name__ == "__main__":
    main()
