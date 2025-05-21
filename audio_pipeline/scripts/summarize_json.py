import json
import argparse
import os
from pathlib import Path
import ollama

DEFAULT_SYSTEM_PROMPT = """Ты мой эффективный AI-ассистент по анализу стенограмм совещаний и лекций.
Вот тебе текст разговора нескольких людей. Сделай из него структурированное саммари на русском языке.
В саммари обязательно выдели следующие пункты (можно использовать маркированные списки):
1. Основные обсуждавшиеся темы или вопросы.
2. Ключевые аргументы, предложения или идеи, высказанные участниками (если были).
3. Принятые решения (если таковые были).
4. Поставленные задачи с указанием ответственных лиц (если это можно однозначно понять из текста).
5. Главные выводы или итоги обсуждения."""

def main():
    parser = argparse.ArgumentParser(description="Summarize a conversation from a tagged JSON file using Ollama.")
    parser.add_argument("tagged_json_filepath", type=str, help="Path to the input tagged JSON file (output of diarize_nemo_auto.py).")
    parser.add_argument("--output_dir", type=str, default="../output/", help="Directory to save the summary (default: ../output/, relative to script).")
    parser.add_argument("--model", type=str, default="gemma:27b", help="Name of the Ollama model to use (e.g., gemma:27b, llama3:8b).")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for waiting for a response from Ollama client.")
    parser.add_argument("--max_chars", type=int, default=15000, help="Maximum characters of the input text to send to LLM. Truncate if longer.")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt to guide the LLM for summarization.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming response from Ollama (wait for full response).")

    args = parser.parse_args()

    input_json_path = Path(args.tagged_json_filepath)

    if not input_json_path.is_file():
        print(f"Error: Input JSON file not found at '{input_json_path}'")
        return

    script_dir = Path(__file__).parent.resolve()
    resolved_output_dir = script_dir / args.output_dir
    
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {resolved_output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{resolved_output_dir}': {e}")
        return

    # Load and prepare conversation text
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            tagged_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'.")
        return
    except Exception as e:
        print(f"Error reading JSON file '{input_json_path}': {e}")
        return

    if not isinstance(tagged_data, list):
        print(f"Error: Expected a list of segments in JSON, got {type(tagged_data)}.")
        return

    conversation_parts = []
    for segment in tagged_data:
        speaker = segment.get("speaker", "Unknown Speaker")
        text = segment.get("text", "").strip()
        if text: # Only add if there's actual text
            conversation_parts.append(f"[{speaker}]: {text}")
    
    conversation_text = "\n".join(conversation_parts)
    
    truncated_notice = ""
    if len(conversation_text) > args.max_chars:
        conversation_text = conversation_text[:args.max_chars]
        truncated_notice = "\n\n[... текст был сокращен ...]"
        conversation_text += truncated_notice
        print(f"Conversation text truncated to {args.max_chars} characters.")

    if not conversation_text.strip():
        print("Error: No text content found in the JSON segments to summarize.")
        return

    # Initialize Ollama client and call API
    try:
        client = ollama.Client(timeout=args.timeout)
        
        messages = [
            {'role': 'system', 'content': args.system_prompt},
            {'role': 'user', 'content': conversation_text},
        ]

        print(f"Sending request to Ollama model '{args.model}'. Streaming: {not args.no_stream}...")
        
        full_summary = ""
        if not args.no_stream:
            response_stream = client.chat(
                model=args.model,
                messages=messages,
                stream=True
            )
            print("\n--- Summary Stream ---")
            for chunk in response_stream:
                content_part = chunk['message']['content']
                print(content_part, end='', flush=True)
                full_summary += content_part
            print("\n--- End of Stream ---")
        else:
            response_dict = client.chat(
                model=args.model,
                messages=messages,
                stream=False
            )
            full_summary = response_dict['message']['content']
            print("\n--- Summary ---")
            print(full_summary)
            print("--- End of Summary ---")

        if truncated_notice and not full_summary.endswith(truncated_notice):
             full_summary += truncated_notice


    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.status_code}")
        if e.response and e.response.content:
             try:
                error_detail = json.loads(e.response.content.decode())
                print(f"Details: {error_detail.get('error')}")
             except json.JSONDecodeError:
                print(f"Details: {e.response.content.decode()}")
        else:
            print(f"Details: {e.error}")

        if e.status_code == 404:
             print(f"Model '{args.model}' not found. Make sure it's pulled with 'ollama pull {args.model}'")
        elif e.status_code == 0 and "connection refused" in str(e.error).lower() : # Heuristic for server down
             print("Could not connect to Ollama. Ensure Ollama server is running.")
        return
    except Exception as e:
        print(f"An unexpected error occurred with Ollama: {e}")
        return

    if not full_summary.strip():
        print("Warning: Received an empty summary from the LLM.")
        # Decide if an empty file should be saved or not. For now, we'll save it.

    # Save summary to TXT and MD
    base_output_filename = input_json_path.stem
    
    summary_txt_path = resolved_output_dir / f"{base_output_filename}_summary.txt"
    summary_md_path = resolved_output_dir / f"{base_output_filename}_summary.md"

    try:
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_summary)
        print(f"Summary saved to TXT: {summary_txt_path}")
    except Exception as e:
        print(f"Error saving summary TXT to '{summary_txt_path}': {e}")

    try:
        # For Markdown, we assume the LLM followed the prompt and generated MD-compatible text.
        # No extra MD conversion is done here on the summary itself.
        with open(summary_md_path, 'w', encoding='utf-8') as f:
            f.write(full_summary)
        print(f"Summary saved to MD: {summary_md_path}")
    except Exception as e:
        print(f"Error saving summary MD to '{summary_md_path}': {e}")

if __name__ == "__main__":
    main()
