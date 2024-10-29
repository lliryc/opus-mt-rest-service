import requests
import argparse

url = "http://localhost:6777/translate_transcripts"

def translate_text(text, chunk_size=100):
    lines = text.split("\n")
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    responses = []
    for chunk in chunks:
        input_text = "\n".join(chunk)
        if input_text == "":
            continue
        data = {
            "text": input_text
        }
        response = requests.post(url, json=data)
        responses.append(response.json())
    result = ""
    for response in responses:
        result += response["translation"] + "\n"
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text from an input file and write the result to an output file.")
    parser.add_argument("input_file", help="Path to the input file containing text to translate")
    parser.add_argument("output_file", help="Path to the output file where the translated text will be saved")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of lines to process per chunk (default: 50)")

    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as file:
        text = file.read()

    result = translate_text(text, chunk_size=args.chunk_size)

    with open(args.output_file, "w", encoding="utf-8") as file:
        file.write(result)

    print(f"Translation completed. The translated text has been saved to '{args.output_file}'.")
