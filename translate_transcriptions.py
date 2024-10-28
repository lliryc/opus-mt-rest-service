import requests

url = "http://localhost:6777/translate_transcripts"

def translate_text(text, chunk_size=100):
    lines = text.split("\n")
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    responses = []
    for chunk in chunks:
        data = {  
            "text": "\n".join(chunk)
        }
        response = requests.post(url, json=data)
        responses.append(response.json())
    result = ""
    for response in responses:
      result += response["translation"] + "\n"
    return result

if __name__ == "__main__":
    with open("transcriptions_example.txt", "r") as file:
        text = file.read()
    
    result = translate_text(text, chunk_size=2)
    
    with open("translated_transcriptions.txt", "w") as file:
        file.write(result)
