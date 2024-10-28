# Opus-MT translation service


It provides an [Opus MT](https://github.com/Helsinki-NLP/Opus-MT) Arabic-English model that can be queried via POST requests.
The only parameter that should be set is the 'text' parameter. While it also takes 'source' and 'target' parameters, those are set to "ar" and "en", respectively.

After installing the dependencies via the usual
```
pip install -r requirements.txt
```
and picking your preferred version of PyTorch, you can start the service with
```
uvicorn opus_mt:app
```
Alternatively, you can build a Docker image with
```
docker build .
```
and run the service in a container via
```
docker run -p 6777:6777 <IMAGE ID>
```
Please change the PyTorch installation settings in the Dockerfile to your needs. By default, it is set up to use the GPU.

## Example query
```
curl -X 'POST' \
  'http://localhost:6777/translate_transcripts' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "الذبانه موتته كل ساعه ها ها ها مات يريد\nهاي شنو هاي شنو السؤال السخيف لا هذا",
  "source": "ar",
  "target": "en"
}'
```
