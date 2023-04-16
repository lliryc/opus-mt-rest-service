# Opus-MT translation service
This is the companion translation REST service for the OmegaT machine translation available at https://github.com/pikatech/omegat-opus-mt-plugin

It provides an [Opus MT](https://github.com/Helsinki-NLP/Opus-MT) English-to-German model that can be queried via GET requests.
The only parameter that should be set is the 'text' parameter. While it also takes 'source' and 'target' parameters, those are set to "en" and "de", respectively, by default and should be left alone. They are only needed to check if the query posed by OmegaT is valid. Otherwise, the service will reply: "This model translates from English to German."

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
docker run -p 8000:80 <IMAGE ID>
```
Please change the PyTorch installation settings in the Dockerfile to your needs. By default, it is set up to use the CPU.

## Example query
```
curl -X 'GET' \
  'http://localhost:8000/translate?text=I%27m%20an%20example%20text.&source=en&target=de' \
  -H 'accept: application/json'
```

Response:
```
{
  "translation": "Ich bin ein Beispieltext."
}
```
