"""An Opus-MT microservice that translates text from Arabic to English"""
import uvicorn
import re
from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from multiprocessing import Pool

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

TOKENIZER = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# Serve the Swagger API locally
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )

class Response(BaseModel):
    translation: str

class TranslateRequestModel(BaseModel):
    text: str = "الذبانه موتته كل ساعه ها ها ها مات يريد\nهاي شنو هاي شنو السؤال السخيف لا هذا"
    #text: str = "مرحبا بالعالم" # text in Arabic
    source: str = "ar"
    target: str = "en"

def translate_text_ar2en(text: str) -> str:
    """Translate the text from Arabic to English."""
    src_text = [sentence for sentence in text.split("\n") if sentence != ""]
    
    translated = MODEL.generate(
        **TOKENIZER(src_text, return_tensors="pt", padding=True)
    )
    tgt_text = TOKENIZER.batch_decode(translated, skip_special_tokens=True)
    
    # Process translations in parallel
    with Pool() as pool:
        postprocessed_text = pool.map(remove_repeated_words, tgt_text)
    
    return "\n".join(postprocessed_text)

def remove_repeated_words(text):
    """
    Removes repeated sequences of words from the text.

    Args:
        text (str): The input text.

    Returns:
        str: Text without repeated sequences.
    """
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Remove consecutive duplicate words (case-insensitive)
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # Pattern to match consecutive repeated phrases
    pattern = r'(\b.+?\b)(?:\s+\1\b)+'

    # Apply the pattern iteratively to remove any repeated sequences
    def remove_repeats(text):
        new_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        while new_text != text:
            text = new_text
            new_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        return new_text

    text = remove_repeats(text)

    return text


@app.post("/translate_transcripts")
def translate_transcripts(request: TranslateRequestModel) -> Response:
    """Translate a single document.

    Arguments:
        request -- TranslateRequestModel containing text and optional source/target languages
    
    Returns:
        A Response object containing the translated text.
    """
    if not request.source.startswith("ar"):
        return Response(translation="This model translates from Arabic to English")
    if not request.target.startswith("en"):
        return Response(translation="This model translates from Arabic to English")
    return Response(translation=translate_text_ar2en(request.text))

  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6777)
