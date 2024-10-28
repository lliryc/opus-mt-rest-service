"""An Opus-MT microservice that translates text from Arabic to English"""

from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import syntok.segmenter as segmenter

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


def translate_text_ar2en(text: str) -> str:
    """Translate the text from Arabic to English.

    Arguments:
        text -- The text to be translated

    Returns:
        The translated text
    """
    translation = []
    src_text = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            src_text.append("".join(token.spacing + token.value for token in sentence))
        translated = MODEL.generate(
            **TOKENIZER(src_text, return_tensors="pt", padding=True)
        )
        tgt_text = [TOKENIZER.decode(t, skip_special_tokens=True) for t in translated]
        translation.extend(tgt_text)
        translation.append("\n")
        # Translating "\n" leads to the model hallucinating a sentence.
        src_text.clear()
    del translation[-1]
    return " ".join(translation)


@app.get("/translate")
def translate(text: str, source: str="ar", target: str="en") -> Response:
    """Translate a single document.

    Arguments:
        text -- A text to be translated
    
    Returns:
        A list of translated sentences.
    """
    if not source.startswith("en"):
        return Response(translation="This model translates from Arabic to English")
    if not target.startswith("de"):
        return Response(translation="This model translates from Arabic to English")
    return Response(translation=translate_text(text))
