from diagrams import Diagram
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.onprem.client import Client


with Diagram("Vad-http Architecture", show=False):
    client = Client("Web Browser")
    app = Flask("Flask App")
    log = Python("Werkzeug(log)")
    asr_llm_tts = Python("asr-llm-tts")
    ollama = Python("Ollama(LLM)")

    client >> app >> asr_llm_tts >> ollama
    app >> log
