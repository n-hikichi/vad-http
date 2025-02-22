from diagrams import Cluster, Diagram
from diagrams.custom import Custom
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.onprem.client import Client

with Diagram("Flask Application Architecture", show=False):
    console = Client("Console")
    client = Client("Web Browser")
    app = Flask("Flask App")
    log = Python("Werkzeug(log)")
    with Cluster("builtin/asr-llm-tts"):
        asr = Custom("ASR", "icons/ASR.png")
        llm = Custom("LLM(Ollama)", "icons/LLM.png")
        tts = Custom("TTS", "icons/TTS.png")
    mic = Custom("Microphone", "icons/mic.png")
    speaker = Custom("Speaker", "icons/speaker.png")

    client >> app >> asr >> llm >> tts
    app >> log
    mic >> asr
    tts >> speaker
    log >> console
