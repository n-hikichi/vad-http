from diagrams import Cluster, Diagram
from diagrams.custom import Custom
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.onprem.client import Client

with Diagram("Multi client architecture", show=False):
    console = Client("Console")
    client = Client("Web Browser")
    app = Flask("Flask App")
    log = Python("Werkzeug(log)")

    with Cluster("Azure Services: ASR, LLM, TTS"):
        asr = Custom("Azure Speech to Text", "../icons/azure-ASR.png")
        llm = Custom("Azure OpenAI API", "../icons/azure-openai.png")
        tts = Custom("Azure Speech Synthesis", "../icons/azure-TTS.png")

    mic = Custom("Microphone", "../icons/mic.png")
    speaker = Custom("Speaker", "../icons/speaker.png")

    client >> app >> asr >> llm >> tts
    app >> log
    mic >> asr
    tts >> speaker
    log >> console
