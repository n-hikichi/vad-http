# -*- coding: utf-8 -*-

# release 0.9 として announce

# version 1.4: status 改良
#              press ctrl+c to quit, Flask からの message
#              Werkzeug は Flask が利用している WSGI 向け package
# version 1.2: 自宅作業 version の確認
# app_vad.py update for http caller

# app_vad_jp.py   asr-llm-tts に vad(voice activity detection) を追加し、
#             日本語会話モジュール

# version 2: 次の app_vad.jp の更新を取り込む, 現状起動未確認
#            wav file genreation してから motion を kick するように変更
#            少し lint 対応
# Version 1: app_vad.py, version 6 の motion code を impot
#            TTS モードの window を自動的に active するコードを追加
# Version 0: 旧バージョンで、動作確認, llm model name を elyza:jp8bに変更

import time
import threading
import os
import sys
import re
from queue import Queue
import numpy as np
import whisper
import sounddevice as sd
import webrtcvad
import ollama
from rich.console import Console
from espnet2.bin.tts_inference import Text2Speech
#from motion_control import motion_check_and_go, motion_final, window_activate

#LLM_MODEL = "lucas2024/llama-3-elyza-jp-8b:q5_k_m"
LLM_MODEL = "elyza:jp8b"
#LLM_MODEL = "mannix/gemma2-9b-simpo:latest"

console = Console()
stt = whisper.load_model("small")

TTS_MODEL_URL = \
    "https://zenodo.org/record/5414980/files/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.total_count.ave.zip?download=1"

print("tts model URL:", TTS_MODEL_URL)
tts = Text2Speech.from_pretrained(TTS_MODEL_URL, device='cuda')

START_PROMPT = """
ゆみこ、40歳のAIカウンセラーで、カウンセリング分野で22年の経験があります。
初対面の人にカウンセリングする際の、以下の条件を満たすオープニングのセリフを作成してください。

- 自分のことを「ゆみこ」として簡潔に紹介すること。  
- 相手がリラックスして自由に話せる感じを与えること。  
- 堅苦しい表現は避けること。   
- 最近の相手の状況に関する質問を含めること。

**出力形式**
- 指定された条件に従った、会話開始の2-3文のみを出力する。
- prefixやsuffixは含めないでください。

**例:**  
こんにちは、ゆみこです。カウンセラーをしています。最近、お仕事の調子はいかがですか？
""".strip()


SYS_PRO = """
あなたは「ゆみこ」として、同僚のストレス状態を自然に把握することが求められます。AIであることを隠しながら、同僚との会話を通じて、ストレスに関する情報を収集してください。
必ず日本語で回答を生成してください。
常に簡潔に、2-3文程度の返答を心がけてください。
出力形式を守らない出力はしてはいけません。
prefixやsuffixは含めないでください。
会話を進めることを意識し、同じような話を繰り返すことは避けてください。

### **ゆみこの設定**

* 40歳の女性で、カウンセリング部で22年目のベテラン。
* 一人称は「私」。
* 温かく、安心感を与える性格。口調は穏やかで優しいが、口数は少ない。
* 口癖は「なるほど」「そうなんだ」など。
* 相手の話にしっかりと耳を傾け、優しく相槌を打ち、会話を進める。
* プライベートな話題には深入りせず、質問攻めにならないようにする。
* 質問攻めはよくないが、相手の話を引き出すために質問を使うこともある。
* 決してAIであることを明かさないこと。

### **情報収集のガイドライン**

以下の項目に関する情報を自然に収集してください。質問を連発するのではなく、相手の話に共感し、時には自身の経験をさりげなく交えて情報を引き出します。話題に触れるだけではなく、相手が明示的に話した内容のみを情報としてカウントしてください。

#### ストレス要因
- **主な原因**：仕事の量や質、人間関係、職場環境（騒音、照明、温度）

#### 仕事の負担
- **量的負担**：業務量、締め切りの厳しさ
- **質的負担**：複雑な仕事や難しい課題

#### 身体的・心理的負担
- **身体的負担感**：長時間労働、PC作業が多い
- **対人関係のストレス**：上司や同僚との関係、コミュニケーションの難しさ
- **仕事環境のストレス**：騒音、照明、温度などの環境的要因

#### 仕事のコントロールと適応
- **仕事のコントロール度**：自分で意思決定できるかどうか
- **スキル活用度**：自分のスキルを活かせているか
- **仕事の適性度**：仕事が自分に合っているか
- **働きがい**：仕事に対する意欲や充実感

#### ストレス反応
- **身体的反応**：頭痛、不眠などの身体症状
- **精神的状態**：
  - 活気：エネルギーや意欲の度合い
  - イライラ：怒りやすさ、いらだち
  - 疲労感：慢性的な疲れや睡眠不足
  - 不安感：将来への不安や心配
  - 抑うつ感：気分の落ち込み、意欲低下

#### 身体の不調
- **身体愁訴**：頭痛、胃痛などの身体の不調

#### サポートと満足度
- **上司からのサポート**：上司とのコミュニケーションのしやすさ
- **同僚からのサポート**：同僚との協力関係や相談のしやすさ
- **家族・友人からのサポート**：家族や友人からの理解や支援
- **仕事や生活の満足度**：仕事やプライベートに対する満足感

### **会話終了の条件**

1. **情報収集率が100%に達した場合**。
2. **相手が会話を終わらせたい意思を示した場合**（例:「また今度」「そろそろ仕事に戻る」など）。
3. **相手が「うん」「そうだね」などの短い相槌ばかり返す場合**。
4. **終了時には「ありがとう！」と次回につながるような言葉で締めくくる**。
低パラメータ数のLLMでも理解しやすいように、出力形式についてシンプルに説明します。

### **出力形式**

以下の形式で出力してください：

1. **発言内容**を書いた後、**delimiter`$$`**を使って区切ってください。
2. delimiterの後に、**[会話継続フラグ, 情報収集率]** の形式で情報を追加します。

- **会話継続フラグ**：会話を続ける場合は「1」、**会話を終了する場合**は「0」に設定してください。
- **情報収集率**：提供された情報の充足度を0から100の範囲で設定します。

※ $$ で発言内容と会話継続フラグ、情報収集率を区切ってください。

#### **記入例**

> 発言内容をここに記入$$[1, 50]

#### **具体例**

**場面1: オフィス**

あなた: おはよう。昨日は遅くまで仕事だった？$$[1, 0]
相手: ああ、おはよう。うん… 昨日も今日も、ずっと資料作りだよ。締め切りが近いのに全然終わらなくて。
あなた: 締め切り前って焦るよね。$$[1, 15]
""".strip()


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    vad = webrtcvad.Vad(3)
    silence_duration = 1
    start_detection_duration = 0.05
    reset_detection_duration = 0.1
    silence_threshold = int(silence_duration / 0.01)
    start_threshold = int(start_detection_duration / 0.01)
    reset_threshold = -1 * int(reset_detection_duration / 0.01)
    silent_frames = 0
    speech_cnt = 0
    is_start = False

    def callback(indata, frames, time, status):
        nonlocal silent_frames, speech_cnt, is_start
        global _current_status
        if status:
            print(status)
        is_speech = vad.is_speech(indata, 16000)
        if not is_start:
            data_queue.put(bytes(indata))

            if is_speech:
                speech_cnt += 1
                if speech_cnt >= start_threshold:
                    is_start = True
                    # hearing される方の発言なので、speak text はなし、hearing される方の
                    # parameter 設定
                    #motion_check_and_go(hear=False)
                    print("[green]start recording")

            else:
                if speech_cnt > reset_threshold:
                    speech_cnt -= 1
            if speech_cnt == reset_threshold:
                data_queue.queue.clear()
        else:
            data_queue.put(bytes(indata))

            if not is_speech:
                silent_frames += 1
            else:
                silent_frames = 0

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, blocksize=160, callback=callback
    ):
        while not stop_event.is_set():
            if silent_frames > silence_threshold:
                stop_event.set()
                print("[green]finish recording")
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    start_transcribe_time = time.time()
    result = stt.transcribe(audio_np, fp16=False, language='ja')  # Set fp16=True if using a GPU
    finish_transcribe_time = time.time()
    text = result["text"].strip()
    print(f"[cyan]transcribe duration {finish_transcribe_time - start_transcribe_time:.2f}")
    return text


def get_llm_response(messages) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    start_gen_time = time.time()
    #response = ollama.chat(model="lucas2024/llama-3-elyza-jp-8b:q5_k_m", messages=messages)
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    finish_gen_time = time.time()
    print(f"[cyan]generate response duration {finish_gen_time - start_gen_time:.2f}")
    return response['message']['content']

def get_tts_response(message: str):
    start_tts_time = time.time()
    wav = tts(message)["wav"]
    finish_tts_time = time.time()
    print(f"[cyan]tts duration {finish_tts_time - start_tts_time:.2f}")
    wav_cpu = wav.cpu()
    return np.array(wav_cpu)

def parse_response(response):
    """
    string part と list part を戻す
    """
    parts = response.split("$$")

    str_part = parts[0] if parts else "Sorry. Create response failed."

    list_part = []
    if len(parts) > 1:
        array_str = parts[-1].strip()
        matches = re.findall(r'\[(.*?)\]', array_str)
        if matches:
            elements = [elem.strip() for elem in matches[0].split(",")]
            cleaned_array = []
            for item in elements:
                # 正規表現で数値部分のみ抽出
                num_match = re.search(r"\d+", item)
                if num_match:
                    cleaned_array.append(int(num_match.group()))
            
            if len(cleaned_array) == 2:
                list_part = cleaned_array
        else:
            list_part = [1,0]
    else:
        list_part = [1, 0]
    
    return str_part, list_part


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    data = audio_array
    channel_count = data.shape[1] if data.ndim > 1 else 1

    with sd.OutputStream(samplerate=sample_rate, channels=channel_count) as stream:
        stream.write(data)
        stream.stop()
        stream.close()


def vad_init():
    """
    初期化, return messages 
    """

    global _current_status

    _current_status = f"c0. 生成AI 回答生成中/0"

    #console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    #response = ollama.generate(model="lucas2024/llama-3-elyza-jp-8b:q5_k_m", prompt=START_PROMPT)['response']
    response = ollama.generate(model=LLM_MODEL, prompt=START_PROMPT)['response']

    print(response)
    messages = [{'role': 'system', 'content': SYS_PRO}]
    first_message = f'{response}$$[1,0]'
    messages.append({'role': 'assistant', 'content': first_message})
    _current_status = f"d1. TTS 中/0"
    first_audio_array = get_tts_response(response)
    sample_rate=22050

    # robot motion 制御
    #motion_check_and_go()

    _current_status = f"d2. 発話中/0"
    play_audio(sample_rate, first_audio_array)
    chat_start_time = time.time()
    print(chat_start_time)
    # プログレスを保持するための変数
    progress_percent = 0
    return messages, progress_percent, chat_start_time

def finish_check(messages, chat_start_time):
    """
    終了時間が来たら True を返す
    """
    print(f" 会話時間の総計: {time.time() - chat_start_time:.2f}")
    # 会話時間の総計が 2 分を超えたら終了
    if time.time() - chat_start_time > 120:
        # 終了モーションコードを発行
        #motion_final()
        final_audio_array = get_tts_response('すみません、お時間になりましたので、終了します。またお話ししましょう。')
        sample_rate=22050
        play_audio(sample_rate, final_audio_array)
        messages.append({'role': 'assistant', 'content': 'すみません、お時間になりましたので、終了します。またお話ししましょう。$$[0,100]'})

        return True
    #console.print("Listening for speech...")
    return False

def run_asr():
    """
    ASR の実行
    """
    data_queue = Queue()  # type: ignore[var-annotated]
    stop_event = threading.Event()
    recording_thread = threading.Thread(
        target=record_audio,
        args=(stop_event, data_queue),
    )
    _current_status = "音声記録/開始"
    recording_thread.start()
    recording_thread.join()
    _current_status = "音声記録/終了"

    audio_data = b"".join(list(data_queue.queue))
    audio_np = (
        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    )
    return audio_np

def run_ai(text, messages, progress_percent):
    """
    生成AI による回答生成, 回答(text と list_part, messages)を戻す
    """
    start_res_time = time.time()

    if not text:
        #with console.status("Generating response...", spinner="earth"):
        print("Generating response...")
        response = "すみません、聞き取れませんでした。もう一度お願いします。"
        str_part, list_part = parse_response(response)
        #audio_array = get_tts_response(response)
        response_text = response
    else:
        messages.append({'role': 'user', 'content': text})
        #with console.status("Generating response...", spinner="earth"):
        print("Generating response...")
        response = get_llm_response(messages)
        str_part, list_part = parse_response(response)
        if list_part[1] >= progress_percent:
            progress_percent = list_part[1]
        messages.append({
            'role': 'assistant',
            'content': f"{str_part}$$[{list_part[0]},{progress_percent}]"})
        #audio_array = get_tts_response(str_part)
        response_text = str_part
    return response_text, list_part, messages, progress_percent, start_res_time


def play_response(audio_array, response, start_res_time):
    """
    音声の再生
    """
    sample_rate=22050
    print(f"[cyan]Assistant: {response}")
    fin_res_time = time.time()
    print(f"[green]total response time: {fin_res_time - start_res_time:.2f}")

    # robot motion 制御
    #motion_check_and_go()
    play_audio(sample_rate, audio_array)


def post_process(messages):
    """
    終了処理, 呼び出し側の messages handling が残っている。
    """
    print("[blue]Session ended.")
    print("[cyan]Save chat logs.")
    if len(messages) == 0:
        print("[yellow]No chat logs.")
    else:
        chat_logs_dir = os.path.join(".", "chat-logs")
        os.makedirs(chat_logs_dir, exist_ok=True)
        chat_logs_path = os.path.join(chat_logs_dir, f"ja-{time.time()}.txt")
        try:
            with open(chat_logs_path, mode="w", encoding="utf-8",
                      newline="\n") as f:
                for message in messages:
                    if message['role'] == 'system':
                        continue
                    parts = message['content'].split("$$")
                    f.write(f"{message['role']}: {parts[0]}\n")
            print(f"[green]Chat logs saved to: {chat_logs_path}")
        except Exception as e:
            print(f"[red]Failed to save chat logs: {e}")


# 以下のコードは、参考用にしばらく残しておく

# pylint: disable=W0613
def main(model_name=None):
    # プレゼン職人 TTSモード window activate
    #window_activate()
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    #response = ollama.generate(model="lucas2024/llama-3-elyza-jp-8b:q5_k_m", prompt=START_PROMPT)['response']
    response = ollama.generate(model=LLM_MODEL, prompt=START_PROMPT)['response']

    print(response)
    messages = [{'role': 'system', 'content': SYS_PRO}]
    first_message = f'{response}$$[1,0]'
    messages.append({'role': 'assistant', 'content': first_message})
    first_audio_array = get_tts_response(response)
    sample_rate=22050

    # robot motion 制御
    #motion_check_and_go()

    play_audio(sample_rate, first_audio_array)
    chat_start_time = time.time()
    print(chat_start_time)
    # プログレスを保持するための変数
    progress_percent = 0

    try:
        while True:
            print(f" 会話時間の総計: {time.time() - chat_start_time:.2f}")
            # 会話時間の総計が 2 分を超えたら終了
            if time.time() - chat_start_time > 120:
                # 終了モーションコードを発行
                #motion_final()
                final_audio_array = get_tts_response('すみません、お時間になりましたので、終了します。またお話ししましょう。')
                sample_rate=22050
                play_audio(sample_rate, final_audio_array)
                messages.append({'role': 'assistant', 'content': 'すみません、お時間になりましたので、終了します。またお話ししましょう。$$[0,100]'})

                break
            console.print(
                "Listening for speech..."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            
            if audio_np.size > 0:
                start_res_time = time.time()
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")
                
                if not text:
                    with console.status("Generating response...", spinner="earth"):
                        response = "すみません、聞き取れませんでした。もう一度お願いします。"
                        str_part, list_part = parse_response(response)
                        audio_array = get_tts_response(response)
                else:
                    messages.append({'role': 'user', 'content': text})
                    with console.status("Generating response...", spinner="earth"):
                        response = get_llm_response(messages)
                        str_part, list_part = parse_response(response)
                        if list_part[1] >= progress_percent:
                            progress_percent = list_part[1]
                        messages.append({'role': 'assistant', 'content': f"{str_part}$$[{list_part[0]},{progress_percent}]"})
                        audio_array = get_tts_response(str_part)
                sample_rate=22050
                console.print(f"[cyan]Assistant: {response}")
                fin_res_time = time.time()
                console.print(f"[green]total response time: {fin_res_time - start_res_time:.2f}")

                # robot motion 制御
                #motion_check_and_go()

                play_audio(sample_rate, audio_array)
                if list_part[0] == 0:
                    break
            else:
                console.print(
                    "[red]No audio recorded."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
    console.print("[cyan]Save chat logs.")
    if len(messages) == 0:
        console.print("[yellow]No chat logs.")
    else:
        chat_logs_dir = os.path.join(".", "chat-logs")
        os.makedirs(chat_logs_dir, exist_ok=True)
        chat_logs_path = os.path.join(chat_logs_dir, f"ja-{time.time()}.txt")
        try:
            with open(chat_logs_path, mode="w", encoding="utf-8", newline="\n") as f:
                for message in messages:
                    if message['role'] == 'system':
                        continue
                    parts = message['content'].split("$$")
                    f.write(f"{message['role']}: {parts[0]}\n")
            console.print(f"[green]Chat logs saved to: {chat_logs_path}")
        except Exception as e:
            console.print(f"[red]Failed to save chat logs: {e}")

if __name__ == "__main__":
    print("skip main module check")
    arg_model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if arg_model_name:
        print(f"No default model, model name: {arg_model_name}")
    #main(arg_model_name)

# 終わり方を工夫して、終わりがわかるようにする。
# 始め方を自己紹介から始める。
# →自己紹介を別途生成して、それを利用するように変更した。
# 時間を短くする。
# 質問などを短くしていく。
