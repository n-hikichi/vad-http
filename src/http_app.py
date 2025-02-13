# -*- coding: utf-8 -*-

""" http module for Web APP """

#
# 開始時は必ず "開始" ボタンをクリックする
#

# release 0.9 として announce

# version 1.4: status 改良
#              press ctrl+c to quit, Flask からの message
#              Werkzeug は Flask が利用している WSGI 向け package

SYSTEM_VERSION = "r0.9"
#SYSTEM_VERSION = "v1.4"
#SYSTEM_VERSION = "v1.3"

CONVERSATION_END_CYCLE = 10
#CONVERSATION_END_CYCLE = 3
SYSTEM_TITLE = f"ストレスチェック向け会話システム({SYSTEM_VERSION})"


from flask import Flask, jsonify, request
import threading
import time


# (a) システム起動中（モデルのロードなど）
_current_status = "a1/- 会話システム起動中 (ASR/TTS/生成AIモデル start loading)"
# モデルロードなどの処理を実施
from app_vad import (
    vad_init, finish_check, run_asr, run_ai, play_response, post_process,
    transcribe, get_tts_response
)
_current_status = "a1/+ 会話システム (ASR/TTS/生成AIモデル finished loading), '開始'ボタン押下待ち"

app = Flask(__name__)


# グローバル変数で会話システムのスレッド、停止フラグ、現在のステータスを管理
_conversation_thread = None
_finish_flag = False
_quit_flag = False
#_current_status = "システム未起動"


def conversation_system():
    """
    会話システムの処理例
    実際はASR/TTS/生成AI の処理が入る想定ですが、ここでは各ステータスをシミュレーションします。
    """
    global _finish_flag, _quit_flag, _current_status

    messages, progress_percent, chat_start_time = vad_init()
    try:

        cycle = 0
        inner_exit = False
        # 会話ループ：ユーザによる停止指示があるか、または自動終了条件に達するまで繰り返す
        while not _finish_flag:
            cycle += 1
            # (b1) マイク入力待機中
            _current_status = f"b1. マイク入力待機中/{cycle}"
            if finish_check(messages, chat_start_time):
                _current_status = f"e. 会話システム終了/制限時間到達, '開始'ボタン押下待ち"
                inner_exit = True
                break
            if _quit_flag:
                break

            # (b2) ASR 中
            _current_status = f"b2. マイク入力処理・ASR 中/{cycle}"
            audio = run_asr()
            # pylint: disable=C0325
            if not (audio.size > 0):
                _current_status = f"e. 会話システム終了/音声の記録がない"
                inner_exit = True
                break
            # 文字起こし
            text = transcribe(audio)
            print(f"[yellow]You: {text}")
            if _quit_flag:
                break

            # (c) 生成AI 回答生成中
            _current_status = f"c. 生成AI 回答生成中/{cycle}"
            response, list_part, messages, progress_percent, start_res_time = \
                run_ai(text, messages, progress_percent)
            if _quit_flag:
                break

            # (d1) TTS 中
            _current_status = f"d1. TTS 中/{cycle}"
            audio_array = get_tts_response(response)
            if _quit_flag:
                break

            # (d2) 発話中
            _current_status = f"d2. 発話中/{cycle}"
            play_response(audio_array, response, start_res_time)
            if list_part[0] == 0:
                _current_status = f"e. 会話システム終了/list_part[0] == 0"
                inner_exit = True
            if _quit_flag:
                break

            # 例として、CONVERSATION_END_CYCLEサイクルで自動終了（必要に応じて条件を変更）
            if cycle >= CONVERSATION_END_CYCLE:
                break

        if not inner_exit:
            # (e) 会話システム自動終了
            _current_status = \
  f"e. 会話システム終了/{cycle}, finish_flag:{_finish_flag},quit_flag:{_quit_flag}"
    finally:
        # 終了後の後始末
        _finish_flag = False
        _quit_flag = False
        post_process(messages)

@app.route("/")
def index():
    """
    クライアント用の HTML ページを返す
    ※開始・終了ボタン・強制終了と、ステータス表示エリアを備えたシンプルな実装例
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{SYSTEM_TITLE}</title>
    </head>
    <body>
        <h1>{SYSTEM_TITLE}</h1>
        <button onclick="startSystem()">開始</button>
        <button onclick="finishSystem()">終了</button>
        <button onclick="quitSystem()">強制終了</button>
        <h2>ステータス:</h2>
        <div id="statusArea">システム未起動</div>
        
        <script>
            // 開始ボタン押下時の処理
            function startSystem() {{
                fetch('/start')
                .then(response => response.json())
                .then(data => {{
                    console.log(data.message);
                }});
            }}
            // 終了ボタン押下時の処理
            function finishSystem() {{
                fetch('/finish')
                .then(response => response.json())
                .then(data => {{
                    console.log(data.message);
                }});
            }}
            // 強制終了ボタン押下時の処理
            function quitSystem() {{
                fetch('/quit')
                .then(response => response.json())
                .then(data => {{
                    console.log(data.message);
                }});
            }}
            // 定期的にサーバからステータスを取得し、画面上に表示する
            function getStatus() {{
                fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('statusArea').innerText = data.status;
                }});
            }}
            // 1秒ごとにステータスを更新
            setInterval(getStatus, 1000);
        </script>
    </body>
    </html>
    """

@app.route("/start", methods=["GET"])
def send_start():
    """
    /start エンドポイント：会話システムの開始処理を行う
    """
    global _conversation_thread, _finish_flag, _quit_flag
    if _conversation_thread is None or not _conversation_thread.is_alive():
        _finish_flag = False
        _quit_flag = False
        _conversation_thread = threading.Thread(target=conversation_system)
        _conversation_thread.start()
        return jsonify({"message": "会話システムを開始しました"})
    else:
        return jsonify({"message": "会話システムは既に起動中です"})

@app.route("/finish", methods=["GET"])
def send_finish():
    """
    /finish エンドポイント：会話システムの終了指示を送る
    """
    global _finish_flag
    _finish_flag = True
    return jsonify({"message": "会話システム終了の指示を送りました"})

@app.route("/quit", methods=["GET"])
def send_quit():
    """
    /finish エンドポイント：会話システムの強制終了指示を送る
    """
    global _quit_flag
    _quit_flag = True
    return jsonify({"message": "会話システム強制終了の指示を送りました"})

@app.route("/status", methods=["GET"])
def send_status():
    """
    /status エンドポイント：現在の会話システムのステータスを返す
    """
    return jsonify({"status": _current_status})

if __name__ == '__main__':
    app.run(debug=True)
