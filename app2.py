from flask import Flask, render_template, Response, jsonify, url_for, request
import cv2
from fer import FER
import time
from collections import Counter
import threading
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import base64
import requests        # ★ 新增：通过 HTTP 调用 Ollama
import re
import aiml

# ================== AIML 初始化 ==================
bot = aiml.Kernel()
BRAIN_FILE = "bot_brain.brn"
AIML_DIR = "aiml_files"
if os.path.exists(BRAIN_FILE):
    bot.loadBrain(BRAIN_FILE)
else:
    bot.learn(os.path.join(AIML_DIR, "std-startup.xml"))
    bot.respond("LOAD AIML B")
    bot.saveBrain(BRAIN_FILE)

# ================== Flask & 图表 ==================
matplotlib.use('Agg')
app = Flask(__name__)

static_folder = 'static'
charts_folder = os.path.join(static_folder, 'charts')
reports_folder = os.path.join(static_folder, 'reports')
os.makedirs(charts_folder, exist_ok=True)
os.makedirs(reports_folder, exist_ok=True)

# ================== 全局状态 ==================
stop_flag = False
emotion_counts = Counter()
emotion_trends = []
timestamps = []

FAQ_KB = {
    "如何开始": "点击页面上方 Start 按钮即可启动实时情绪检测。",
    "如何停止": "点击 Stop 按钮或直接刷新页面即可停止检测。",
    "数据保存": "截取的照片在 static/ 目录，统计图表在 static/charts/ 文件夹。",
    "生成报告": "点击 Generate Report 按钮生成包含情绪统计和图表的 PDF 报告。",
    "你是谁": "我是情绪识别机器人，负责回答使用说明相关问题。",
}

def faq_reply(msg: str) -> str:
    norm = re.sub(r'[？?]', '', msg.lower()).replace(' ', '')
    for key, ans in FAQ_KB.items():
        if key in norm:
            return ans
    aiml_resp = bot.respond(msg)
    return aiml_resp.strip() or "抱歉，暂时没有找到答案，请尝试换个问法~"

# ================== 摄像头与表情检测 ==================
detector = FER(mtcnn=True)
camera = None
lock = threading.Lock()

def initialize_camera():
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("无法打开摄像头")
    return camera

def detect_and_draw_emotions(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_emotions(rgb_frame)
    for face in detections:
        x, y, w, h = face["box"]
        dominant_emotion = max(face["emotions"], key=face["emotions"].get)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame, detections

def save_frame(frame):
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{ts}.jpg"
    cv2.imwrite(os.path.join(static_folder, filename), frame)
    return filename, ts

def capture_emotions():
    global stop_flag, emotion_counts, emotion_trends, timestamps
    last_capture = time.time()
    interval = 1.0
    while not stop_flag:
        if camera is not None:
            ret, frame = camera.read()
            if ret:
                frame, detections = detect_and_draw_emotions(frame)
                if time.time() - last_capture >= interval:
                    last_capture = time.time()
                    for face in detections:
                        dominant = max(face["emotions"], key=face["emotions"].get)
                        emotion_counts[dominant] += 1
                    _, ts = save_frame(frame)
                    emotion_trends.append(dict(emotion_counts))
                    timestamps.append(ts)
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

def emotion_trends_chart():
    if not emotion_trends:
        return jsonify({'error': 'No emotion trends to display.'})
    fig, ax = plt.subplots()
    all_emotions = set(e for trend in emotion_trends for e in trend)
    for emo in all_emotions:
        ax.plot(timestamps, [t.get(emo, 0) for t in emotion_trends], label=emo)
    ax.set_xlabel("Time"); ax.set_ylabel("Count"); ax.legend()
    filename = os.path.join(charts_folder, 'trend_chart.jpg')
    plt.tight_layout(); plt.savefig(filename); plt.close(fig)
    return jsonify({'image_path': url_for('static', filename='charts/trend_chart.jpg')})

def emotion_proportions():
    if not emotion_counts:
        return jsonify({'error': 'No emotion data to display.'})
    labels, sizes = zip(*emotion_counts.items())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90); ax.axis('equal')
    filename = os.path.join(charts_folder, 'pie_chart.jpg')
    plt.savefig(filename); plt.close(fig)
    return jsonify({'image_path': url_for('static', filename='charts/pie_chart.jpg')})

# ================== ★ 使用 Ollama 生成文本报告 ==================
OLLAMA_URL = "http://localhost:11434/api/generate"  # 默认端口

def generate_ai_report():
    global emotion_counts
    if not emotion_counts:
        return jsonify({'error': 'No emotion data to generate report.'})

    # 构造 Prompt
    summary_lines = [f"- {emo}: {cnt}" for emo, cnt in emotion_counts.items()]
    prompt = (
        "你是一名情绪分析师，请根据下面的人脸情绪检测结果，注意这是同一个人在一段时间的情感"
        ""
        "用简洁的中文进行情绪概况总结、可能原因推测，并给出改善建议：\n\n"
        + "\n".join(summary_lines)
    )

    # 调用 Ollama 本地模型
    payload = {
        "model": "mistral",     # 如果换模型，改这里
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        ai_text = r.json().get("response", "").strip()
        return jsonify({'report': ai_text})
    except Exception as e:
        return jsonify({'error': f'Ollama API error: {str(e)}'})

# ================== Flask 路由 ==================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_emotions(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/statistics')
def statistics():
    return jsonify({
        'emotion_counts': dict(emotion_counts),
        'emotion_trends': emotion_trends,
        'timestamps': timestamps
    })

@app.route('/start')
def start():
    global stop_flag
    with lock:
        stop_flag = False; initialize_camera()
    return "Started"

@app.route('/stop')
def stop():
    global stop_flag, camera
    with lock:
        stop_flag = True
        if camera: camera.release(); camera = None
    return "Stopped"

@app.route('/generate_trend_chart')
def generate_trend_chart(): return emotion_trends_chart()

@app.route('/generate_pie_chart')
def generate_pie_chart(): return emotion_proportions()

@app.route('/generate_ai_report')
def generate_ai_report_route(): return generate_ai_report()

@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({'response': faq_reply(request.json.get('message', ''))})

@app.errorhandler(Exception)
def handle_exception(e): return jsonify({'error': str(e)}), 500

# ================== 可选：Mock 数据快速测试 ==================
@app.route('/mock_data')
def mock_data():
    global emotion_counts, emotion_trends, timestamps
    emotion_counts = Counter({'happy': 6, 'sad': 2, 'angry': 1})
    emotion_trends = [dict(emotion_counts)]
    timestamps = [time.strftime("%Y%m%d_%H%M%S")]
    return "Mock data loaded."

@app.route('/generate_text_report')
def generate_text_report():
    return generate_ai_report()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
