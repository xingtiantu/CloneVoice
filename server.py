"""声音克隆训练器 — Flask 后端服务"""
import os
import sys
import json
import uuid
import time
import shutil
import logging
import zipfile
import threading
import numpy as np
import torch

from flask import Flask, request, jsonify, Response, send_file, send_from_directory

# 确保 trainer 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import train, text_processor, audio_processor, device as device_module

import datetime
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"server_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("server")

app = Flask(__name__, static_folder=".", static_url_path="")

# 目录配置
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _make_filename(prefix, ext=".wav", suffix_len=8):
    """生成带日期时间的文件名：{prefix}_{YYYYMMDD_HHMMSS}_{suffix}{ext}"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:suffix_len]}{ext}"


def _cleanup_old_previews(max_age_hours=24):
    """删除 uploads 目录中超过指定时长的预览文件。"""
    now = time.time()
    try:
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith("preview_"):
                fpath = os.path.join(UPLOAD_DIR, fname)
                age_hours = (now - os.path.getmtime(fpath)) / 3600
                if age_hours > max_age_hours:
                    os.remove(fpath)
    except Exception:
        pass


# 推荐参考文本
REFERENCE_TEXTS = [
    "今天天气真好，阳光明媚，适合出去散步。",
    "欢迎来到声音克隆训练器，请清晰朗读这段文字。",
    "人工智能正在改变我们的生活方式，语音合成是其中的重要应用。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "哇，这个消息太让人惊喜了，简直不敢相信自己的耳朵！",
    "没关系，慢慢来，我一直都在你身边陪着你呢。",
    "时间过得真快啊，那些美好的回忆再也回不去了。",
    "各位观众朋友大家好，欢迎收看今天的节目，我是主持人。",
    "生活虽然不易，但正因为有酸甜苦辣，才显得那么真实动人。",
]


# ============================================================
# 静态文件
# ============================================================
@app.route("/")
def index():
    return send_file("index.html")


# ============================================================
# API: 获取推荐文本
# ============================================================
@app.route("/api/reference-texts", methods=["GET"])
def get_reference_texts():
    return jsonify({"texts": REFERENCE_TEXTS})


# ============================================================
# API: 上传/录制参考音频
# ============================================================
@app.route("/api/upload-audio", methods=["POST"])
def upload_audio():
    """接收音频文件（上传或录制）。"""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "未提供音频文件"}), 400

        file = request.files["audio"]
        if not file.filename:
            return jsonify({"error": "文件名为空"}), 400

        # 生成唯一文件名（原始格式）
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"):
            return jsonify({"error": f"不支持的音频格式: {ext}"}), 400

        raw_filename = _make_filename("raw", ext, 8)
        raw_path = os.path.join(UPLOAD_DIR, raw_filename)
        file.save(raw_path)

        # 预处理并获取信息
        try:
            wav, mel = audio_processor.load_and_preprocess(raw_path)
            duration = wav.shape[1] / audio_processor.SAMPLE_RATE
        except Exception as e:
            os.remove(raw_path)
            return jsonify({"error": f"音频处理失败: {e}"}), 400

        # 保存处理后的 WAV（带日期时间命名，统一格式）
        processed_name = _make_filename("voice", ".wav", 8)
        processed_path = os.path.join(UPLOAD_DIR, processed_name)
        audio_processor.save_wav(wav.squeeze(0), processed_path)

        # 原始文件不再需要，删除避免累积
        os.remove(raw_path)

        # 生成波形数据用于前端可视化
        wav_np = wav.squeeze().cpu().numpy()
        # 降采样用于前端绘制（最多 1000 个点）
        step = max(1, len(wav_np) // 1000)
        waveform = wav_np[::step].tolist()

        logger.info(f"[Upload] 音频已上传: {raw_filename} → {processed_name}, 时长: {duration:.2f}s")

        return jsonify({
            "success": True,
            "filename": os.path.basename(processed_path),
            "original_filename": file.filename,
            "duration": round(duration, 2),
            "sample_rate": audio_processor.SAMPLE_RATE,
            "waveform": waveform,
        })

    except Exception as e:
        logger.error(f"[Upload] 上传失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# API: 开始训练
# ============================================================
@app.route("/api/start-training", methods=["POST"])
def start_training():
    """启动训练任务。"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "未提供请求数据"}), 400

        clips = data.get("clips", [])
        model_type = data.get("model_type", "fastspeech2")
        text_mode = data.get("text_mode", "pinyin")
        voice_name = data.get("voice_name", "我的声音").strip() or "我的声音"
        params = data.get("params", {})

        if not clips:
            return jsonify({"error": "未提供音频数据"}), 400

        # 验证每段音频
        clip_paths = []
        for c in clips:
            fn = c.get("filename", "")
            txt = c.get("text", "").strip()
            if not fn or not txt:
                return jsonify({"error": "音频或文本不完整"}), 400
            fp = os.path.join(UPLOAD_DIR, fn)
            if not os.path.exists(fp):
                return jsonify({"error": f"音频文件不存在: {fn}"}), 400
            clip_paths.append({"path": fp, "text": txt, "emotion": c.get("emotion", "neutral")})

        if model_type not in ("fastspeech2", "vits"):
            return jsonify({"error": f"不支持的模型类型: {model_type}"}), 400

        # 生成 job ID
        import datetime as _dt
        _now = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        job_id = f"voice_{_now}_{uuid.uuid4().hex[:4]}"

        # 启动训练（多段音频）
        job = train.start_training(
            job_id=job_id,
            clips=clip_paths,
            model_type=model_type,
            text_mode=text_mode,
            voice_name=voice_name,
            params=params,
        )

        # 获取设备信息
        _, device_label = device_module.get_device()

        logger.info(f"[Train] 训练任务已创建: {job_id}")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "model_type": model_type,
            "text_mode": text_mode,
            "device": device_label,
            "epochs": job.epochs,
        })

    except Exception as e:
        logger.error(f"[Train] 创建训练任务失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# API: 训练状态（SSE）
# ============================================================
@app.route("/api/training-status", methods=["GET"])
def training_status():
    """SSE 端点，实时推送训练进度。"""
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "未提供 job_id"}), 400

    job = train.get_job(job_id)
    if not job:
        return jsonify({"error": f"训练任务不存在: {job_id}"}), 404

    def generate():
        """SSE 事件生成器。"""
        while True:
            events = job.get_events()
            for event in events:
                data = json.dumps(event, ensure_ascii=False)
                yield f"data: {data}\n\n"

                if event.get("type") in ("complete", "error"):
                    return

            # 没有事件时也推送一次状态
            if job.status in ("training", "exporting"):
                job._notify()
            elif job.status in ("done", "error"):
                # 推送最终状态
                elapsed = time.time() - job.start_time if job.start_time else 0
                yield f"data: {json.dumps({'type': 'complete', 'status': job.status, 'elapsed': round(elapsed, 1), 'message': job.error_msg or '训练完成'}, ensure_ascii=False)}\n\n"
                return

            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# API: 试听合成
# ============================================================
@app.route("/api/try-voice", methods=["POST"])
def try_voice():
    """试听：用训练好的模型合成语音。"""
    try:
        data = request.json
        model_id = data.get("model_id")
        text = data.get("text", "").strip()
        speed = data.get("speed", 1.0)
        pitch_shift = data.get("pitch", 0)

        if not model_id:
            return jsonify({"error": "未指定模型"}), 400
        if not text:
            return jsonify({"error": "未输入文本"}), 400

        model_dir = os.path.join(MODEL_DIR, model_id)
        config_path = os.path.join(model_dir, "voice_config.json")

        # 检查分段模型 (encoder/decoder/postnet) 或单文件模型
        enc_path = os.path.join(model_dir, "encoder.onnx")
        onnx_path = os.path.join(model_dir, "voice_model.onnx")

        if not os.path.exists(enc_path) and not os.path.exists(onnx_path):
            return jsonify({"error": "模型文件不存在"}), 400

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        text_mode = config.get("text_mode", "pinyin")
        sample_rate = config.get("sample_rate", 22050)

        # 文本编码
        seq = text_processor.text_to_sequence(text, text_mode)
        text_seq = np.array([seq], dtype=np.int64)

        # ONNX 推理
        providers = device_module.get_onnx_providers()
        device, _ = device_module.get_device()

        if os.path.exists(enc_path):
            # 分段模型: encoder → expand → decoder → postnet
            mel_np = audio_processor.run_inference_split_fast(model_dir, text_seq, providers)
        else:
            # 单文件模型 (VITS 等)
            mel_np = audio_processor.run_inference_onnx_fast(onnx_path, text_seq, providers)

        # 统一使用 vocoder 封装（支持预训练声码器或 Griffin-Lim 回退）
        if mel_np.ndim == 3 and mel_np.shape[0] == 1 and mel_np.shape[1] == 1:
            # 老版本 ONNX 直接输出波形（hifigan.onnx）
            wav = mel_np.squeeze(0).squeeze(0)
        else:
            from trainer.vocoder import mel_to_wav
            mel_tensor = torch.FloatTensor(mel_np.squeeze(0).T)
            checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
            wav = mel_to_wav(mel_tensor, checkpoint_path=checkpoint_path, device=device).numpy()

        # 调整语速
        if speed != 1.0:
            import librosa
            wav = librosa.effects.time_stretch(wav, rate=speed)

        # 调整音调
        if pitch_shift != 0:
            import librosa
            wav = librosa.effects.pitch_shift(wav, sr=sample_rate, n_steps=pitch_shift)

        # 保存为临时文件（带日期时间命名）
        output_filename = _make_filename("preview", ".wav", 8)
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        audio_processor.save_wav(wav, output_path, sr=sample_rate)

        # 清理超过 24 小时的旧预览文件
        _cleanup_old_previews()

        # 波形数据
        step = max(1, len(wav) // 1000)
        waveform = wav[::step].tolist()

        return jsonify({
            "success": True,
            "audio_url": f"/uploads/{output_filename}",
            "duration": round(len(wav) / sample_rate, 2),
            "waveform": waveform,
        })

    except Exception as e:
        logger.error(f"[Preview] 试听失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# API: 模型列表
# ============================================================
@app.route("/api/models", methods=["GET"])
def list_models():
    """列出已训练的模型。"""
    models = []
    if not os.path.exists(MODEL_DIR):
        return jsonify({"models": models})

    for name in sorted(os.listdir(MODEL_DIR), reverse=True):
        model_dir = os.path.join(MODEL_DIR, name)
        if not os.path.isdir(model_dir):
            continue

        config_path = os.path.join(model_dir, "voice_config.json")
        onnx_path = os.path.join(model_dir, "voice_model.onnx")

        info = {
            "id": name,
            "name": name,
            "has_model": os.path.exists(onnx_path),
            "size_mb": 0,
        }

        # 加载配置
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                info["name"] = config.get("name", name)
                info["reference_text"] = config.get("reference_text", "")
                info["model_type"] = config.get("model_type", "unknown")
                info["sample_rate"] = config.get("sample_rate", 22050)
                info["text_mode"] = config.get("text_mode", "pinyin")
            except Exception:
                pass

        # 计算总大小
        total_size = 0
        for f in os.listdir(model_dir):
            fp = os.path.join(model_dir, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
        info["size_mb"] = round(total_size / (1024 * 1024), 2)

        # 修改时间
        info["created"] = os.path.getctime(model_dir)

        models.append(info)

    return jsonify({"models": models})


# ============================================================
# API: 导出模型
# ============================================================
@app.route("/api/export/<model_id>", methods=["GET"])
def export_model(model_id):
    """导出模型为 zip。"""
    model_dir = os.path.join(MODEL_DIR, model_id)
    if not os.path.exists(model_dir):
        return jsonify({"error": "模型不存在"}), 404

    # 读取配置获取音色名称
    config_path = os.path.join(model_dir, "voice_config.json")
    zip_name = model_id
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            zip_name = config.get("name", model_id)
        except Exception:
            pass

    # 创建 zip
    zip_path = os.path.join(UPLOAD_DIR, f"{model_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 打包模型文件（支持分段和单文件两种格式）
        export_files = [
            "encoder.onnx", "decoder.onnx", "postnet.onnx",
            "voice_model.onnx", "voice_config.json", "reference_audio.wav",
        ]
        for fname in export_files:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)

    # 清理名称（去除非法字符）
    safe_name = "".join(c for c in zip_name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        safe_name = model_id

    return send_file(
        zip_path,
        as_attachment=True,
        download_name=f"{safe_name}.zip",
        mimetype="application/zip",
    )


# ============================================================
# API: 重命名模型
# ============================================================
@app.route("/api/models/<model_id>/rename", methods=["POST"])
def rename_model(model_id):
    """重命名模型。"""
    data = request.json
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "名称不能为空"}), 400

    model_dir = os.path.join(MODEL_DIR, model_id)
    config_path = os.path.join(model_dir, "voice_config.json")

    if not os.path.exists(model_dir):
        return jsonify({"error": "模型不存在"}), 404

    # 更新配置
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    config["name"] = new_name

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "name": new_name})


# ============================================================
# API: 删除模型
# ============================================================
@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """删除模型。"""
    model_dir = os.path.join(MODEL_DIR, model_id)
    if not os.path.exists(model_dir):
        return jsonify({"error": "模型不存在"}), 404

    shutil.rmtree(model_dir)
    logger.info(f"[Model] 已删除模型: {model_id}")
    return jsonify({"success": True})


# ============================================================
# API: 设备信息
# ============================================================
@app.route("/api/device-info", methods=["GET"])
def device_info():
    """获取当前设备信息。"""
    _, label = device_module.get_device()
    return jsonify({"device": label})


# ============================================================
# 验证音频文件是否存在（解决 localStorage 残留问题）
# ============================================================
@app.route("/api/verify-clips", methods=["POST"])
def verify_clips():
    """检查上传的音频文件是否仍存在，前端用于清理 localStorage 中的残留记录。"""
    data = request.get_json(silent=True)
    if not data or "filenames" not in data:
        return jsonify({"error": "缺少 filenames 参数"}), 400
    result = {}
    for fn in data["filenames"]:
        result[fn] = os.path.isfile(os.path.join(UPLOAD_DIR, fn))
    return jsonify(result)


# ============================================================
# 静态文件：上传目录
# ============================================================
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ============================================================
# CORS
# ============================================================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="声音克隆训练器服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    logger.info(f"=" * 50)
    logger.info(f"声音克隆训练器 v1.0")
    logger.info(f"访问地址: http://{args.host}:{args.port}")
    logger.info(f"=" * 50)

    try:
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        raise


