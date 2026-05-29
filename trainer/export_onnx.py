"""ONNX 导出模块：将训练好的模型导出为 ONNX 格式"""
import os
import json
import shutil
import logging
import numpy as np
import torch

from . import audio_processor

logger = logging.getLogger(__name__)


def _save_config_and_audio(output_dir, voice_name, reference_text, audio_path, model_type, text_mode):
    """Export failed but save config + audio anyway."""
    config = {
        "name": voice_name,
        "reference_text": reference_text,
        "sample_rate": audio_processor.SAMPLE_RATE,
        "model_type": model_type,
        "text_mode": text_mode,
    }
    config_path = os.path.join(output_dir, "voice_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    ref_path = os.path.join(output_dir, "reference_audio.wav")
    if os.path.exists(audio_path):
        try:
            wav, _ = audio_processor.load_and_preprocess(audio_path)
            audio_processor.save_wav(wav.squeeze(0), ref_path)
        except Exception as e:
            logger.warning(f"[Export] 参考音频保存失败: {e}")


def export_model(model_type, checkpoint_path, output_dir,
                 voice_name, reference_text, audio_path,
                 text_mode="pinyin", device=None):
    """Export trained model to ONNX format."""
    if device is None:
        device = torch.device("cpu")

    logger.info(f"[Export] Exporting {model_type} to {output_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint.get("vocab_size", 80)

    if model_type == "fastspeech2":
        _export_fastspeech2_split(checkpoint, vocab_size, device, output_dir)
    else:
        model, hifigan = _load_vits(checkpoint, vocab_size, device)
        wrapper = _VITSWrapper(model, hifigan)
        wrapper.eval()
        dummy_input = torch.randint(1, max(vocab_size, 2), (1, 32), dtype=torch.long).to(device)
        onnx_path = os.path.join(output_dir, "voice_model.onnx")
        try:
            torch.onnx.export(wrapper, dummy_input, onnx_path,
                              input_names=["text_seq"], output_names=["audio_mel"],
                              dynamic_axes={"text_seq": {1: "seq_len"}, "audio_mel": {1: "mel_len"}},
                              opset_version=14, do_constant_folding=True)
        except Exception as e:
            _save_config_and_audio(output_dir, voice_name, reference_text, audio_path, model_type, text_mode)
            raise RuntimeError(f"ONNX export failed: {e}. Checkpoint saved for manual export.")
        _verify_onnx(onnx_path, dummy_input.cpu().numpy())

    # Save config
    config = {
        "name": voice_name,
        "reference_text": reference_text,
        "sample_rate": audio_processor.SAMPLE_RATE,
        "model_type": model_type,
        "text_mode": text_mode,
        "vocab_size": vocab_size,
        "n_mels": audio_processor.N_MELS,
        "training_epochs": checkpoint.get("epoch", 0),
    }
    config_path = os.path.join(output_dir, "voice_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"[Export] Config saved: {config_path}")

    # Save reference audio
    ref_path = os.path.join(output_dir, "reference_audio.wav")
    if os.path.exists(audio_path):
        wav, _ = audio_processor.load_and_preprocess(audio_path)
        audio_processor.save_wav(wav.squeeze(0), ref_path)
        logger.info(f"[Export] Ref audio saved: {ref_path}")

    # Copy checkpoint
    import time as _time
    dst_ckpt = os.path.join(output_dir, "checkpoint.pt")
    for _attempt in range(3):
        try:
            shutil.copy2(checkpoint_path, dst_ckpt)
            break
        except PermissionError:
            _time.sleep(1)
    else:
        logger.warning("[Export] Could not copy checkpoint (file locked), referencing original")

    logger.info(f"[Export] Done! Files in {output_dir}:")
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            logger.info(f"  {f}: {os.path.getsize(fp)/(1024*1024):.2f} MB")


# ============================================================
# FastSpeech2 split export: encoder + decoder + postnet
# LengthRegulator expansion is done in Python (NumPy)
# ============================================================

class _EncoderWrapper(torch.nn.Module):
    """FastSpeech2 encoder + duration predictor."""
    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_encoding = model.pos_encoding
        self.encoder = model.encoder
        self.duration_predictor = model.duration_predictor

    def forward(self, text_seq):
        import math
        x = self.embedding(text_seq) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        duration_pred = self.duration_predictor(x)
        return x, duration_pred


class _DecoderWrapper(torch.nn.Module):
    """FastSpeech2 decoder (expanded hidden -> mel)."""
    def __init__(self, model):
        super().__init__()
        self.pos_encoding = model.pos_encoding
        self.decoder = model.decoder
        self.mel_linear = model.mel_linear

    def forward(self, expanded_hidden):
        x = self.pos_encoding(expanded_hidden)
        for layer in self.decoder:
            x = layer(x)
        return self.mel_linear(x)


class _PostNetWrapper(torch.nn.Module):
    """FastSpeech2 PostNet (mel refinement)."""
    def __init__(self, model):
        super().__init__()
        self.postnet = model.postnet

    def forward(self, mel):
        return self.postnet(mel)


def _export_fastspeech2_split(checkpoint, vocab_size, device, output_dir):
    """Export FastSpeech2 as 3 separate ONNX models."""
    from .fastspeech2 import FastSpeech2Lite
    model = FastSpeech2Lite(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randint(1, max(vocab_size, 2), (1, 32), dtype=torch.long).to(device)

    # 1. Encoder
    encoder = _EncoderWrapper(model).eval()
    enc_path = os.path.join(output_dir, "encoder.onnx")
    logger.info("[Export] Exporting encoder...")
    torch.onnx.export(
        encoder, dummy_input, enc_path,
        input_names=["text_seq"],
        output_names=["hidden", "duration_pred"],
        dynamic_axes={"text_seq": {1: "seq_len"}, "hidden": {1: "seq_len"}, "duration_pred": {1: "seq_len"}},
        opset_version=14, do_constant_folding=True, external_data=False,
    )
    logger.info(f"[Export] Encoder saved: {enc_path}")

    # 2. Decoder
    decoder = _DecoderWrapper(model).eval()
    dec_path = os.path.join(output_dir, "decoder.onnx")
    dummy_expanded = torch.randn(1, 64, model.d_model).to(device)
    logger.info("[Export] Exporting decoder...")
    torch.onnx.export(
        decoder, dummy_expanded, dec_path,
        input_names=["expanded_hidden"],
        output_names=["mel_pred"],
        dynamic_axes={"expanded_hidden": {1: "mel_len"}, "mel_pred": {1: "mel_len"}},
        opset_version=14, do_constant_folding=True, external_data=False,
    )
    logger.info(f"[Export] Decoder saved: {dec_path}")

    # 3. PostNet
    postnet = _PostNetWrapper(model).eval()
    post_path = os.path.join(output_dir, "postnet.onnx")
    dummy_mel = torch.randn(1, 64, model.n_mels).to(device)
    logger.info("[Export] Exporting postnet...")
    torch.onnx.export(
        postnet, dummy_mel, post_path,
        input_names=["mel_in"],
        output_names=["mel_out"],
        dynamic_axes={"mel_in": {1: "mel_len"}, "mel_out": {1: "mel_len"}},
        opset_version=14, do_constant_folding=True, external_data=False,
    )
    logger.info(f"[Export] PostNet saved: {post_path}")

    # 4. HiFi-GAN vocoder
    hifigan_ckpt = os.path.join(os.path.dirname(output_dir) if not os.path.exists(os.path.join(output_dir, "hifigan.pt")) else output_dir, "hifigan.pt")
    # Try same dir first
    hifigan_ckpt = os.path.join(output_dir, "hifigan.pt")
    if not os.path.exists(hifigan_ckpt):
        # Try checkpoint
        hifigan_sd = checkpoint.get("hifigan_state_dict")
        if hifigan_sd is None:
            logger.warning("[Export] No HiFi-GAN weights found, will use Griffin-Lim fallback")
        else:
            from .hifigan_lite import HiFiGANGenerator
            hifigan_model = HiFiGANGenerator().to(device)
            hifigan_model.load_state_dict(hifigan_sd)
            hifigan_model.eval()
            gan_path = os.path.join(output_dir, "hifigan.onnx")
            dummy_mel = torch.randn(1, 80, 100).to(device)
            torch.onnx.export(
                hifigan_model, dummy_mel, gan_path,
                input_names=["mel"], output_names=["wav"],
                dynamic_axes={"mel": {2: "mel_len"}, "wav": {2: "wav_len"}},
                opset_version=14, do_constant_folding=True, external_data=False,
            )
            logger.info(f"[Export] HiFi-GAN saved: {gan_path}")
    else:
        from .hifigan_lite import HiFiGANGenerator
        hifigan_model = HiFiGANGenerator().to(device)
        hifigan_model.load_state_dict(torch.load(hifigan_ckpt, map_location=device))
        hifigan_model.eval()
        gan_path = os.path.join(output_dir, "hifigan.onnx")
        dummy_mel = torch.randn(1, 80, 100).to(device)
        torch.onnx.export(
            hifigan_model, dummy_mel, gan_path,
            input_names=["mel"], output_names=["wav"],
            dynamic_axes={"mel": {2: "mel_len"}, "wav": {2: "wav_len"}},
            opset_version=14, do_constant_folding=True, external_data=False,
        )
        logger.info(f"[Export] HiFi-GAN saved: {gan_path}")


# ============================================================
# Loaders
# ============================================================

def _load_fastspeech2(checkpoint, vocab_size, device):
    from .fastspeech2 import FastSpeech2Lite
    model = FastSpeech2Lite(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _load_vits(checkpoint, vocab_size, device):
    from .vits_lite import VITSLite
    from .hifigan_lite import HiFiGANGenerator
    model = VITSLite(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    hifigan = HiFiGANGenerator().to(device)
    if "hifigan_state_dict" in checkpoint:
        hifigan.load_state_dict(checkpoint["hifigan_state_dict"])
    return model, hifigan


class _VITSWrapper(torch.nn.Module):
    def __init__(self, model, hifigan):
        super().__init__()
        self.model = model
        self.hifigan = hifigan

    def forward(self, text_seq):
        result = self.model(text_seq)
        return result["mel_pred"]


def _verify_onnx(onnx_path, dummy_input):
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input.astype(np.int64)})
        logger.info(f"[Export] ONNX verified! Output shape: {output[0].shape}")
    except Exception as e:
        logger.warning(f"[Export] ONNX verify failed (file saved): {e}")


# ============================================================
# Inference
# ============================================================

def run_inference_onnx(onnx_path, text_seq, providers=None):
    """Single-model ONNX inference (legacy, for VITS)."""
    import onnxruntime as ort
    if providers is None:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: text_seq.astype(np.int64)})
    return output[0]


def run_inference_split(model_dir, text_seq, providers=None):
    """Split-model inference: encoder -> expand -> decoder -> postnet -> HiFi-GAN."""
    import onnxruntime as ort
    if providers is None:
        providers = ["CPUExecutionProvider"]

    enc_path = os.path.join(model_dir, "encoder.onnx")
    dec_path = os.path.join(model_dir, "decoder.onnx")
    post_path = os.path.join(model_dir, "postnet.onnx")
    gan_path = os.path.join(model_dir, "hifigan.onnx")

    # 1. Encoder
    enc_session = ort.InferenceSession(enc_path, providers=providers)
    hidden, duration_pred = enc_session.run(None, {"text_seq": text_seq.astype(np.int64)})

    # 2. Length expansion (NumPy)
    dur_raw = duration_pred.squeeze(-1).squeeze(0)
    dur_mean = dur_raw.mean()

    if dur_mean > 1.5:
        durations = np.clip(np.round(np.exp(dur_raw) - 1), 1, None).astype(int)
    elif dur_mean > 0.5:
        durations = np.clip(np.round(dur_raw), 1, None).astype(int)
    else:
        target_mel_len = max(len(dur_raw) * 5, 64)
        dur_float = np.maximum(dur_raw - dur_raw.min() + 1.0, 0.5)
        dur_float = dur_float / dur_float.sum() * target_mel_len
        durations = np.clip(np.round(dur_float), 1, None).astype(int)

    hidden_sq = hidden.squeeze(0)
    expanded = np.expand_dims(np.repeat(hidden_sq, durations, axis=0), 0)

    # 3. Decoder
    dec_session = ort.InferenceSession(dec_path, providers=providers)
    mel_pred = dec_session.run(None, {"expanded_hidden": expanded.astype(np.float32)})[0]

    # 4. PostNet
    post_session = ort.InferenceSession(post_path, providers=providers)
    mel_post = post_session.run(None, {"mel_in": mel_pred.astype(np.float32)})[0]
    mel_post = np.clip(mel_post, -12.0, 2.0)

    # 5. HiFi-GAN vocoder (if available)
    if os.path.exists(gan_path):
        try:
            gan_session = ort.InferenceSession(gan_path, providers=providers)
            mel_for_gan = mel_post.transpose(0, 2, 1).astype(np.float32)  # (1, 80, T_mel)
            wav = gan_session.run(None, {"mel": mel_for_gan})[0]
            return wav  # (1, 1, T_wav)
        except Exception as e:
            logger.warning(f"[Inference] HiFi-GAN failed, returning mel: {e}")

    # Fallback: return mel for Griffin-Lim
    return mel_post
