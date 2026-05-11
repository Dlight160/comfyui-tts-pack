import gc
import io
import os
import queue
import sys
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# cosyvoice: submodule root (FunAudioLLM/CosyVoice) that contains cosyvoice/ package
cv_root = os.path.join(current_dir, 'cosyvoice')
if os.path.isdir(cv_root):
    sys.path.insert(0, cv_root)

# fishspeech: submodule root (fishaudio/fish-speech) that contains fish_speech/ package
fs_root = os.path.join(current_dir, 'fishspeech')
if os.path.isdir(fs_root):
    sys.path.insert(0, fs_root)

# Matcha-TTS is a nested submodule inside cosyvoice submodule
matcha = os.path.join(current_dir, 'cosyvoice', 'third_party', 'Matcha-TTS')
if os.path.isdir(matcha):
    sys.path.append(matcha)

# PYTHONPATH for vLLM worker subprocesses (they inherit PYTHONPATH, not sys.path)
pp = os.environ.get('PYTHONPATH', '')
new_pp_parts = [p for p in [cv_root, current_dir] if p]
if pp:
    new_pp_parts.append(pp)
os.environ['PYTHONPATH'] = os.pathsep.join(new_pp_parts)

# Monkey-patch vLLM EngineArgs to default enforce_eager=True,
# so we don't need to modify the cosyvoice submodule source.
try:
    from vllm.engine.arg_utils import EngineArgs as _EngineArgs
    _engine_args_init = _EngineArgs.__init__
    def _engine_args_patched(self, *args, **kwargs):
        kwargs.setdefault('enforce_eager', True)
        _engine_args_init(self, *args, **kwargs)
    _EngineArgs.__init__ = _engine_args_patched
except ImportError:
    pass

# ── Imports ──────────────────────────────────────────────────────────────────

from cosyvoice.cli.cosyvoice import CosyVoice2
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    launch_thread_safe_queue,
)
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

# ── Shared helpers ────────────────────────────────────────────────────────────

PROMPT_SR = 16000

def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device


# ── CosyVoice nodes ──────────────────────────────────────────────────────────

COSY_DEFAULT_MODEL_PATH = os.environ.get(
    "COSYVOICE_MODEL_PATH",
    os.path.join(os.path.dirname(current_dir), "models", "tts", "cosyvoice", "CosyVoice2-0.5B"),
)

class CosyVoiceModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": COSY_DEFAULT_MODEL_PATH}),
                "vllm": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "TTS/CosyVoice"
    TITLE = "CosyVoice2 Model Loader"

    def load_model(self, model_path, vllm):
        if vllm:
            model = CosyVoice2(model_path, load_jit=True, load_trt=True, load_vllm=True, fp16=False)
        else:
            model = CosyVoice2(model_path)
        return (model,)


class CosyVoiceInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TTS_MODEL",),
                "inference_type": (["zero_shot", "cross_lingual", "instruct"], {
                    "default": "zero_shot",
                    "description": "zero_shot/cross_lingual/instruct"
                }),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "prompt_audio": ("AUDIO",),
                "prompt_text": ("STRING", {
                    "default": "怎么会离开呢，嗯？是觉得姐姐这里吸引不了你吗，还是说有其他的人更吸引。你带我看看他是什么样子的，然后，我会让你下一次见不到他。",
                    "multiline": True,
                    "placeholder": "Enter prompt text",
                    "description": "Prompt text corresponding to prompt audio"
                }),
                "tts_text": ("STRING", {
                    "default": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗",
                    "multiline": True,
                    "placeholder": "Enter text to be synthesized",
                    "description": "Text to be synthesized (min_length: 1)"
                }),
            },
            "optional": {
                "instruct_text": ("STRING", {"default": "", "multiline": True, "placeholder": "Enter instruction text"}),
                "is_japanese": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "inference"
    CATEGORY = "TTS/CosyVoice"
    TITLE = "CosyVoice2 Inference"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return datetime.now().timestamp()

    def inference(self, model, inference_type, prompt_audio, prompt_text, tts_text, instruct_text, speed, is_japanese):
        if is_japanese:
            waveform = torch.zeros((1, 1, 1), dtype=torch.float32)
            return ({"waveform": waveform, "sample_rate": model.sample_rate},)

        waveform = prompt_audio["waveform"]
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        sample_rate = prompt_audio["sample_rate"]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            torchaudio.save(tmp_path, waveform, sample_rate)

            speeches = []
            if inference_type == 'zero_shot':
                for i, j in enumerate(model.inference_zero_shot(tts_text=tts_text, prompt_text=prompt_text, prompt_wav=tmp_path, stream=False, speed=speed)):
                    speeches.append(j['tts_speech'])
            elif inference_type == 'instruct':
                for i, j in enumerate(model.inference_instruct2(tts_text=tts_text, instruct_text=instruct_text, prompt_wav=tmp_path, stream=False, speed=speed)):
                    speeches.append(j['tts_speech'])
            elif inference_type == 'cross_lingual':
                for i, j in enumerate(model.inference_cross_lingual(tts_text=tts_text, prompt_wav=tmp_path, stream=False, speed=speed)):
                    speeches.append(j['tts_speech'])
            tts_speech = torch.cat(speeches, dim=1)
            tts_speech = tts_speech.unsqueeze(0)
            return ({"waveform": tts_speech, "sample_rate": model.sample_rate},)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# ── FishSpeech nodes ─────────────────────────────────────────────────────────

FISH_DEFAULT_LLAMA_PATH = os.environ.get(
    "FISH_LLAMA_PATH",
    os.path.join(os.path.dirname(current_dir), "models", "tts", "fish-speech", "fs-int8"),
)
FISH_DEFAULT_DECODER_PATH = os.environ.get(
    "FISH_DECODER_PATH",
    os.path.join(FISH_DEFAULT_LLAMA_PATH, "codec.pth"),
)

_FISH_MODEL_CACHE: dict[tuple, TTSInferenceEngine] = {}
_FISH_PATCHED_SEND_LLAMA = False


def _fish_unload_engine(engine: TTSInferenceEngine):
    engine.llama_queue.put(None)
    del engine.decoder_model
    del engine.llama_queue
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _fish_patch_send_llama_request_once():
    global _FISH_PATCHED_SEND_LLAMA
    if _FISH_PATCHED_SEND_LLAMA:
        return

    def _send_llama_request(self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list):
        request = dict(
            device=getattr(self, "llama_device", self.decoder_model.device),
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )
        response_queue = queue.Queue()
        self.llama_queue.put(GenerateRequest(request=request, response_queue=response_queue))
        return response_queue

    TTSInferenceEngine.send_Llama_request = _send_llama_request
    _FISH_PATCHED_SEND_LLAMA = True


def _audio_to_wav_bytes(audio: dict) -> bytes:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        wav = waveform[0]
    elif waveform.ndim == 2:
        wav = waveform
    else:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_np = wav.squeeze(0).detach().cpu().float().numpy()
    buf = io.BytesIO()
    sf.write(buf, wav_np, sample_rate, format="WAV")
    return buf.getvalue()


class FishSpeechModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama_checkpoint_path": ("STRING", {"default": FISH_DEFAULT_LLAMA_PATH}),
                "decoder_checkpoint_path": ("STRING", {"default": FISH_DEFAULT_DECODER_PATH}),
                "decoder_config_name": ("STRING", {"default": "modded_dac_vq"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "decoder_device": (["same_as_model", "cpu"], {"default": "same_as_model"}),
                "precision": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "compile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "TTS/FishSpeech"
    TITLE = "FishSpeech Model Loader"

    def load_model(
        self,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
        device: str,
        decoder_device: str,
        precision: str,
        compile: bool,
    ):
        resolved_device = _resolve_device(device)
        resolved_decoder_device = (
            resolved_device if decoder_device == "same_as_model" else decoder_device
        )
        torch_precision = torch.half if precision == "float16" else torch.bfloat16

        cache_key = (
            os.path.abspath(llama_checkpoint_path),
            os.path.abspath(decoder_checkpoint_path),
            decoder_config_name,
            resolved_device,
            resolved_decoder_device,
            str(torch_precision),
            bool(compile),
        )

        if cache_key in _FISH_MODEL_CACHE:
            return (_FISH_MODEL_CACHE[cache_key],)

        for old_key in list(_FISH_MODEL_CACHE.keys()):
            _fish_unload_engine(_FISH_MODEL_CACHE.pop(old_key))

        _fish_patch_send_llama_request_once()
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=resolved_device,
            precision=torch_precision,
            compile=compile,
        )
        decoder_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=decoder_checkpoint_path,
            device=resolved_decoder_device,
        )

        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=torch_precision,
            compile=compile,
        )
        engine.llama_device = resolved_device

        _original_decode_vq_tokens = engine.decode_vq_tokens

        def _decode_vq_tokens_with_device_sync(codes, _orig=_original_decode_vq_tokens):
            target_device = engine.decoder_model.device
            if isinstance(codes, torch.Tensor) and codes.device != target_device:
                codes = codes.to(target_device)
            if target_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(target_device)
                torch.cuda.empty_cache()
            try:
                return _orig(codes)
            except RuntimeError as e:
                msg = str(e)
                if target_device.type == "cuda" and "CUDNN_STATUS_NOT_INITIALIZED" in msg:
                    engine.decoder_model = engine.decoder_model.to("cpu")
                    if isinstance(codes, torch.Tensor):
                        codes = codes.to("cpu")
                    return _orig(codes)
                raise

        engine.decode_vq_tokens = _decode_vq_tokens_with_device_sync

        _FISH_MODEL_CACHE[cache_key] = engine
        return (engine,)


class FishSpeechTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，欢迎使用 Fish Speech。"}),
                "chunk_length": ("INT", {"default": 200, "min": 100, "max": 300, "step": 1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.9, "max": 2.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "is_japanese": ("BOOLEAN",),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "TTS/FishSpeech"
    TITLE = "FishSpeech TTS"

    def generate(
        self,
        model: TTSInferenceEngine,
        text: str,
        chunk_length: int,
        max_new_tokens: int,
        top_p: float,
        repetition_penalty: float,
        temperature: float,
        seed: int,
        reference_audio: Optional[dict] = None,
        reference_text: str = "",
        is_japanese: bool = True,
    ):
        if not is_japanese:
            waveform = torch.zeros((1, 1, 1), dtype=torch.float32)
            return ({"waveform": waveform, "sample_rate": 44100},)

        references = []
        if reference_audio is not None:
            if not reference_text.strip():
                raise ValueError("reference_audio provided but reference_text is empty")
            ref_wav_bytes = _audio_to_wav_bytes(reference_audio)
            references = [ServeReferenceAudio(audio=ref_wav_bytes, text=reference_text)]

        req = ServeTTSRequest(
            text=text,
            chunk_length=chunk_length,
            format="wav",
            references=references,
            reference_id=None,
            seed=None if seed < 0 else int(seed),
            use_memory_cache="off",
            streaming=False,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        final_audio = None
        sample_rate = None
        for result in model.inference(req):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final" and isinstance(result.audio, tuple):
                sample_rate, final_audio = result.audio

        if final_audio is None or sample_rate is None:
            raise RuntimeError("FishSpeech did not produce valid audio")

        wav_np = np.asarray(final_audio, dtype=np.float32)
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0)
        return ({"waveform": wav_tensor, "sample_rate": int(sample_rate)},)


# ── Utility node ──────────────────────────────────────────────────────────────

class ConditionalBranchAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"condition": ("BOOLEAN", {"default": False})},
            "optional": {
                "input_if_true": ("AUDIO",),
                "input_if_false": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "TTS/FishSpeech"

    def check_lazy_status(self, condition, input_if_true=None, input_if_false=None):
        return ["input_if_true"] if condition else ["input_if_false"]

    def execute(self, condition, input_if_true=None, input_if_false=None):
        return (input_if_true if condition else input_if_false,)


# ── Node registry ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CosyVoiceModelLoader": CosyVoiceModelLoader,
    "CosyVoiceInference": CosyVoiceInference,
    "FishSpeechModelLoader": FishSpeechModelLoader,
    "FishSpeechTTS": FishSpeechTTS,
    "ConditionalBranchAudio": ConditionalBranchAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FishSpeechModelLoader": "FishSpeech Model Loader",
    "FishSpeechTTS": "FishSpeech TTS",
    "ConditionalBranchAudio": "Conditional Branch Audio (Lazy)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]