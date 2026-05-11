# comfyui-tts-pack

ComfyUI 自定义节点，集成 **CosyVoice 2** 和 **FishSpeech** 两种 TTS 引擎。

## 安装

```bash
# 1. 进入 ComfyUI custom_nodes 目录
cd ComfyUI/custom_nodes/

# 2. 克隆仓库（包含子模块）
git clone --recursive https://github.com/your-org/comfyui-tts-pack.git

# 3. 安装依赖
pip install -r comfyui-tts-pack/requirements.txt
```

如果已经 clone 了但忘了 `--recursive`：

```bash
cd ComfyUI/custom_nodes/comfyui-tts-pack
git submodule update --init --recursive
```

## 下载模型

### CosyVoice 2

```bash
# 方式一：从 ModelScope 下载
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='ComfyUI/models/tts/cosyvoice/CosyVoice2-0.5B')"

# 方式二：从 HuggingFace 下载
pip install huggingface-hub
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir ComfyUI/models/tts/cosyvoice/CosyVoice2-0.5B
```

### FishSpeech

从 HuggingFace 下载：

```bash
huggingface-cli download fishaudio/fish-speech-1.5-sft --local-dir ComfyUI/models/tts/fish-speech/fs-int8
```

## 模型路径配置

默认路径为 `ComfyUI/models/tts/<engine>/...`，可通过环境变量覆盖：

| 环境变量 | 作用 |
|---|---|
| `COSYVOICE_MODEL_PATH` | CosyVoice 模型路径 |
| `FISH_LLAMA_PATH` | FishSpeech LLAMA 检查点路径 |
| `FISH_DECODER_PATH` | FishSpeech 解码器路径 |

也可以在 ComfyUI 节点的输入框中手动修改路径。

## 节点说明

| 节点名 | 分类 | 功能 |
|---|---|---|
| CosyVoice2 Model Loader | TTS/CosyVoice | 加载 CosyVoice2 模型（可选 vLLM 加速） |
| CosyVoice2 Inference | TTS/CosyVoice | 三种推理模式：zero_shot / cross_lingual / instruct |
| FishSpeech Model Loader | TTS/FishSpeech | 加载 FishSpeech 模型 |
| FishSpeech TTS | TTS/FishSpeech | 文本转语音（支持参考音频） |
| Conditional Branch Audio (Lazy) | TTS/FishSpeech | 懒执行音频分支选择 |

## 子模块说明

本仓库使用 git submodule 管理 TTS 引擎源码：

- `cosyvoice/` → [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- `fishspeech/` → [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

更新子模块到最新版：

```bash
git submodule update --remote --merge
```

## 注意事项

- vLLM 加速为可选项，CosyVoice 不使用 vLLM 也能跑
- FishSpeech 的 `descript-audio-codec` 可能需要单独安装
- 如果遇到 `CUDNN_STATUS_NOT_INITIALIZED` 错误，FishSpeech 会自动回退到 CPU 解码