# comfyui-tts-pack

ComfyUI 自定义节点，集成 **CosyVoice 2** 和 **FishSpeech** 两种 TTS 引擎。

## 安装

```bash
# 1. 进入 ComfyUI custom_nodes 目录
cd ComfyUI/custom_nodes/

# 2. 克隆仓库（包含子模块）
git clone --recursive https://github.com/your-org/comfyui-tts-pack.git

# 3. 创建python3.12的conda环境
conda create -n comfy-tts-pack python=3.12
conda activate comfy-tts-pack

# 4. 安装依赖
cd comfyui-tts-pack
pip install -r requirements.txt
```

如果已经 clone 了但忘了 `--recursive`：

```bash
cd ComfyUI/custom_nodes/comfyui-tts-pack
git submodule update --init --recursive
```

## 本地模型路径

### CosyVoice

```bash
/mist/dengliang/cosy/pretrained_models/CosyVoice2-0.5B
```

### FishSpeech

```bash
/mist/dengliang/fish-speech/checkpoints/fs-int8-20260427_182050
```

## 模型路径配置

默认路径为 `ComfyUI/models/tts/<engine>/...`，可通过环境变量覆盖：

| 环境变量 | 作用 |
|---|---|
| `COSYVOICE_MODEL_PATH` | CosyVoice 模型路径 |
| `FISH_LLAMA_PATH` | FishSpeech LLAMA 检查点路径 |

也可以在 ComfyUI 节点的输入框中手动修改路径。

## 节点说明

| 节点名 | 分类 | 功能 |
|---|---|---|
| CosyVoice2 Model Loader | TTS/CosyVoice | 加载 CosyVoice2 模型（可选 vLLM 加速） |
| CosyVoice2 Inference | TTS/CosyVoice | 三种推理模式：zero_shot / cross_lingual / instruct |
| FishSpeech Model Loader | TTS/FishSpeech | 加载 FishSpeech 模型 |
| FishSpeech TTS | TTS/FishSpeech | 文本转语音（支持参考音频） |
| Conditional Branch Audio (Lazy) | TTS/FishSpeech | 懒执行音频分支选择 |

## 示例工作流

`tts-pack.json` 是一个完整的 ComfyUI 工作流，展示了 CosyVoice 和 FishSpeech 双引擎的联合使用：

- **共享输入**：通过 `LoadAudio` 加载同一份参考音频，使用 `PrimitiveStringMultiline` 输入 prompt 文本和 TTS 文本
- **CosyVoice 链路**：`CosyVoiceModelLoader` → `CosyVoiceInference`（zero_shot 模式）
- **FishSpeech 链路**：`FishSpeechModelLoader` → `FishSpeechTTS`
- **音频路由**：`ConditionalBranchAudio` 根据条件选择输出源（方便对比两个引擎的效果）
- **输出**：同时接入 `SaveAudioOpus` 和 `SaveAudioMP3` 两种格式

使用前需将工作流中 `CosyVoiceModelLoader.model_path` 和 `FishSpeechModelLoader.llama_checkpoint_path` 修改为本地实际路径。

## 子模块说明

本仓库使用 git submodule 管理 TTS 引擎源码：

- `cosyvoice/` → [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- `fishspeech/` → [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

更新子模块到最新版：

```bash
git submodule update --remote --merge
```
