# comfyui-tts-pack

ComfyUI 自定义节点，集成 **CosyVoice 2** 和 **FishSpeech** 两种 TTS 引擎。

## 安装

```bash
# 1. 进入 ComfyUI custom_nodes 目录
cd ComfyUI/custom_nodes/

# 2. 克隆仓库（包含子模块）
git clone --recursive https://github.com/Dlight160/comfyui-tts-pack.git

# 3. 创建 python3.12 的 conda 环境
conda create -n comfy-tts-pack python=3.12
conda activate comfy-tts-pack

# 4. 先安装 ComfyUI 自身依赖（避免版本冲突）
cd ../../
pip install -r requirements.txt
cd custom_nodes/comfyui-tts-pack

# 5. 再安装 TTS Pack 依赖
pip install -r requirements.txt
```

如果已经 clone 了但忘了 `--recursive`：

```bash
cd ComfyUI/custom_nodes/comfyui-tts-pack
git submodule update --init --recursive
```

## 模型路径配置

模型路径可以通过以下两种方式传入节点：

### 方式一：使用默认路径（开箱即用）

节点输入框已预填默认路径，无需额外配置即可使用。

### 方式二：自定义路径

在节点的 `model_path`（或 `llama_checkpoint_path`）输入框中填写路径，支持两种格式：

- **绝对路径** — 直接使用该路径加载模型  
  例如：`/data/models/CosyVoice2-0.5B`
- **相对路径** — 从以下搜索目录中查找：
  1. `ComfyUI/models/tts/`
  2. `extra_model_paths.yaml` 中注册的 `tts` 路径（如有配置）

  例如：`CosyVoice/CosyVoice2-0.5B` 会依次检查 `models/tts/CosyVoice/CosyVoice2-0.5B` 等路径。

留空则使用代码中预设的默认路径。

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
- **输出**：`SaveAudioAdvanced` 可选 Opus / MP3 / FLAC 格式

使用前需确保模型路径正确。节点输入框已预填默认路径，如需使用其他路径，可直接修改输入框（支持绝对路径或相对路径）。

## 子模块说明

本仓库使用 git submodule 管理 TTS 引擎源码：

- `cosyvoice/` → [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- `fishspeech/` → [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

更新子模块到最新版：

```bash
git submodule update --remote --merge
```
