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

### 方式一：extra_model_paths.yaml（推荐）

在 ComfyUI 根目录的 `extra_model_paths.yaml`（没有则新建）中添加路径：

```yaml
shared:
  base_path: /path/to/your/models
  cosyvoice: models/tts/cosyvoice/
  fishspeech: models/tts/fishspeech/
```

`cosyvoice` 和 `fishspeech` 指向的目录下按模型版本创建子目录，例如：

```
/path/to/your/models/
└── models/tts/
    ├── cosyvoice/
    │   └── CosyVoice2-0.5B/
    ├── fishspeech/
    │   └── fs-int8-20260427_182050/
```

配置完成后重启 ComfyUI，节点输入框会自动填入对应路径。

### 方式二：直接在节点中输入

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
- **输出**：`SaveAudioAdvanced` 可选 Opus / MP3 / FLAC 格式

使用前需确保模型路径已正确配置。如果已配置 `extra_model_paths.yaml`，节点会自动填入路径；否则需在节点输入框中手动填写本地实际路径。

## 子模块说明

本仓库使用 git submodule 管理 TTS 引擎源码：

- `cosyvoice/` → [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- `fishspeech/` → [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

更新子模块到最新版：

```bash
git submodule update --remote --merge
```
