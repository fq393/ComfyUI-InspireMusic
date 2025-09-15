# ComfyUI InspireMusic Plugin

**InspireMusic ComfyUI Plugin** - AI音乐生成的ComfyUI集成插件

基于阿里巴巴InspireMusic模型的ComfyUI节点插件，支持文本到音乐生成和音乐续写功能。

## 特性
- 🎵 文本到音乐生成 (Text-to-Music)
- 🎼 音乐续写 (Music Continuation) 
- 🔧 模块化设计，易于维护和扩展
- ⚡ 支持快速模式和高质量模式
- 🎛️ 丰富的音频处理选项

## 项目结构
```
ComfyUI-InspireMusic/
├── __init__.py              # 插件入口文件
├── nodes/                   # ComfyUI节点
│   ├── __init__.py
│   └── inspiremusic_node.py # InspireMusic节点实现
├── modules/                 # 工具模块
│   ├── __init__.py
│   ├── audio_utils.py       # 音频处理工具
│   └── model_manager.py     # 模型管理器
├── inspiremusic/           # InspireMusic核心库
│   ├── cli/                 # 命令行接口
│   ├── dataset/             # 数据集处理
│   ├── flow/                # 流匹配模块
│   ├── hifigan/             # HiFiGAN声码器
│   ├── llm/                 # 大语言模型
│   ├── metrics/             # 评估指标
│   ├── music_tokenizer/     # 音乐分词器
│   ├── text/                # 文本处理
│   ├── transformer/         # Transformer模块
│   ├── utils/               # 工具函数
│   └── wavtokenizer/        # 波形分词器
├── example/                 # 配置示例
└── requirements.txt        # 依赖列表
```

## 安装

### 环境要求
- Python >= 3.8
- PyTorch >= 2.0.1
- ComfyUI
- CUDA >= 11.8 (可选，用于GPU加速)

### 安装步骤

1. **克隆插件到ComfyUI目录**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI-InspireMusic.git
cd ComfyUI-InspireMusic
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

   > **注意**: 依赖包含了 `matcha-tts`，这是解决 Matcha-TTS 模块导入问题的关键依赖。

3. **下载模型**
```bash
# 创建ComfyUI模型目录
mkdir -p ComfyUI/models/InspireMusic

# 从ModelScope下载
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long

# 或从HuggingFace下载
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long
```

4. **重启ComfyUI**
重启ComfyUI后，在节点列表中找到"InspireMusic"分类。

## 使用示例

### 文本到音乐生成
```
输入文本: "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance."
输出: 30秒的爵士乐音频文件
```

### 音乐续写
```
输入: 音频文件 + 文本描述
输出: 续写的音乐片段
```

## 快速开始

### 在ComfyUI中使用

1. **启动ComfyUI**
2. **添加InspireMusic节点**
   - 在节点菜单中找到 `InspireMusic` → `InspireMusic Text To Music`
3. **配置节点参数**
   - `text_prompt`: 输入音乐描述文本
   - `model_name`: 选择模型 (如 "InspireMusic-1.5B-Long")
   - `duration`: 设置生成时长 (1-180秒)
   - `fast_mode`: 选择快速模式或高质量模式
4. **连接输出**
   - 将音频输出连接到音频预览或保存节点
5. **执行工作流**

### 节点参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text_prompt | STRING | "" | 音乐描述文本 |
| model_name | COMBO | "InspireMusic-1.5B-Long" | 模型名称 |
| duration | FLOAT | 30.0 | 生成时长(秒) |
| fast_mode | BOOLEAN | False | 快速模式 |
| apply_fade_out | BOOLEAN | True | 应用淡出效果 |
| trim_silence | BOOLEAN | True | 修剪静音 |
| output_sample_rate | INT | 48000 | 输出采样率 |

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否正确下载
   - 确认模型路径配置正确
   - 检查磁盘空间是否充足

2. **内存不足**
   - 尝试使用快速模式 (`fast_mode=True`)
   - 减少生成时长
   - 关闭其他占用内存的程序

3. **生成速度慢**
   - 确保使用GPU加速
   - 启用快速模式
   - 检查CUDA版本兼容性
## 支持的模型

插件支持以下InspireMusic预训练模型：

| 模型名称 | 采样率 | 时长 | 说明 |
|---------|--------|------|------|
| InspireMusic-Base-24kHz | 24kHz | 30s | 基础模型，单声道 |
| InspireMusic-Base | 48kHz | 30s | 基础模型，立体声 |
| InspireMusic-1.5B-24kHz | 24kHz | 30s | 1.5B参数模型，单声道 |
| InspireMusic-1.5B | 48kHz | 30s | 1.5B参数模型，立体声 |
| InspireMusic-1.5B-Long | 48kHz | 数分钟 | 1.5B参数长音频模型 |

### 模型下载

**ModelScope (推荐)**
```bash
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long
```

**HuggingFace**
```bash
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long
```

## 开发计划

- ✅ 基础文本到音乐生成功能
- ✅ 模块化架构设计
- ✅ 多模型支持
- 🔄 音乐续写功能 (开发中)
- 📋 批量生成功能
- 📋 更多音频格式支持
- 📋 自定义模型路径配置

## 贡献

欢迎提交Issue和Pull Request来改进这个插件！

## 许可证

本项目基于原始InspireMusic项目的许可证。请查看LICENSE.txt文件了解详情。

## 致谢

本插件基于阿里巴巴的InspireMusic项目开发。感谢原作者团队的杰出工作。

### 原始论文引用
```bibtex
@inproceedings{InspireMusic2025,
      title={InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation}, 
      author={Chong Zhang and Yukun Ma and Qian Chen and Wen Wang and Shengkui Zhao and Zexu Pan and Hao Wang and Chongjia Ni and Trung Hieu Nguyen and Kun Zhou and Yidi Jiang and Chaohong Tan and Zhifu Gao and Zhihao Du and Bin Ma},
      year={2025},
      eprint={2503.00084},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.00084}
}
```

### 相关链接
- [InspireMusic 原始项目](https://github.com/FunAudioLLM/InspireMusic)
- [InspireMusic 论文](http://arxiv.org/abs/2503.00084)
- [ComfyUI 官方网站](https://github.com/comfyanonymous/ComfyUI)

---

**免责声明**: 本内容仅供研究目的使用。
