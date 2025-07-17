ai

# 🗣️ ASR Maid Python

> 轻量级、线程安全的 HTTP 服务，用于语音识别（ASR），基于 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)。

## 📌 项目简介

这是一个使用 Python 编写的轻量级 HTTP 服务器，利用 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) 实现离线语音识别功能。支持 IPv4 / IPv6 / Dual Stack，支持多线程处理请求，适用于本地部署或嵌入式设备。

你可以通过 HTTP POST 请求上传 WAV 文件（16-bit PCM mono @ 16 kHz），服务将返回转写文本。

---

## 🚀 快速开始

### ✅ 安装依赖

```bash
pip install sherpa-onnx numpy
```

### 📁 模型目录结构

将 `model.onnx` 和 `tokens.txt` 放置在以下目录中：

```
assets/sensevoicesmallonnx/
├── model.onnx
└── tokens.txt
```

你也可以修改源码中的 `MODEL_DIR` 变量来自定义模型路径。

### ▶️ 启动服务器

```bash
python3 asr_server.py
```

默认监听端口为 `7860`，支持 IPv4 和 IPv6。

---

## 📡 API 说明

### `GET /`

检查服务状态：

```json
{
  "status": "ok",
  "model_loaded": true,
  "usage": "POST a 16 kHz / 16-bit / mono WAV file to /asr"
}
```

### `POST /asr`

上传一个 `WAV` 文件进行语音识别。

#### 请求

- 内容类型：`audio/wav`
- 格式要求：16 kHz, 16-bit, 单声道 (mono)

#### 响应示例

```json
{
  "status": "success",
  "result": "你好，世界"
}
```

---

## ⚙️ 启动参数

| 参数             | 描述                                             | 示例                         |
|------------------|--------------------------------------------------|------------------------------|
| `--host`         | 指定绑定地址（默认根据 `--scope` 自动设置）     | `--host 0.0.0.0`             |
| `--port`         | 自定义端口号                                     | `--port 8000`                |
| `--ip-version`   | IPv4 / IPv6 / Dual 模式                          | `--ip-version dual`          |
| `--scope`        | `local` 仅监听本地回环，`all` 监听所有接口地址   | `--scope all`                |

---

## 🧠 技术细节

- 音频解析：使用标准库 `wave` 模块解析 PCM WAV 文件
- 推理模型：基于 `sherpa-onnx` 的 `OfflineRecognizer`
- 支持双协议栈：IPv4 / IPv6 同时监听
- 使用 `ThreadingHTTPServer` 实现多线程并发请求处理

---

## 🧪 示例请求（使用 `curl`）

```bash
curl -X POST http://localhost:7860/asr \
     -H "Content-Type: audio/wav" \
     --data-binary @example.wav
```

---

## 🛠️ 开发与调试

内网穿透(frp,ngrok)、异地组网(tailscale,zerotier)、docker(抱脸网)，都能免费部署自用。

你可以设置日志级别以获得更详细的运行信息：

```bash
export LOGLEVEL=DEBUG
python3 asr_server.py
```

---

## 📄 License

MIT License
