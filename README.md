# droid2api

OpenAI 兼容 API 代理服务器，用于在不同 LLM API 格式之间进行转换。

## 功能特性

- **标准 OpenAI API 接口**：提供完全兼容 OpenAI 的 API 端点
- **多格式支持**：支持 Anthropic 和自定义 OpenAI 格式之间的自动转换
- **流式响应**：自动转换 SSE (Server-Sent Events) 流式响应为标准 OpenAI 格式
- **自动刷新 API Key**：集成 WorkOS 认证，自动管理和刷新访问令牌（8小时有效期，每6小时自动刷新）
- **智能 Header 管理**：自动添加和管理所有必需的 Factory 特定 headers
- **配置化路由**：通过 config.json 灵活配置模型和端点映射
- **开发模式**：详细的日志输出，便于调试

## 安装

```bash
npm install
```

## 配置

### 1. 配置端点和模型

编辑 `config.json` 文件：

```json
{
  "port": 3000,
  "endpoint": [
    {
      "name": "openai",
      "base_url": "https://app.factory.ai/api/llm/o/v1/responses"
    },
    {
      "name": "anthropic",
      "base_url": "https://app.factory.ai/api/llm/a/v1/messages"
    }
  ],
  "models": [
    {
      "name": "Claude Opus 4",
      "id": "claude-opus-4-1-20250805",
      "type": "anthropic"
    },
    {
      "name": "GPT-5 Codex",
      "id": "gpt-5-codex",
      "type": "openai"
    }
  ],
  "dev_mode": false
}
```

### 2. 配置认证（二选一）

#### 方式一：使用环境变量（推荐用于开发/测试）

```bash
export DROID_REFRESH_KEY="your_refresh_token_here"
```

刷新后的 API key 会保存到工作目录的 `auth.json` 文件。

#### 方式二：使用配置文件（推荐用于生产环境）

确保 `~/.factory/auth.json` 文件存在并包含有效的 tokens：

```json
{
  "access_token": "your_access_token_here",
  "refresh_token": "your_refresh_token_here"
}
```

刷新后的 tokens 会自动更新到原文件。

## 使用方法

### 启动服务器

```bash
npm start
```

或使用快捷脚本：

```bash
./start.sh
```

服务器默认运行在 `http://localhost:3000`。

### API 端点

#### 1. 获取可用模型列表

```bash
GET /v1/models
```

**示例：**
```bash
curl http://localhost:3000/v1/models
```

**响应：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-opus-4-1-20250805",
      "object": "model",
      "created": 1704067200000,
      "owned_by": "factory"
    }
  ]
}
```

#### 2. 对话补全（兼容 OpenAI）

```bash
POST /v1/chat/completions
```

**请求参数：**
- `model` (必需): 模型 ID
- `messages` (必需): 对话消息数组
- `stream` (可选): 是否使用流式响应，默认 true
- `max_tokens` (可选): 最大输出 tokens 数
- `temperature` (可选): 温度参数 0-1
- `top_p` (可选): Top-p 采样参数

**示例（流式响应）：**
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "stream": true,
    "max_tokens": 2000
  }'
```

**示例（非流式响应）：**
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-codex",
    "messages": [
      {"role": "user", "content": "写一个 Python 快速排序"}
    ],
    "stream": false
  }'
```

## API Key 自动刷新机制

代理服务器会自动管理 API key 的刷新：

1. **启动时刷新**：服务器启动时自动获取新的 access token
2. **定期刷新**：每次 API 请求前检查，如果距离上次刷新超过 6 小时则自动刷新
3. **令牌有效期**：access token 有效期为 8 小时
4. **自动保存**：刷新后的 tokens 自动保存到相应的配置文件

**刷新日志示例：**
```
[INFO] Refreshing API key...
[INFO] Authenticated as: user@example.com (John Doe)
[INFO] User ID: user_01K69S755R2TWYFWKPSP74TRKZ
[INFO] Organization ID: org_01K69S7KKYK6F2WYJ8CB384GW6
[INFO] API key refreshed successfully
```

## 格式转换说明

### Anthropic 格式转换

**请求转换：**
- `messages` → `messages`（提取 system 消息到顶层）
- `max_tokens` → `max_tokens`（默认 4096）
- 文本内容包装为 `{type: 'text', text: '...'}`
- 工具格式转换

**响应转换：**
- 转换 SSE 事件：`message_start`, `content_block_delta`, `message_delta`, `message_stop`
- 转换为标准 OpenAI chunk 格式
- 映射停止原因：`end_turn` → `stop`, `max_tokens` → `length`

### OpenAI 格式转换

**请求转换：**
- `messages` → `input`
- `max_tokens` → `max_output_tokens`
- 用户消息：`text` → `input_text`
- 助手消息：`text` → `output_text`
- 提取 system 消息为 `instructions` 参数

**响应转换：**
- 转换 SSE 事件：`response.created`, `response.in_progress`, `response.done`
- 转换为标准 OpenAI chunk 格式

## Header 管理

代理服务器会自动添加所有必需的 headers：

### Anthropic 端点
- `x-model-provider: anthropic`
- `x-factory-client: cli`
- `user-agent: a$/JS 0.57.0`
- `anthropic-version: 2023-06-01`
- `anthropic-beta: interleaved-thinking-2025-05-14`
- `x-stainless-helper-method: stream`（流式请求）
- 自动生成的 UUID：`x-session-id`, `x-assistant-message-id`

### OpenAI 端点
- `x-factory-client: cli`
- `user-agent: cB/JS 5.22.0`
- 自动生成的 UUID：`x-session-id`, `x-assistant-message-id`

## 开发模式

在 `config.json` 中设置 `dev_mode: true` 可以启用详细日志：

```json
{
  "dev_mode": true
}
```

**日志内容包括：**
- 完整的请求和响应 headers
- 请求体和响应体
- 格式转换过程
- SSE 事件处理详情

## 端口冲突处理

如果端口 3000 已被占用，可以：

1. **修改配置文件**：编辑 `config.json` 中的 `port` 字段
2. **或者结束占用进程**：
   ```bash
   lsof -ti:3000 | xargs kill -9
   ```

## 故障排查

### 启动时报错 "Refresh token not found"

**原因**：未配置 refresh token

**解决方案**：
- 设置环境变量 `DROID_REFRESH_KEY`
- 或配置 `~/.factory/auth.json` 文件

### 请求返回 401 错误

**可能原因**：
1. refresh token 已过期或无效
2. API key 刷新失败

**解决方案**：
- 检查日志中的刷新错误信息
- 重新获取有效的 refresh token
- 确认 `~/.factory/auth.json` 中的 tokens 正确

### 响应格式错误

**原因**：模型类型配置错误

**解决方案**：
- 检查 `config.json` 中模型的 `type` 字段
- Anthropic 模型使用 `"type": "anthropic"`
- OpenAI 模型使用 `"type": "openai"`

## 技术架构

- **语言**：Node.js (ES Modules)
- **框架**：Express
- **HTTP 客户端**：node-fetch
- **认证**：WorkOS OAuth 2.0 Refresh Token Flow

## 许可证

MIT
