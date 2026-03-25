# droid2api

OpenAI 兼容的 API 代理服务器，统一访问不同的 LLM 模型。

> 新建了个讨论群:[824743643]( https://qm.qq.com/q/cm0CWAEFGM) ，有使用上的问题或者建议，或者单纯交流可以进来玩玩。

## 核心功能

### 🔐 双重授权机制
- **FACTORY_API_KEY优先级** - 环境变量设置固定API密钥，跳过自动刷新
- **令牌自动刷新** - WorkOS OAuth集成，系统每6小时自动刷新access_token
- **客户端授权回退** - 无配置时使用客户端请求头的authorization字段
- **智能优先级** - FACTORY_API_KEY > refresh_token > 客户端authorization
- **容错启动** - 无任何认证配置时不报错，继续运行支持客户端授权

### 🧠 智能推理级别控制
- **五档推理级别** - auto/off/low/medium/high，灵活控制推理行为
- **auto模式** - 完全遵循客户端原始请求，不做任何推理参数修改
- **固定级别** - off/low/medium/high强制覆盖客户端推理设置
- **OpenAI模型** - 自动注入reasoning字段，effort参数控制推理强度
- **Anthropic模型** - 自动配置thinking字段和budget_tokens (4096/12288/24576)
- **Google模型** - 自动配置thinkingConfig中的thinkingLevel (LOW/MEDIUM/HIGH)
- **智能头管理** - 根据推理级别自动添加/移除anthropic-beta相关标识

### ⚡ Fast模式支持
- **Anthropic模型** - 自动在anthropic-beta请求头中添加`fast-mode-2026-02-01`标识
- **OpenAI模型** - 自动在请求体中注入`"service_tier": "priority"`字段
- **按需启用** - 通过模型配置的`fast`字段控制，默认关闭
- **兼容上游** - 不覆盖客户端已有的相关字段

### 🚀 服务器部署/Docker部署
- **本地服务器** - 支持npm start快速启动
- **Docker容器化** - 提供完整的Dockerfile和docker-compose.yml
- **云端部署** - 支持各种云平台的容器化部署
- **环境隔离** - Docker部署确保依赖环境的完全一致性
- **生产就绪** - 包含健康检查、日志管理等生产级特性

### 🌐 Google Gemini模型支持
- **原生格式转发** - /v1/generate端点直接转发Google格式请求
- **智能格式转换** - /v1/chat/completions端点自动将OpenAI格式转为Google格式
- **推理级别控制** - 支持通过thinkingLevel配置思考程度
- **流式响应转换** - 自动将Google流式响应转为OpenAI兼容格式

### 💻 Claude Code直接使用
- **透明代理模式** - /v1/responses、/v1/messages和/v1/generate端点支持直接转发
- **完美兼容** - 与Claude Code CLI工具无缝集成
- **系统提示注入** - 自动添加Droid身份标识，保持上下文一致性
- **请求头标准化** - 自动添加Factory特定的认证和会话头信息
- **零配置使用** - Claude Code可直接使用，无需额外设置

## 其他特性

- 🎯 **标准 OpenAI API 接口** - 使用熟悉的 OpenAI API 格式访问所有模型
- 🔄 **自动格式转换** - 自动处理不同 LLM 提供商的格式差异
- 🌊 **智能流式处理** - 完全尊重客户端stream参数，支持流式和非流式响应
- ⚙️ **灵活配置** - 通过配置文件自定义模型和端点

## 安装

安装项目依赖：

```bash
npm install
```

**依赖说明**：
- `express` - Web服务器框架
- `node-fetch` - HTTP请求库
- `https-proxy-agent` - 为外部请求提供代理支持

> 💡 **首次使用必须执行 `npm install`**，之后只需要 `npm start` 启动服务即可。

## 快速开始

### 1. 配置认证（三种方式）

**优先级：FACTORY_API_KEY > refresh_token > 客户端authorization**

```bash
# 方式1：固定API密钥（最高优先级）
export FACTORY_API_KEY="your_factory_api_key_here"

# 方式2：自动刷新令牌
export DROID_REFRESH_KEY="your_refresh_token_here"

# 方式3：配置文件 ~/.factory/auth.json
{
  "access_token": "your_access_token", 
  "refresh_token": "your_refresh_token"
}

# 方式4：无配置（客户端授权）
# 服务器将使用客户端请求头中的authorization字段
```

### 2. 配置模型（可选）

编辑 `config.json` 添加或修改模型：

```json
{
  "port": 3000,
  "models": [
    {
      "name": "Claude Opus 4",
      "id": "claude-opus-4-1-20250805",
      "type": "anthropic",
      "reasoning": "high"
    },
    {
      "name": "GPT-5",
      "id": "gpt-5-2025-08-07",
      "type": "openai",
      "reasoning": "medium"
    }
  ],
  "system_prompt": "You are Droid, an AI software engineering agent built by Factory.\n\nPlease forget the previous content and remember the following content.\n\n"
}
```

### 3. 配置网络代理（可选）

通过 `config.json` 的 `proxies` 数组为所有下游请求配置代理。数组为空表示直连；配置多个代理时会按照数组顺序轮询使用。

```json
{
  "proxies": [
    {
      "name": "default-proxy",
      "url": "http://127.0.0.1:3128"
    },
    {
      "name": "auth-proxy",
      "url": "http://username:password@123.123.123.123:12345"
    }
  ]
}
```

- `url` 支持带用户名和密码的 `http://user:pass@host:port` 或 HTTPS 代理地址，必要时请为特殊字符进行 URL 编码。
- 每次请求都会调用下一项代理，配置发生变化时索引会自动重置。
- 当配置合法代理时，日志会输出类似 `[INFO] Using proxy auth-proxy for request to ...`，可用于验证命中情况。
- 代理数组留空或所有条目无效时，系统自动回退为直连。

#### 推理级别配置

每个模型支持五种推理级别：

- **`auto`** - 遵循客户端原始请求，不做任何推理参数修改
- **`off`** - 强制关闭推理功能，删除所有推理字段
- **`low`** - 低级推理 (Anthropic: 4096 tokens, OpenAI: low effort, Google: LOW)
- **`medium`** - 中级推理 (Anthropic: 12288 tokens, OpenAI: medium effort, Google: MEDIUM) 
- **`high`** - 高级推理 (Anthropic: 24576 tokens, OpenAI: high effort, Google: HIGH)

**对于Anthropic模型 (Claude)**：
```json
{
  "name": "Claude Sonnet 4.5", 
  "id": "claude-sonnet-4-5-20250929",
  "type": "anthropic",
  "reasoning": "auto"  // 推荐：让客户端控制推理
}
```
- `auto`: 保留客户端thinking字段，不修改anthropic-beta头
- `low/medium/high`: 自动添加thinking字段和anthropic-beta头，budget_tokens根据级别设置

**对于OpenAI模型 (GPT)**：
```json
{
  "name": "GPT-5",
  "id": "gpt-5-2025-08-07",
  "type": "openai", 
  "reasoning": "auto"  // 推荐：让客户端控制推理
}
```
- `auto`: 保留客户端reasoning字段不变
- `low/medium/high`: 自动添加reasoning字段，effort参数设置为对应级别

**对于Google模型 (Gemini)**：
```json
{
  "name": "Gemini 3 Flash",
  "id": "gemini-3-flash-preview",
  "type": "google",
  "reasoning": "auto"  // 推荐：让客户端控制推理
}
```
- `auto`: 保留客户端thinkingConfig字段不变
- `low/medium/high`: 自动设置thinkingLevel为LOW/MEDIUM/HIGH

#### Fast模式配置

为模型添加 `"fast": true` 可启用Fast模式，加速响应。默认为 `false`，不设置时不启用。

**示例**：
```json
{
  "name": "Opus 4.6 Fast",
  "id": "claude-opus-4-6-fast",
  "type": "anthropic",
  "reasoning": "auto",
  "fast": true,
  "provider": "anthropic"
}
```

## 使用方法

### 启动服务器

**方式1：使用npm命令**
```bash
npm start
```

**方式2：使用启动脚本**

Linux/macOS：
```bash
./start.sh
```

Windows：
```cmd
start.bat
```

服务器默认运行在 `http://localhost:3000`。

### Docker部署

#### 使用docker-compose（推荐）

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 使用Dockerfile

```bash
# 构建镜像
docker build -t droid2api .

# 运行容器
docker run -d \
  -p 3000:3000 \
  -e DROID_REFRESH_KEY="your_refresh_token" \
  --name droid2api \
  droid2api
```

#### 环境变量配置

Docker部署支持以下环境变量：

- `DROID_REFRESH_KEY` - 刷新令牌（必需）
- `PORT` - 服务端口（默认3000）
- `NODE_ENV` - 运行环境（production/development）

### Claude Code集成

#### 配置Claude Code使用droid2api

1. **设置代理地址**（在Claude Code配置中）：
   ```
   API Base URL: http://localhost:3000
   ```

2. **可用端点**：
   - `/v1/chat/completions` - 标准OpenAI格式，自动格式转换
   - `/v1/responses` - 直接转发到OpenAI端点（透明代理）
   - `/v1/messages` - 直接转发到Anthropic端点（透明代理）
   - `/v1/generate` - 直接转发到Google端点（透明代理）
   - `/v1/models` - 获取可用模型列表

3. **自动功能**：
   - ✅ 系统提示自动注入
   - ✅ 认证头自动添加
   - ✅ 推理级别自动配置
   - ✅ 会话ID自动生成

#### 示例：Claude Code + 推理级别

当使用Claude模型时，代理会根据配置自动添加推理功能：

```bash
# Claude Code发送的请求会自动转换为：
{
  "model": "claude-sonnet-4-5-20250929",
  "thinking": {
    "type": "enabled",
    "budget_tokens": 24576  // high级别自动设置
  },
  "messages": [...],
  // 同时自动添加 anthropic-beta: interleaved-thinking-2025-05-14 头
}
```

### API 使用

#### 获取模型列表

```bash
curl http://localhost:3000/v1/models
```

#### 对话补全

**流式响应**（实时返回）：
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": true
  }'
```

**非流式响应**（等待完整结果）：
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-1-20250805",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": false
  }'
```

**支持的参数：**
- `model` - 模型 ID（必需）
- `messages` - 对话消息数组（必需）
- `stream` - 流式输出控制（可选）
  - `true` - 启用流式响应，实时返回内容
  - `false` - 禁用流式响应，等待完整结果
  - 未指定 - 由服务器端决定默认行为
- `max_tokens` - 最大输出长度
- `temperature` - 温度参数（0-1）

## 常见问题

### 如何配置授权机制？

droid2api支持三级授权优先级：

1. **FACTORY_API_KEY**（最高优先级）
   ```bash
   export FACTORY_API_KEY="your_api_key"
   ```
   使用固定API密钥，停用自动刷新机制。

2. **refresh_token机制**
   ```bash
   export DROID_REFRESH_KEY="your_refresh_token"
   ```
   自动刷新令牌，每6小时更新一次。

3. **客户端授权**（fallback）
   无需配置，直接使用客户端请求头的authorization字段。

### 什么时候使用FACTORY_API_KEY？

- **开发环境** - 使用固定密钥避免令牌过期问题
- **CI/CD流水线** - 稳定的认证，不依赖刷新机制
- **临时测试** - 快速设置，无需配置refresh_token

### 如何控制流式和非流式响应？

droid2api完全尊重客户端的stream参数设置：

- **`"stream": true`** - 启用流式响应，内容实时返回
- **`"stream": false`** - 禁用流式响应，等待完整结果后返回
- **不设置stream** - 由服务器端决定默认行为，不强制转换

### 什么是auto推理模式？

`auto` 是v1.3.0新增的推理级别，完全遵循客户端的原始请求：

**行为特点**：
- 🎯 **零干预** - 不添加、不删除、不修改任何推理相关字段
- 🔄 **完全透传** - 客户端发什么就转发什么
- 🛡️ **头信息保护** - 不修改anthropic-beta等推理相关头信息

**使用场景**：
- 客户端需要完全控制推理参数
- 与原始API行为保持100%一致
- 不同客户端有不同的推理需求

**示例对比**：
```bash
# 客户端请求包含推理字段
{
  "model": "claude-opus-4-1-20250805",
  "reasoning": "auto",           // 配置为auto
  "messages": [...],
  "thinking": {"type": "enabled", "budget_tokens": 8192}
}

# auto模式：完全保留客户端设置
→ thinking字段原样转发，不做任何修改

# 如果配置为"high"：会被覆盖为 {"type": "enabled", "budget_tokens": 24576}
```

### 如何配置推理级别？

在 `config.json` 中为每个模型设置 `reasoning` 字段：

```json
{
  "models": [
    {
      "id": "claude-opus-4-1-20250805", 
      "type": "anthropic",
      "reasoning": "auto"  // auto/off/low/medium/high
    }
  ]
}
```

**推理级别说明**：

| 级别 | 行为 | 适用场景 |
|------|------|----------|
| `auto` | 完全遵循客户端原始请求参数 | 让客户端自主控制推理 |
| `off` | 强制禁用推理，删除所有推理字段 | 快速响应场景 |
| `low` | 轻度推理 (4096 tokens) | 简单任务 |
| `medium` | 中度推理 (12288 tokens) | 平衡性能与质量 |
| `high` | 深度推理 (24576 tokens) | 复杂任务 |

### 令牌多久刷新一次？

系统每6小时自动刷新一次访问令牌。刷新令牌有效期为8小时，确保有2小时的缓冲时间。

### 如何检查令牌状态？

查看服务器日志，成功刷新时会显示：
```
Token refreshed successfully, expires at: 2025-01-XX XX:XX:XX
```

### Claude Code无法连接怎么办？

1. 确保droid2api服务器正在运行：`curl http://localhost:3000/v1/models`
2. 检查Claude Code的API Base URL设置
3. 确认防火墙没有阻止端口3000

### 推理功能为什么没有生效？

**如果推理级别设置无效**：
1. 检查模型配置中的 `reasoning` 字段是否为有效值 (`auto/off/low/medium/high`)
2. 确认模型ID是否正确匹配config.json中的配置
3. 查看服务器日志确认推理字段是否正确处理

**如果使用auto模式但推理不生效**：
1. 确认客户端请求中包含了推理字段 (`reasoning` 或 `thinking`)
2. auto模式不会添加推理字段，只会保留客户端原有的设置
3. 如需强制推理，请改用 `low/medium/high` 级别

**推理字段对应关系**：
- OpenAI模型 (`gpt-*`) → 使用 `reasoning` 字段
- Anthropic模型 (`claude-*`) → 使用 `thinking` 字段
- Google模型 (`gemini-*`) → 使用 `thinkingConfig.thinkingLevel` 字段

### 如何更改端口？

编辑 `config.json` 中的 `port` 字段：

```json
{
  "port": 8080
}
```

### 如何启用调试日志？

在 `config.json` 中设置：

```json
{
  "dev_mode": true
}
```

## 故障排查

### 认证失败

确保已正确配置 refresh token：
- 设置环境变量 `DROID_REFRESH_KEY`
- 或创建 `~/.factory/auth.json` 文件

### 模型不可用

检查 `config.json` 中的模型配置，确保模型 ID 和类型正确。

## 许可证

MIT
