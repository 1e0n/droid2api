# 🏗️ droid2api - 完整架构文档

> **BaSui 的超详细架构指南** | 最后更新：2025-10-13

---

## 📋 目录

1. [系统架构总览](#系统架构总览)
2. [请求处理流程](#请求处理流程)
3. [核心模块详解](#核心模块详解)
4. [API 端点参考](#api-端点参考)
5. [数据流图](#数据流图)
6. [技术栈](#技术栈)

---

## 🎯 系统架构总览

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端请求                                 │
│            (OpenAI/Anthropic/Common 格式)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Express 服务器 (server.js)                   │
│                     • 端口：3000 (可配置)                          │
│                     • 集群模式支持 (可选)                          │
│                     • CORS 启用                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        中间件层 (middleware/)                     │
│  ┌──────────────┬───────────────────┬─────────────────────┐     │
│  │ client-auth  │  log-collector    │  stats-tracker      │     │
│  │ API_ACCESS   │  日志缓冲         │  请求统计            │     │
│  │ _KEY 验证    │                   │                     │     │
│  └──────────────┴───────────────────┴─────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                          路由层 (routes.js)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  客户端 API (OpenAI 兼容)                                 │    │
│  │  • GET  /v1/models                                       │    │
│  │  • POST /v1/chat/completions  (自动转换)                 │    │
│  │  • POST /v1/responses         (OpenAI 直接转发)          │    │
│  │  • POST /v1/messages          (Anthropic 直接转发)       │    │
│  │  • POST /v1/messages/count_tokens (Token 计数)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  管理 API (admin-routes.js)                              │    │
│  │  • GET/POST/PUT/PATCH/DELETE /admin/*                    │    │
│  │  • 密钥池管理、Token 使用量、请求统计、日志流            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      密钥池管理 (auth.js)                          │
│  • KeyPoolManager 类                                             │
│  • 轮询算法 (6种)                                                 │
│  • 健康检查                                                       │
│  • 自动封禁 (402 错误)                                            │
│  • 多级密钥池 (v1.5+)                                             │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      格式转换层 (transformers/)                    │
│  ┌──────────────────────┬───────────────────────────────┐       │
│  │  请求转换             │  响应转换                      │       │
│  │  • request-anthropic │  • response-anthropic         │       │
│  │  • request-openai    │  • response-openai            │       │
│  │  • request-common    │                               │       │
│  │  • headers-common    │                               │       │
│  └──────────────────────┴───────────────────────────────┘       │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP 连接池 (utils/http-client.js)           │
│  • Keep-Alive 连接复用                                            │
│  • 超时控制 (120s)                                                │
│  • 性能优化 (延迟降低 80%)                                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      上游 LLM API (Factory AI)                    │
│  • api.factory.us.aliyun.ws/v1/messages                          │
│  • api.factory.us.aliyun.ws/v1/responses                         │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                          响应处理                                  │
│  • Token 使用量提取                                               │
│  • 格式转换 (Anthropic → OpenAI)                                 │
│  • 流式/非流式响应处理                                            │
│  • 请求统计记录                                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                         返回客户端                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 请求处理流程

### 流程图

```mermaid
graph TD
    A[客户端请求] --> B{API_ACCESS_KEY 验证}
    B -->|失败| C[403 Forbidden]
    B -->|成功| D[日志收集中间件]
    D --> E[统计追踪中间件]
    E --> F{路由匹配}

    F --> G[/v1/models]
    F --> H[/v1/chat/completions]
    F --> I[/v1/responses]
    F --> J[/v1/messages]
    F --> K[/admin/*]

    H --> L{模型类型判断}
    L -->|Anthropic| M[transformToAnthropic]
    L -->|OpenAI| N[transformToOpenAI]
    L -->|Common| O[transformToCommon]

    M --> P[密钥池获取密钥]
    N --> P
    O --> P

    P --> Q{密钥可用?}
    Q -->|否| R[500 密钥池错误]
    Q -->|是| S[HTTP 连接池请求]

    S --> T{响应状态}
    T -->|402| U[封禁密钥 + 返回错误]
    T -->|400-599| V[返回上游错误]
    T -->|200| W{流式响应?}

    W -->|是| X[流式转换 + 转发]
    W -->|否| Y[非流式转换 + 返回]

    X --> Z[记录 Token 使用量]
    Y --> Z
    Z --> AA[返回客户端]
```

---

## 📦 核心模块详解

### 1. 服务器入口 (server.js)

**职责**：Express 服务器启动，支持集群模式

**关键代码**：
```javascript
// 单进程模式
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use('/admin', adminRoutes);
app.use('/', routes);
app.listen(PORT);

// 集群模式 (CLUSTER_MODE=true)
if (cluster.isPrimary) {
  for (let i = 0; i < numWorkers; i++) {
    cluster.fork();
  }
} else {
  // Worker 进程运行上述代码
}
```

**特性**：
- ✅ 支持集群模式 (多核 CPU 并行处理)
- ✅ 自动重启崩溃的 Worker
- ✅ 零停机重载 (SIGUSR2 信号)

---

### 2. 密钥池管理 (auth.js)

**职责**：密钥轮询、健康检查、自动封禁

**核心类**：`KeyPoolManager`

**轮询算法**：
| 算法 | 适用场景 | 特点 |
|------|----------|------|
| `round-robin` | 密钥额度相同 | 平均分配请求，依次轮流使用 |
| `random` | 无特殊要求 | 随机分散负载 |
| `least-used` | 新密钥加入 | 优先使用次数最少的密钥 |
| `weighted-score` | 密钥质量不同 | 基于成功率和经验值加权选择 |
| `least-token-used` | Token 额度相同 | 均衡 Token 使用量 |
| `max-remaining` | 密钥额度不同 | 优先使用剩余 Token 最多的密钥 |

**多级密钥池 (v1.5+)**：
```javascript
// 示例：白嫖池 → 主力池
poolGroups: [
  { id: 'freebies', name: '白嫖池', priority: 1 },
  { id: 'main', name: '主力池', priority: 2 }
]
```

**自动封禁**：
- 触发条件：上游返回 402 Payment Required
- 操作：`status: 'banned'`，永久排除
- 恢复方式：管理员手动切换到 `active`

---

### 3. 格式转换层 (transformers/)

#### 请求转换

**request-anthropic.js** - OpenAI → Anthropic
```javascript
// OpenAI 格式
{
  model: 'claude-sonnet-4-5',
  messages: [
    { role: 'user', content: 'Hello' }
  ]
}

// ↓ 转换后 ↓

// Anthropic 格式
{
  model: 'claude-sonnet-4-5-20250929',
  messages: [
    { role: 'user', content: 'Hello' }
  ],
  max_tokens: 8192
}
```

**request-openai.js** - OpenAI → OpenAI Responses API
```javascript
// OpenAI 格式 (Chat Completions)
{
  model: 'gpt-4',
  messages: [...]
}

// ↓ 转换后 ↓

// OpenAI Responses API 格式
{
  model: 'gpt-4',
  input: [...],
  instructions: '...'
}
```

**request-common.js** - 最小化转换 (GLM 等)
```javascript
// 直接透传，仅添加必要字段
```

#### 响应转换

**response-anthropic.js** - Anthropic SSE → OpenAI SSE
```javascript
// Anthropic 流式响应
data: {"type":"content_block_delta","delta":{"text":"Hello"}}

// ↓ 转换后 ↓

// OpenAI 流式响应
data: {"choices":[{"delta":{"content":"Hello"}}]}
```

**response-openai.js** - OpenAI Responses → Chat Completions
```javascript
// OpenAI Responses 格式
{
  id: 'resp_xxx',
  output: [{ type: 'message', content: [...] }]
}

// ↓ 转换后 ↓

// OpenAI Chat Completions 格式
{
  id: 'chatcmpl-xxx',
  choices: [{ message: { role: 'assistant', content: '...' } }]
}
```

---

### 4. HTTP 连接池 (utils/http-client.js)

**职责**：复用 TCP 连接，减少握手开销

**性能对比**：
| 场景 | 无连接池 | 有连接池 | 性能提升 |
|------|----------|----------|----------|
| 平均延迟 | 250ms | 50ms | ↓ 80% |
| 吞吐量 | 500 RPS | 2000+ RPS | ↑ 300% |

**核心代码**：
```javascript
const agent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 100,
  maxFreeSockets: 10,
  timeout: 120000
});

export default function fetchWithPool(url, options) {
  return fetch(url, { ...options, agent });
}
```

---

### 5. 日志系统 (logger.js)

**职责**：分级日志（INFO/WARN/DEBUG/ERROR）

**日志级别**：
- `INFO` - 正常请求/响应
- `DEBUG` - 详细调试信息 (需配置 `log_level: 'debug'`)
- `WARN` - 警告信息
- `ERROR` - 错误信息

**开发模式**：
```bash
export NODE_ENV=development
npm start
# 所有日志输出到控制台，不写入文件
```

**生产模式**：
```bash
export NODE_ENV=production
npm start
# INFO/ERROR 写入 logs/app.log
```

---

## 🌐 API 端点参考

### 客户端 API (OpenAI 兼容)

#### 1. GET /v1/models

**描述**：返回可用模型列表

**认证**：需要 `API_ACCESS_KEY` (可选)

**请求示例**：
```bash
curl http://localhost:3000/v1/models
```

**响应示例**：
```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-sonnet-4-5-20250929",
      "object": "model",
      "created": 1696000000,
      "owned_by": "anthropic"
    }
  ]
}
```

---

#### 2. POST /v1/chat/completions

**描述**：OpenAI 格式聊天补全 (自动转换为 Anthropic/OpenAI/Common)

**认证**：需要 `API_ACCESS_KEY` (可选)

**请求示例**：
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

**支持参数**：
- `model` (必需) - 模型 ID
- `messages` (必需) - 消息列表
- `stream` (可选) - 流式响应 (默认 false)
- `temperature` (可选) - 温度 (0-2)
- `max_tokens` (可选) - 最大 Token 数

**响应格式**：
- 流式：OpenAI SSE 格式
- 非流式：OpenAI Chat Completions 格式

---

#### 3. POST /v1/responses

**描述**：OpenAI Responses API 直接转发 (不做格式转换)

**认证**：需要 `API_ACCESS_KEY` (可选)

**请求示例**：
```bash
curl -X POST http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [...],
    "instructions": "...",
    "stream": true
  }'
```

**限制**：仅支持 `type: 'openai'` 的模型

---

#### 4. POST /v1/messages

**描述**：Anthropic Messages API 直接转发 (不做格式转换)

**认证**：需要 `API_ACCESS_KEY` (可选)

**请求示例**：
```bash
curl -X POST http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024,
    "stream": true
  }'
```

**限制**：仅支持 `type: 'anthropic'` 的模型

---

#### 5. POST /v1/messages/count_tokens

**描述**：Anthropic Token 计数 (不调用模型，只计算 Token)

**认证**：需要 `API_ACCESS_KEY` (可选)

**请求示例**：
```bash
curl -X POST http://localhost:3000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

**响应示例**：
```json
{
  "input_tokens": 10
}
```

---

### 管理后台 API (admin-routes.js)

**认证**：所有管理 API 需要 `X-Admin-Key` 头

```bash
curl -H "X-Admin-Key: YOUR_ADMIN_KEY" http://localhost:3000/admin/stats
```

#### 密钥池管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/stats` | 获取密钥池统计信息 |
| GET | `/admin/keys` | 获取密钥列表 (支持分页) |
| GET | `/admin/keys/:id` | 获取单个密钥详情 |
| GET | `/admin/keys/export` | 导出密钥为 txt 文件 |
| POST | `/admin/keys` | 添加单个密钥 |
| POST | `/admin/keys/batch` | 批量导入密钥 |
| POST | `/admin/keys/:id/test` | 测试单个密钥 |
| POST | `/admin/keys/test-all` | 批量测试所有密钥 |
| PUT | `/admin/keys/:id` | 完整更新密钥 |
| PATCH | `/admin/keys/:id/toggle` | 切换密钥状态 (active ↔ disabled) |
| PATCH | `/admin/keys/:id/notes` | 更新密钥备注 |
| PATCH | `/admin/keys/:id/pool` | 修改密钥所属池子 |
| DELETE | `/admin/keys/:id` | 删除单个密钥 |
| DELETE | `/admin/keys/disabled` | 删除所有禁用的密钥 |
| DELETE | `/admin/keys/banned` | 删除所有封禁的密钥 |

#### 多级密钥池管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/pool-groups` | 获取所有池子的统计信息 |
| POST | `/admin/pool-groups` | 创建新的密钥池组 |
| DELETE | `/admin/pool-groups/:id` | 删除密钥池组 |

#### 配置管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/config` | 获取当前轮询配置 |
| PUT | `/admin/config` | 更新轮询配置 |
| POST | `/admin/config/reset` | 重置轮询配置为默认值 |

#### Token 使用量管理 (token-usage-routes.js)

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/token/stats` | 获取 Token 统计摘要 |
| GET | `/admin/token/usage` | 获取所有密钥的使用量 |
| GET | `/admin/token/usage/:keyId` | 获取单个密钥的使用量 |
| POST | `/admin/token/sync` | 手动同步 Token 使用量 |
| GET | `/admin/token/trend` | 获取 Token 使用趋势 |
| DELETE | `/admin/token/cache` | 清除 Token 使用量缓存 |

#### 请求统计 (stats-routes.js)

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/stats/summary` | 获取请求统计摘要 |
| GET | `/admin/stats/full` | 获取完整请求统计 |
| GET | `/admin/stats/today` | 获取今日请求统计 |
| GET | `/admin/stats/trend` | 获取请求趋势 |
| POST | `/admin/stats/reset` | 重置请求统计 |

#### 日志管理 (log-stream-routes.js)

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/admin/logs/stream` | 实时日志流 (SSE) |
| GET | `/admin/logs/history` | 获取历史日志 |
| DELETE | `/admin/logs/clear` | 清除日志缓冲 |
| GET | `/admin/logs/stats` | 获取日志统计 |

---

## 📊 数据流图

### Token 使用量同步流程

```
┌──────────────────────────────────────────────────────────────┐
│  定时任务 (每 30 分钟)                                         │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│  factory-api-client.js                                        │
│  • GET /api/organization (获取真实额度)                       │
│  • GET /api/organization/members/chat-usage (获取已使用量)    │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│  计算剩余额度                                                  │
│  remaining = realAllowance - orgTotalTokensUsed              │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│  写入 data/token_usage.json (异步批量写入)                     │
│  {                                                            │
│    "keys": {                                                  │
│      "key_xxx": {                                             │
│        "success": true,                                       │
│        "standard": {                                          │
│          "totalAllowance": 38000000,                          │
│          "orgTotalTokensUsed": 5000000,                       │
│          "remaining": 33000000                                │
│        },                                                     │
│        "last_sync": "2025-10-12T18:00:00.000Z"                │
│      }                                                        │
│    }                                                          │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Node.js | 18+ | 运行时环境 |
| Express | ^4.18.2 | Web 服务器框架 |
| node-fetch | ^3.3.2 | HTTP 请求库 (支持 HTTP/2) |
| dotenv | ^17.2.3 | 环境变量加载 |

### 可选依赖

| 技术 | 版本 | 用途 |
|------|------|------|
| Redis | ^4.x | 缓存密钥池数据 (可选) |

### 前端 (管理界面)

| 技术 | 用途 |
|------|------|
| 原生 HTML/CSS/JS | 管理面板 UI |
| Fetch API | 调用管理 API |
| Server-Sent Events | 实时日志流 |

---

## 📂 目录结构

```
droid2api/
├── api/                          # 管理后台 API
│   ├── admin-routes.js           # 密钥池 CRUD
│   ├── token-usage-routes.js     # Token 统计
│   ├── stats-routes.js           # 请求统计
│   ├── log-stream-routes.js      # 实时日志流
│   └── admin-error-handlers.js   # 错误处理
├── transformers/                 # 格式转换层
│   ├── request-anthropic.js      # OpenAI → Anthropic
│   ├── request-openai.js         # OpenAI → OpenAI Responses
│   ├── request-common.js         # 最小化转换
│   ├── response-anthropic.js     # Anthropic → OpenAI SSE
│   ├── response-openai.js        # OpenAI Responses → Chat Completions
│   └── headers-common.js         # 公共头信息
├── utils/                        # 工具函数
│   ├── route-helpers.js          # 路由辅助
│   ├── factory-api-client.js     # Factory API 客户端
│   ├── http-client.js            # HTTP 连接池
│   ├── async-file-writer.js      # 异步文件写入
│   ├── redis-cache.js            # Redis 缓存
│   ├── request-stats.js          # 请求统计
│   ├── uuid.js                   # UUID 生成
│   └── advanced-algorithms.js    # 高级算法
├── middleware/                   # 中间件层
│   ├── client-auth.js            # 客户端认证
│   ├── stats-tracker.js          # 统计追踪
│   └── log-collector.js          # 日志收集
├── public/                       # 前端管理界面
│   ├── index.html                # 管理面板
│   ├── app.js                    # 前端逻辑
│   └── style.css                 # 样式
├── data/                         # 数据存储
│   ├── config.json               # 项目配置
│   ├── key_pool.json             # 密钥池数据
│   ├── token_usage.json          # Token 使用量
│   └── request_stats.json        # 请求统计
├── tests/                        # 测试套件
│   ├── test-token-usage.js       # Token 使用量测试
│   ├── test-factory-balance.js   # Factory API 测试
│   └── benchmark.js              # 性能测试
├── docs/                         # 文档
│   ├── ARCHITECTURE.md           # 架构文档 (本文档)
│   └── MULTI_TIER_POOL.md        # 多级密钥池文档
├── server.js                     # Express 服务器入口
├── routes.js                     # 主路由
├── auth.js                       # 密钥池管理
├── config.js                     # 配置加载
├── logger.js                     # 日志系统
├── auth-oauth.js                 # OAuth 认证
├── package.json                  # 依赖清单
├── .env.example                  # 环境变量示例
└── README.md                     # 项目说明
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
nano .env  # 编辑环境变量
```

**必需变量**：
```bash
ADMIN_ACCESS_KEY=your-admin-key-here
```

### 3. 启动服务器

```bash
npm start
```

### 4. 访问管理界面

打开浏览器访问：`http://localhost:3000`

### 5. 测试 API

```bash
# 获取模型列表
curl http://localhost:3000/v1/models

# 发送聊天请求
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 📚 相关文档

- [多级密钥池功能说明](./MULTI_TIER_POOL.md)
- [项目主文档](../CLAUDE.md)
- [Docker 部署指南](../DOCKER_DEPLOY.md)
- [README](../README.md)

---

## 🎉 总结

**droid2api** 是一个生产级的 OpenAI 兼容 API 代理服务器，具备以下核心特性：

✅ **统一接口标准** - OpenAI API 格式，兼容多种 LLM 模型
✅ **透明代理模式** - 支持 Claude Code 直接集成
✅ **密钥池轮询** - 6 种算法，智能选择密钥
✅ **多级密钥池** - 白嫖池优先，主力池备用
✅ **Token 使用量追踪** - Factory API 集成
✅ **性能优化** - HTTP 连接池 + Redis 缓存 + 集群模式
✅ **企业级管理** - 完整的管理后台 API

**有问题？**
- 查看日志：`tail -f logs/app.log`
- 查看密钥池状态：`GET /admin/stats`
- 查看 Token 使用量：`GET /admin/token/stats`

---

**祝你使用愉快！🎊 —— BaSui 敬上**
