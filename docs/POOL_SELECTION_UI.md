# 🎯 密钥池选择 UI 功能文档

> 版本：v1.5.0 | 更新时间：2025-10-13 | BaSui 搞笑专业工程师出品 😎

---

## 🎉 功能概述

**密钥池选择 UI** 是 droid2api v1.5.0 新增的功能增强模块，为管理界面的四个核心功能按钮添加了**密钥池选择**能力！

### 升级的功能按钮

| 功能按钮 | 原功能 | 新增能力 | 使用场景 |
|---------|--------|----------|----------|
| ➕ **添加密钥** | 添加单个密钥到默认池 | 选择添加到哪个池 | 白嫖密钥放"白嫖池"，付费密钥放"主力池" |
| 📥 **批量导入** | 批量导入到默认池 | 选择导入到哪个池 | 批量导入 100 个白嫖密钥到"白嫖池" |
| 📤 **导出密钥** | 导出所有密钥 | 选择导出哪个池 + 多种格式 | 只导出"白嫖池"的密钥，或导出所有池 |
| 🧪 **批量测试** | 测试所有密钥 | 选择测试哪个池 + 并发数 | 只测试"白嫖池"的密钥，节省时间 |

---

## 🚀 详细功能说明

### 1️⃣ 添加密钥 - 池子选择

**弹窗内容：**

```
┌─────────────────────────────────────┐
│  ➕ 添加密钥                          │
├─────────────────────────────────────┤
│  密钥 (API Key):                     │
│  ┌─────────────────────────────────┐│
│  │ fk-xxx                          ││
│  └─────────────────────────────────┘│
│                                      │
│  🎯 所属密钥池:  🆕                  │
│  ┌─────────────────────────────────┐│
│  │ ▼ 默认池 (default)              ││ ← 下拉选择
│  │   白嫖池 (freebies)             ││
│  │   主力池 (main)                 ││
│  └─────────────────────────────────┘│
│  💡 选择密钥所属的池子，支持多级    │
│     密钥池优先级管理                 │
│                                      │
│  备注 (可选):                        │
│  ┌─────────────────────────────────┐│
│  │                                 ││
│  └─────────────────────────────────┘│
│                                      │
│  [ 添加 ]                            │
└─────────────────────────────────────┘
```

**使用场景：**
- 添加白嫖密钥时，选择"白嫖池"
- 添加付费密钥时，选择"主力池"
- 不选择池子时，自动添加到"默认池"

---

### 2️⃣ 批量导入 - 池子选择

**弹窗内容：**

```
┌─────────────────────────────────────┐
│  📤 批量导入密钥                      │
├─────────────────────────────────────┤
│  🎯 导入到哪个池:  🆕                │
│  ┌─────────────────────────────────┐│
│  │ ▼ 默认池 (default)              ││
│  │   白嫖池 (freebies)             ││
│  │   主力池 (main)                 ││
│  └─────────────────────────────────┘│
│  💡 所有导入的密钥将添加到选中的池子 │
│                                      │
│  密钥列表 (每行一个):                │
│  ┌─────────────────────────────────┐│
│  │ fk-xxx                          ││
│  │ fk-yyy                          ││
│  │ fk-zzz                          ││
│  └─────────────────────────────────┘│
│                                      │
│  [ 📥 开始导入 ]                     │
│                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  ✅ 成功导入 100 个密钥！             │
│  📊 总数: 100                        │
│  ✅ 成功: 100                        │
│  🔄 重复: 0                          │
│  ❌ 无效: 0                          │
│  🎯 导入到池: freebies               │
└─────────────────────────────────────┘
```

**使用场景：**
- 批量导入 100 个白嫖密钥到"白嫖池"
- 批量导入 10 个付费密钥到"主力池"
- 导入结果实时显示统计信息

---

### 3️⃣ 导出密钥 - 池子选择 + 格式选择

**弹窗内容：**

```
┌─────────────────────────────────────┐
│  📥 导出密钥                          │
├─────────────────────────────────────┤
│  🎯 导出哪个池:  🆕                  │
│  ┌─────────────────────────────────┐│
│  │ ☑ 全部池 (导出所有)              ││ ← 多选框
│  └─────────────────────────────────┘│
│                                      │
│  或选择特定池子:                     │
│  ┌─────────────────────────────────┐│
│  │ ☑ 默认池 (default)              ││ ← 多选列表
│  │ ☑ 白嫖池 (freebies)             ││   (按住 Ctrl 多选)
│  │ ☐ 主力池 (main)                 ││
│  └─────────────────────────────────┘│
│  💡 按住 Ctrl/Cmd 可多选池子         │
│                                      │
│  导出格式:                           │
│  ⦿ JSON (完整数据)                  │
│  ○ 纯文本 (仅密钥)                  │
│  ○ CSV (表格)                       │
│                                      │
│  状态筛选:                           │
│  ┌─────────────────────────────────┐│
│  │ ▼ 全部状态                      ││
│  │   仅可用                        ││
│  │   仅禁用                        ││
│  │   仅封禁                        ││
│  └─────────────────────────────────┘│
│                                      │
│  [ 📥 导出文件 ]                     │
└─────────────────────────────────────┘
```

**使用场景：**
- 只导出"白嫖池"的密钥，保存为 JSON 文件
- 导出"白嫖池" + "主力池"，排除默认池
- 导出所有可用状态的密钥，排除封禁的
- 导出为 CSV 格式，方便 Excel 打开

**支持的导出格式：**
- **JSON**：完整数据，包含所有字段（推荐⭐）
- **纯文本**：只包含密钥字符串，每行一个
- **CSV**：表格格式，适合 Excel/Numbers 打开

---

### 4️⃣ 批量测试 - 池子选择 + 并发数控制

**弹窗内容：**

```
┌─────────────────────────────────────┐
│  🧪 批量测试密钥                      │
├─────────────────────────────────────┤
│  🎯 测试哪个池:  🆕                  │
│  ⦿ 全部池 (测试所有密钥)             │
│  ○ 指定池                            │
│                                      │
│  选择池子:                           │
│  ┌─────────────────────────────────┐│
│  │ ▼ 默认池 (default)              ││
│  │   白嫖池 (freebies)             ││
│  │   主力池 (main)                 ││
│  └─────────────────────────────────┘│
│                                      │
│  测试并发数:                         │
│  ┌───┐                               │
│  │ 5 │ 个密钥/批次                  │
│  └───┘                               │
│  💡 同时测试几个密钥，建议 3-5 个    │
│                                      │
│  ☑ 测试完成后自动刷新列表            │
│                                      │
│  [ 🧪 开始测试 ]                     │
│                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  🎉 批量测试完成                     │
│  📊 总密钥数: 100 个                 │
│  🔍 已测试: 100 个                   │
│  ✅ 测试通过: 95 个                  │
│  ❌ 测试失败: 5 个                   │
│  🚫 自动封禁: 5 个                   │
│                                      │
│  💡 提示: 已自动封禁余额不足的密钥   │
└─────────────────────────────────────┘
```

**使用场景：**
- 只测试"白嫖池"的 100 个密钥，节省时间
- 测试所有池的密钥，检查健康状态
- 设置并发数为 3，避免触发 API 限流
- 测试完成后自动刷新列表，查看最新状态

---

## 🎨 UI 设计特点

### 1. 统一的设计语言

- **颜色方案**：紫色渐变主题 (#667eea → #764ba2)
- **圆角卡片**：12px-20px 圆角，柔和美观
- **动画效果**：淡入淡出、滑动、脉冲动画
- **Emoji 图标**：直观易懂，增强用户体验

### 2. 交互细节优化

**弹窗效果：**
- 背景模糊 (backdrop-filter: blur)
- 从下向上滑动进入 (slideUp animation)
- 关闭按钮悬停旋转效果

**下拉框优化：**
- 多选下拉框高亮选中项
- 选中项显示渐变紫色背景
- 支持 Ctrl/Cmd 多选

**结果反馈：**
- 成功：绿色渐变背景 + 边框
- 失败：红色渐变背景 + 边框
- 警告：黄色渐变背景 + 边框

### 3. 响应式设计

- 手机端：弹窗宽度适配，按钮垂直排列
- 平板端：网格布局自适应
- 桌面端：最佳视觉效果，动画流畅

---

## 🛠️ 技术实现

### 文件结构

```
f:\git_repository\droid2api\
├── public/
│   ├── index.html              # HTML 模态框结构 (新增 3 个模态框)
│   ├── app.js                  # 原有功能函数 (部分被覆盖)
│   ├── pool-selection-ui.js   # 🆕 池子选择 UI 逻辑 (新文件)
│   └── style.css               # 🆕 新增样式 (末尾追加)
└── docs/
    └── POOL_SELECTION_UI.md   # 🆕 本文档
```

### 核心函数

#### 1. 加载池子选项

```javascript
/**
 * 加载池子选项到指定的 select 元素
 * @param {string} selectId - select 元素的 ID
 * @param {boolean} includeDefault - 是否包含"默认池"选项
 */
async function loadPoolGroupOptions(selectId, includeDefault = true)
```

**调用时机：**
- 页面加载完成后 1 秒
- 打开任意模态框之前

**支持的 select 元素：**
- `newKeyPoolGroup` - 添加密钥
- `batchImportPoolGroup` - 批量导入
- `exportPoolGroup` - 导出密钥
- `testPoolGroup` - 批量测试
- `changePoolSelect` - 修改密钥池
- `poolGroupFilter` - 密钥管理页筛选器

#### 2. 显示模态框函数

```javascript
// 添加密钥（增强版）
function showAddKeyModal()

// 批量导入（增强版）
function showBatchImportModal()

// 导出密钥（新增）
function showExportKeysModal()

// 批量测试（新增）
function showBatchTestModal()
```

#### 3. 确认操作函数

```javascript
// 添加密钥（增强版，支持 poolGroup 参数）
async function addKey()

// 批量导入（增强版，支持 poolGroup 参数）
async function batchImport()

// 确认导出密钥（新增）
async function confirmExportKeys()

// 确认批量测试（新增）
async function confirmBatchTest()
```

#### 4. UI 交互函数

```javascript
// 切换"全部池"选项
function toggleExportAllPools()

// 切换测试池子选项
function toggleTestPoolOptions()
```

---

## 📡 API 调用

### 1. 获取密钥池列表

**请求：**
```javascript
GET /admin/pool-groups
Headers: {
  'x-admin-key': 'your-admin-key'
}
```

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "id": "freebies",
      "name": "白嫖池",
      "priority": 1,
      "description": "优先消耗白嫖密钥"
    },
    {
      "id": "main",
      "name": "主力池",
      "priority": 2,
      "description": "付费密钥兜底"
    }
  ]
}
```

### 2. 添加密钥（支持池子选择）

**请求：**
```javascript
POST /admin/keys
Headers: {
  'x-admin-key': 'your-admin-key',
  'Content-Type': 'application/json'
}
Body: {
  "key": "fk-xxx",
  "notes": "备注信息",
  "poolGroup": "freebies"  // 🆕 新增参数
}
```

### 3. 批量导入（支持池子选择）

**请求：**
```javascript
POST /admin/keys/batch
Headers: {
  'x-admin-key': 'your-admin-key',
  'Content-Type': 'application/json'
}
Body: {
  "keys": ["fk-xxx", "fk-yyy", "fk-zzz"],
  "poolGroup": "freebies"  // 🆕 新增参数
}
```

### 4. 导出密钥（支持池子选择）

**请求：**
```javascript
GET /admin/keys/export?status=all&format=json&poolGroups=freebies,main
Headers: {
  'x-admin-key': 'your-admin-key'
}
```

**参数说明：**
- `status` - 状态筛选 (all/active/disabled/banned)
- `format` - 导出格式 (json/text/csv)
- `poolGroups` - 池子 ID 列表（逗号分隔）

### 5. 批量测试（支持池子选择）

**请求：**
```javascript
POST /admin/keys/test-all?concurrency=5&poolGroup=freebies
Headers: {
  'x-admin-key': 'your-admin-key'
}
```

**参数说明：**
- `concurrency` - 并发数 (1-10)
- `poolGroup` - 池子 ID (可选)

---

## 🎯 使用指南

### 场景 1：区分白嫖密钥和付费密钥

**步骤：**

1️⃣ **创建两个池子**

访问管理界面 → 点击"创建密钥池"

- 池子 1：ID = `freebies`, 名称 = "白嫖池", 优先级 = 1
- 池子 2：ID = `main`, 名称 = "主力池", 优先级 = 2

2️⃣ **批量导入白嫖密钥**

点击"📥 批量导入" → 选择池子 "白嫖池" → 粘贴 100 个密钥 → 开始导入

3️⃣ **添加付费密钥**

点击"➕ 添加密钥" → 选择池子 "主力池" → 输入密钥 → 添加

4️⃣ **测试白嫖池密钥**

点击"🧪 批量测试" → 选择"指定池" → 选择"白嫖池" → 并发数 5 → 开始测试

5️⃣ **导出白嫖池密钥**

点击"📤 导出密钥" → 取消"全部池" → 选择"白嫖池" → 格式 JSON → 导出文件

---

### 场景 2：快速导出所有可用密钥

**步骤：**

1️⃣ 点击"📤 导出密钥"
2️⃣ 保持"全部池"勾选
3️⃣ 状态筛选选择"仅可用"
4️⃣ 格式选择"纯文本"
5️⃣ 点击"导出文件"

**得到的文件内容：**
```
fk-xxx
fk-yyy
fk-zzz
...
```

---

### 场景 3：测试特定池子的密钥健康状态

**步骤：**

1️⃣ 点击"🧪 批量测试"
2️⃣ 选择"指定池"
3️⃣ 选择"白嫖池"
4️⃣ 并发数设置为 3（避免限流）
5️⃣ 勾选"测试完成后自动刷新列表"
6️⃣ 点击"开始测试"

**测试结果：**
- ✅ 95 个密钥通过
- ❌ 5 个密钥失败（余额不足）
- 🚫 自动封禁 5 个失败密钥

---

## 🔧 后端 API 修改建议

为了完整支持这些功能，后端需要添加以下 API 能力：

### 1. 导出 API 支持池子筛选

**修改文件：** `api/admin-routes.js`

**修改点：** `/admin/keys/export` 路由

```javascript
router.get('/keys/export', wrapAsync(async (req, res) => {
  const { status, format = 'json', poolGroups } = req.query;

  // 🆕 解析 poolGroups 参数
  let selectedPools = null;
  if (poolGroups) {
    selectedPools = poolGroups.split(',').map(p => p.trim());
  }

  // 获取密钥列表
  let keys = keyPoolManager.getKeys();

  // 🆕 按池子筛选
  if (selectedPools && selectedPools.length > 0) {
    keys = keys.filter(key =>
      selectedPools.includes(key.poolGroup || 'default')
    );
  }

  // 按状态筛选
  if (status && status !== 'all') {
    keys = keys.filter(key => key.status === status);
  }

  // 导出为不同格式
  if (format === 'json') {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename="keys_export.json"');
    res.json(keys);
  } else if (format === 'text') {
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Content-Disposition', 'attachment; filename="keys_export.txt"');
    res.send(keys.map(k => k.key).join('\n'));
  } else if (format === 'csv') {
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="keys_export.csv"');
    const csv = convertToCSV(keys);  // 自定义 CSV 转换函数
    res.send(csv);
  }
}, 'export keys'));
```

### 2. 批量测试 API 支持池子筛选

**修改文件：** `api/admin-routes.js`

**修改点：** `/admin/keys/test-all` 路由

```javascript
router.post('/keys/test-all', wrapAsync(async (req, res) => {
  const { concurrency = 5, poolGroup } = req.query;

  // 获取密钥列表
  let keys = keyPoolManager.getKeys();

  // 🆕 按池子筛选
  if (poolGroup) {
    keys = keys.filter(key =>
      (key.poolGroup || 'default') === poolGroup
    );
  }

  // 批量测试逻辑（使用并发数控制）
  const results = await testKeysWithConcurrency(keys, concurrency);

  sendSuccessResponse(res, results);
}, 'test all keys'));
```

### 3. 添加/导入密钥支持 poolGroup 参数

**修改文件：** `api/admin-routes.js`

**修改点：** `/admin/keys` (POST) 和 `/admin/keys/batch` (POST)

```javascript
// 添加密钥
router.post('/keys', wrapAsync(async (req, res) => {
  const { key, notes, poolGroup } = req.body;  // 🆕 新增 poolGroup

  // 添加密钥到池子
  const result = keyPoolManager.addKey({
    key,
    notes,
    poolGroup: poolGroup || undefined  // 🆕 传递池子 ID
  });

  sendSuccessResponse(res, result);
}, 'add key'));

// 批量导入
router.post('/keys/batch', wrapAsync(async (req, res) => {
  const { keys, poolGroup } = req.body;  // 🆕 新增 poolGroup

  const results = {
    total: keys.length,
    success: 0,
    duplicate: 0,
    invalid: 0
  };

  for (const key of keys) {
    try {
      keyPoolManager.addKey({
        key,
        poolGroup: poolGroup || undefined  // 🆕 传递池子 ID
      });
      results.success++;
    } catch (error) {
      if (error.message.includes('已存在')) {
        results.duplicate++;
      } else {
        results.invalid++;
      }
    }
  }

  sendSuccessResponse(res, results);
}, 'batch import keys'));
```

---

## 📊 性能优化

### 1. 池子选项缓存

**问题：** 每次打开模态框都重新加载池子列表

**优化：** 缓存池子列表，定期刷新

```javascript
let poolGroupsCache = null;
let poolGroupsCacheTime = 0;
const CACHE_TTL = 5 * 60 * 1000;  // 5 分钟

async function loadPoolGroupOptions(selectId, includeDefault = true) {
    // 检查缓存
    if (poolGroupsCache && Date.now() - poolGroupsCacheTime < CACHE_TTL) {
        updateSelectOptions(selectId, poolGroupsCache, includeDefault);
        return;
    }

    // 重新加载
    const response = await apiRequest('/pool-groups');
    poolGroupsCache = response.data || [];
    poolGroupsCacheTime = Date.now();

    updateSelectOptions(selectId, poolGroupsCache, includeDefault);
}
```

### 2. 并发测试优化

**问题：** 同时测试 100 个密钥会触发 API 限流

**优化：** 使用批量并发控制

```javascript
async function testKeysWithConcurrency(keys, concurrency = 5) {
    const results = {
        total: keys.length,
        tested: 0,
        success: 0,
        failed: 0,
        banned: 0
    };

    // 分批测试
    for (let i = 0; i < keys.length; i += concurrency) {
        const batch = keys.slice(i, i + concurrency);
        const batchResults = await Promise.allSettled(
            batch.map(key => testSingleKey(key))
        );

        // 统计结果
        batchResults.forEach((result, index) => {
            results.tested++;
            if (result.status === 'fulfilled' && result.value.success) {
                results.success++;
            } else {
                results.failed++;
                if (shouldBanKey(result.reason)) {
                    results.banned++;
                }
            }
        });

        // 延迟 500ms，避免限流
        await sleep(500);
    }

    return results;
}
```

---

## 🐛 故障排查

### 问题 1：池子选项不显示

**症状：** 打开模态框后，池子下拉框只有"默认池"选项

**原因：**
1. 后端 `/admin/pool-groups` API 未实现
2. 前端 API 请求失败
3. 池子数据为空

**解决方案：**
```bash
# 1. 检查后端 API
curl -H "x-admin-key: your-admin-key" http://localhost:3000/admin/pool-groups

# 2. 检查浏览器控制台
# 查看是否有 API 错误日志

# 3. 手动创建池子
# 访问管理界面 → 点击"创建密钥池"
```

---

### 问题 2：导出文件时报 404 错误

**症状：** 点击"导出文件"后弹出 404 错误

**原因：** 后端 `/admin/keys/export` 路由未支持 `poolGroups` 参数

**解决方案：**
```javascript
// 修改 api/admin-routes.js
router.get('/keys/export', async (req, res) => {
  const { poolGroups } = req.query;  // 添加参数解析
  // ... 后续逻辑
});
```

---

### 问题 3：批量测试卡住不动

**症状：** 点击"开始测试"后一直显示"正在测试中..."

**原因：**
1. 并发数设置过高，触发 API 限流
2. 密钥数量太多，测试超时
3. 前端 API 超时配置不足

**解决方案：**
```bash
# 1. 降低并发数
并发数从 10 降到 3-5

# 2. 分批测试
先测试"白嫖池"，再测试"主力池"

# 3. 增加超时配置
apiRequest('/keys/test-all', 'POST', null, { timeout: 60000 })
```

---

## 🎨 自定义样式

### 修改主题颜色

**文件：** `public/style.css`

**修改点：** 搜索 `#667eea` 和 `#764ba2`，替换为你的主题色

```css
/* 原色：紫色渐变 */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* 示例：改为蓝色渐变 */
background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);

/* 示例:改为绿色渐变 */
background: linear-gradient(135deg, #10b981 0%, #059669 100%);
```

### 修改弹窗宽度

**文件：** `public/style.css`

**修改点：** `.modal-content` 类

```css
.modal-content {
  max-width: 550px;  /* 修改为你想要的宽度 */
}
```

---

## 📝 待办事项

### 短期（v1.5.1）

- [ ] 后端支持导出 CSV 格式
- [ ] 后端支持并发测试控制
- [ ] 添加"导出配置"预设功能
- [ ] 批量测试进度条显示

### 长期（v2.0）

- [ ] 支持自定义导出字段
- [ ] 支持按 Token 使用量排序导出
- [ ] 支持定时自动测试
- [ ] 支持批量修改密钥池

---

## 🙏 致谢

**感谢以下技术栈：**
- Express.js - Web 服务器框架
- Vanilla JavaScript - 原生 JS，轻量高效
- CSS3 - 现代 CSS 特性（渐变、动画、Grid 布局）
- Emoji - 让界面更生动有趣 😎

**BaSui 搞笑专业工程师团队出品！** 🎉

> 代码能跑就行？不！代码要写得漂亮！
> 态度要严肃？不！编程应该很快乐！
> 专业的技术 + 搞笑的灵魂 = 完美产品！💪🚀

---

## 📞 联系我们

如有问题或建议，欢迎提 Issue！

**项目地址：** https://github.com/your-repo/droid2api

**文档更新时间：** 2025-10-13

**版本：** v1.5.0

---

**Happy Coding! 🎊**
