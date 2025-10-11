import express from 'express';
import fetch from 'node-fetch';
import { getConfig, getModelById, getEndpointByType, getSystemPrompt, getModelReasoning } from './config.js';
import { logInfo, logDebug, logError, logRequest, logResponse } from './logger.js';
import { transformToAnthropic, getAnthropicHeaders } from './transformers/request-anthropic.js';
import { transformToOpenAI, getOpenAIHeaders } from './transformers/request-openai.js';
import { transformToCommon, getCommonHeaders } from './transformers/request-common.js';
import { AnthropicResponseTransformer } from './transformers/response-anthropic.js';
import { OpenAIResponseTransformer } from './transformers/response-openai.js';
import { getApiKey, recordRequestResult } from './auth.js';
import { getKeyManager } from './key-manager.js';
import { isServerKeySet, setServerKey } from './server-auth.js';

const router = express.Router();

/**
 * Convert a /v1/responses API result to a /v1/chat/completions-compatible format.
 * Works for non-streaming responses.
 */
function convertResponseToChatCompletion(resp) {
  if (!resp || typeof resp !== 'object') {
    throw new Error('Invalid response object');
  }

  const outputMsg = (resp.output || []).find(o => o.type === 'message');
  const textBlocks = outputMsg?.content?.filter(c => c.type === 'output_text') || [];
  const content = textBlocks.map(c => c.text).join('');

  const chatCompletion = {
    id: resp.id ? resp.id.replace(/^resp_/, 'chatcmpl-') : `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: resp.created_at || Math.floor(Date.now() / 1000),
    model: resp.model || 'unknown-model',
    choices: [
      {
        index: 0,
        message: {
          role: outputMsg?.role || 'assistant',
          content: content || ''
        },
        finish_reason: resp.status === 'completed' ? 'stop' : 'unknown'
      }
    ],
    usage: {
      prompt_tokens: resp.usage?.input_tokens ?? 0,
      completion_tokens: resp.usage?.output_tokens ?? 0,
      total_tokens: resp.usage?.total_tokens ?? 0
    }
  };

  return chatCompletion;
}

router.get('/v1/models', (req, res) => {
  logInfo('GET /v1/models');
  
  try {
    const config = getConfig();
    const models = config.models.map(model => ({
      id: model.id,
      object: 'model',
      created: Date.now(),
      owned_by: model.type,
      permission: [],
      root: model.id,
      parent: null
    }));

    const response = {
      object: 'list',
      data: models
    };

    logResponse(200, null, response);
    res.json(response);
  } catch (error) {
    logError('Error in GET /v1/models', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// 标准 OpenAI 聊天补全处理函数（带格式转换）
async function handleChatCompletions(req, res) {
  logInfo('POST /v1/chat/completions');
  
  try {
    const openaiRequest = req.body;
    const modelId = openaiRequest.model;

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Routing to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key (will auto-refresh if needed)
    // When server access key is enabled, Authorization header is reserved for server auth.
    // To pass a client-provided upstream token (client mode), use X-Endpoint-Authorization.
    let authHeader;
    try {
      const clientUpstreamAuth = req.headers['x-endpoint-authorization'] || null;
      authHeader = await getApiKey(clientUpstreamAuth);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    let transformedRequest;
    let headers;
    const clientHeaders = req.headers;

    // Log received client headers for debugging
    logDebug('Client headers received', {
      'x-factory-client': clientHeaders['x-factory-client'],
      'x-session-id': clientHeaders['x-session-id'],
      'x-assistant-message-id': clientHeaders['x-assistant-message-id'],
      'user-agent': clientHeaders['user-agent']
    });

    if (model.type === 'anthropic') {
      transformedRequest = transformToAnthropic(openaiRequest);
      const isStreaming = openaiRequest.stream === true;
      headers = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, modelId);
    } else if (model.type === 'openai') {
      transformedRequest = transformToOpenAI(openaiRequest);
      headers = getOpenAIHeaders(authHeader, clientHeaders);
    } else if (model.type === 'common') {
      transformedRequest = transformToCommon(openaiRequest);
      headers = getCommonHeaders(authHeader, clientHeaders);
    } else {
      return res.status(500).json({ error: `Unknown endpoint type: ${model.type}` });
    }

    logRequest('POST', endpoint.base_url, headers, transformedRequest);

    const response = await fetch(endpoint.base_url, {
      method: 'POST',
      headers,
      body: JSON.stringify(transformedRequest)
    });

    logInfo(`Response status: ${response.status}`);
    
    // Record request result (2xx = success)
    const isSuccess = response.status >= 200 && response.status < 300;
    recordRequestResult(endpoint.base_url, isSuccess);

    if (!response.ok) {
      const errorText = await response.text();
      logError(`Endpoint error: ${response.status}`, new Error(errorText));
      return res.status(response.status).json({ 
        error: `Endpoint returned ${response.status}`,
        details: errorText 
      });
    }

    const isStreaming = transformedRequest.stream === true;

    if (isStreaming) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // common 类型直接转发，不使用 transformer
      if (model.type === 'common') {
        try {
          for await (const chunk of response.body) {
            res.write(chunk);
          }
          res.end();
          logInfo('Stream forwarded (common type)');
        } catch (streamError) {
          logError('Stream error', streamError);
          res.end();
        }
      } else {
        // anthropic 和 openai 类型使用 transformer
        let transformer;
        if (model.type === 'anthropic') {
          transformer = new AnthropicResponseTransformer(modelId, `chatcmpl-${Date.now()}`);
        } else if (model.type === 'openai') {
          transformer = new OpenAIResponseTransformer(modelId, `chatcmpl-${Date.now()}`);
        }

        try {
          for await (const chunk of transformer.transformStream(response.body)) {
            res.write(chunk);
          }
          res.end();
          logInfo('Stream completed');
        } catch (streamError) {
          logError('Stream error', streamError);
          res.end();
        }
      }
    } else {
      const data = await response.json();
      if (model.type === 'openai') {
        try {
          const converted = convertResponseToChatCompletion(data);
          logResponse(200, null, converted);
          res.json(converted);
        } catch (e) {
          // 如果转换失败，回退为原始数据
          logResponse(200, null, data);
          res.json(data);
        }
      } else {
        // anthropic/common: 保持现有逻辑，直接转发
        logResponse(200, null, data);
        res.json(data);
      }
    }

  } catch (error) {
    logError('Error in /v1/chat/completions', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}

// 直接转发 OpenAI 请求（不做格式转换）
async function handleDirectResponses(req, res) {
  logInfo('POST /v1/responses');
  
  try {
    const openaiRequest = req.body;
    const modelId = openaiRequest.model;

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    // 只允许 openai 类型端点
    if (model.type !== 'openai') {
      return res.status(400).json({ 
        error: 'Invalid endpoint type',
        message: `/v1/responses 接口只支持 openai 类型端点，当前模型 ${modelId} 是 ${model.type} 类型`
      });
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Direct forwarding to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key - support client x-api-key for anthropic endpoint
    let authHeader;
    try {
      const clientAuthFromXApiKey = req.headers['x-api-key']
        ? `Bearer ${req.headers['x-api-key']}`
        : null;
      const clientUpstreamAuth = req.headers['x-endpoint-authorization'] || null;
      authHeader = await getApiKey(clientUpstreamAuth || clientAuthFromXApiKey);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    const clientHeaders = req.headers;
    
    // 获取 headers
    const headers = getOpenAIHeaders(authHeader, clientHeaders);

    // 注入系统提示到 instructions 字段
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...openaiRequest };
    if (systemPrompt) {
      // 如果已有 instructions，则在前面添加系统提示
      if (modifiedRequest.instructions) {
        modifiedRequest.instructions = systemPrompt + modifiedRequest.instructions;
      } else {
        // 否则直接设置系统提示
        modifiedRequest.instructions = systemPrompt;
      }
    }

    // 处理reasoning字段
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === 'auto') {
      // Auto模式：保持原始请求的reasoning字段不变
      // 如果原始请求有reasoning字段就保留，没有就不添加
    } else if (reasoningLevel && ['low', 'medium', 'high'].includes(reasoningLevel)) {
      modifiedRequest.reasoning = {
        effort: reasoningLevel,
        summary: 'auto'
      };
    } else {
      // 如果配置是off或无效，移除reasoning字段
      delete modifiedRequest.reasoning;
    }

    logRequest('POST', endpoint.base_url, headers, modifiedRequest);

    // 转发修改后的请求
    const response = await fetch(endpoint.base_url, {
      method: 'POST',
      headers,
      body: JSON.stringify(modifiedRequest)
    });

    logInfo(`Response status: ${response.status}`);
    
    // Record request result (2xx = success)
    const isSuccess = response.status >= 200 && response.status < 300;
    recordRequestResult(endpoint.base_url, isSuccess, response.status);

    if (!response.ok) {
      const errorText = await response.text();
      logError(`Endpoint error: ${response.status}`, new Error(errorText));
      return res.status(response.status).json({ 
        error: `Endpoint returned ${response.status}`,
        details: errorText 
      });
    }

    const isStreaming = openaiRequest.stream === true;

    if (isStreaming) {
      // 直接转发流式响应，不做任何转换
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      try {
        // 直接将原始响应流转发给客户端
        for await (const chunk of response.body) {
          res.write(chunk);
        }
        res.end();
        logInfo('Stream forwarded successfully');
      } catch (streamError) {
        logError('Stream error', streamError);
        res.end();
      }
    } else {
      // 直接转发非流式响应，不做任何转换
      const data = await response.json();
      logResponse(200, null, data);
      res.json(data);
    }

  } catch (error) {
    logError('Error in /v1/responses', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}

// 直接转发 Anthropic 请求（不做格式转换）
async function handleDirectMessages(req, res) {
  logInfo('POST /v1/messages');
  
  try {
    const anthropicRequest = req.body;
    const modelId = anthropicRequest.model;

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    // 只允许 anthropic 类型端点
    if (model.type !== 'anthropic') {
      return res.status(400).json({ 
        error: 'Invalid endpoint type',
        message: `/v1/messages 接口只支持 anthropic 类型端点，当前模型 ${modelId} 是 ${model.type} 类型`
      });
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Direct forwarding to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key - support client x-api-key for anthropic endpoint
    let authHeader;
    try {
      const clientAuthFromXApiKey = req.headers['x-api-key']
        ? `Bearer ${req.headers['x-api-key']}`
        : null;
      const clientUpstreamAuth = req.headers['x-endpoint-authorization'] || null;
      authHeader = await getApiKey(clientUpstreamAuth || clientAuthFromXApiKey);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    const clientHeaders = req.headers;
    
    // 获取 headers
    const isStreaming = anthropicRequest.stream === true;
    const headers = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, modelId);

    // 注入系统提示到 system 字段
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...anthropicRequest };
    if (systemPrompt) {
      if (modifiedRequest.system && Array.isArray(modifiedRequest.system)) {
        // 如果已有 system 数组，则在最前面插入系统提示
        modifiedRequest.system = [
          { type: 'text', text: systemPrompt },
          ...modifiedRequest.system
        ];
      } else {
        // 否则创建新的 system 数组
        modifiedRequest.system = [
          { type: 'text', text: systemPrompt }
        ];
      }
    }

    // 处理thinking字段
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === 'auto') {
      // Auto模式：保持原始请求的thinking字段不变
      // 如果原始请求有thinking字段就保留，没有就不添加
    } else if (reasoningLevel && ['low', 'medium', 'high'].includes(reasoningLevel)) {
      const budgetTokens = {
        'low': 4096,
        'medium': 12288,
        'high': 24576
      };
      
      modifiedRequest.thinking = {
        type: 'enabled',
        budget_tokens: budgetTokens[reasoningLevel]
      };
    } else {
      // 如果配置是off或无效，移除thinking字段
      delete modifiedRequest.thinking;
    }

    logRequest('POST', endpoint.base_url, headers, modifiedRequest);

    // 转发修改后的请求
    const response = await fetch(endpoint.base_url, {
      method: 'POST',
      headers,
      body: JSON.stringify(modifiedRequest)
    });

    logInfo(`Response status: ${response.status}`);
    
    // Record request result (2xx = success)
    const isSuccess = response.status >= 200 && response.status < 300;
    recordRequestResult(endpoint.base_url, isSuccess, response.status);

    if (!response.ok) {
      const errorText = await response.text();
      logError(`Endpoint error: ${response.status}`, new Error(errorText));
      return res.status(response.status).json({ 
        error: `Endpoint returned ${response.status}`,
        details: errorText 
      });
    }

    if (isStreaming) {
      // 直接转发流式响应，不做任何转换
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      try {
        // 直接将原始响应流转发给客户端
        for await (const chunk of response.body) {
          res.write(chunk);
        }
        res.end();
        logInfo('Stream forwarded successfully');
      } catch (streamError) {
        logError('Stream error', streamError);
        res.end();
      }
    } else {
      // 直接转发非流式响应，不做任何转换
      const data = await response.json();
      logResponse(200, null, data);
      res.json(data);
    }

  } catch (error) {
    logError('Error in /v1/messages', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}

// Status接口 - 展示统计信息
router.get('/status', (req, res) => {
  logInfo('GET /status');
  
  try {
    // First-visit setup: if no server key yet, present setup form
    if (!isServerKeySet()) {
      return res.send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <title>droid2api Status - Setup</title>
          <style>
            body { font-family: Arial, sans-serif; max-width: 720px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .warn { background: #fff8e1; border-left: 4px solid #FFC107; padding: 12px; margin-bottom: 16px; }
            label { display: block; margin: 12px 0 6px; color: #555; }
            input[type="password"], input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; }
            button { margin-top: 16px; background: #4CAF50; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; }
            button:hover { background: #43A047; }
            .hint { color: #666; font-size: 0.9em; margin-top: 8px; }
            code { background: #eee; padding: 2px 6px; border-radius: 3px; }
          </style>
        </head>
        <body>
          <h1>droid2api v2.0.0 Status</h1>
          <div class="card">
            <div class="warn">
              <p><strong>First-time setup:</strong> Set a server access key. After saving, all endpoints require this key via header <code>Authorization: Bearer &lt;key&gt;</code>. The /status page remains open.</p>
            </div>
            <form method="POST" action="/status/set-key">
              <label for="key">Server Access Key</label>
              <input type="password" id="key" name="key" placeholder="Enter a strong key" required />
              <div class="hint">Remember this key. It will be stored locally on the server.</div>
              <button type="submit">Save Key</button>
            </form>
          </div>
        </body>
        </html>
      `);
    }

    const keyManager = getKeyManager();
    
    if (!keyManager) {
      // 如果没有使用KeyManager（例如使用refresh token或client auth）
      return res.send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <title>droid2api Status</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              max-width: 1600px;
              margin: 50px auto;
              padding: 20px;
              background-color: #f5f5f5;
            }
            h1 {
              color: #333;
              text-align: center;
            }
            .info {
              background: white;
              padding: 20px;
              border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              text-align: center;
            }
          </style>
        </head>
        <body>
          <h1>droid2api v2.0.0 Status</h1>
          <div class="info">
            <p>Multi-key statistics are not available.</p>
            <p>This feature is only enabled when using FACTORY_API_KEY or factory_keys.txt with multiple keys.</p>
          </div>
        </body>
        </html>
      `);
    }
    
    const stats = keyManager.getStats();
    
    // 生成HTML页面
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>droid2api Status</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 1600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
          }
          h1 {
            color: #333;
            text-align: center;
          }
          .notice { margin: 10px 0 20px; padding: 10px; background: #e8f5e9; border-left: 4px solid #4CAF50; }
          .section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
          h2 {
            color: #555;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
          }
          .info {
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #2196F3;
          }
          table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
          }
          th, td { vertical-align: middle; }
          .btn { background: #4CAF50; color: #fff; border: none; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
          .btn:hover { filter: brightness(0.95); }
          .btn-refresh { background: #2196F3; }
          .btn-refresh:hover { background: #1976D2; }
          .btn { white-space: nowrap; display: inline-block; }
          .btn[disabled] { opacity: 0.55; cursor: not-allowed; }
          .controls { display: flex; gap: 10px; align-items: center; margin: 8px 0 12px; flex-wrap: wrap; }
          .controls label { color: #444; }
          .controls input[type="number"] { width: 80px; padding: 6px; border: 1px solid #ccc; border-radius: 6px; }
          .number { text-align: right; font-variant-numeric: tabular-nums; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Microsoft YaHei', sans-serif; letter-spacing: normal; }
          .date { color: #555; font-family: monospace; }
          .nowrap { white-space: nowrap; }
          code { background: #f1f3f5; padding: 6px 10px; border-radius: 6px; display: inline-block; }
          tr.depleted td { opacity: 0.7; background: #f8f9fa; }
          th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            white-space: nowrap;
          }
          td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
          }
          tr:hover {
            background-color: #f5f5f5;
          }
          .success {
            color: #4CAF50;
            font-weight: bold;
          }
          .fail {
            color: #f44336;
            font-weight: bold;
          }
          .rate {
            font-weight: bold;
            color: #2196F3;
          }
          code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
          }
        </style>
        <script>
          let refreshTimer = null;
          let refreshInterval = 30; // Default: 30 seconds
          
          // 页面加载时恢复之前的设置
          window.addEventListener('DOMContentLoaded', function() {
            // 读取保存的设置
            const savedAutoRefresh = localStorage.getItem('autoRefresh');
            const savedInterval = localStorage.getItem('refreshInterval');
            
            // 恢复刷新间隔选择
            if (savedInterval) {
              refreshInterval = parseInt(savedInterval);
              const select = document.getElementById('refreshInterval');
              select.value = savedInterval;
            }
            
            // 恢复自动刷新开关
            if (savedAutoRefresh === 'true') {
              const checkbox = document.getElementById('autoRefresh');
              checkbox.checked = true;
              startAutoRefresh();
            }
          });
          
          function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            // 保存开关状态
            localStorage.setItem('autoRefresh', checkbox.checked);
            
            if (checkbox.checked) {
              startAutoRefresh();
            } else {
              stopAutoRefresh();
            }
          }
          
          function updateRefreshInterval() {
            const select = document.getElementById('refreshInterval');
            refreshInterval = parseInt(select.value);
            // 保存刷新间隔
            localStorage.setItem('refreshInterval', select.value);
            
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
              stopAutoRefresh();
              startAutoRefresh();
            }
          }
          
          function startAutoRefresh() {
            refreshTimer = setInterval(() => {
              location.reload();
            }, refreshInterval * 1000);
          }
          
          function stopAutoRefresh() {
            if (refreshTimer) {
              clearInterval(refreshTimer);
              refreshTimer = null;
            }
          }
        </script>
      </head>
      <body>
        <h1>droid2api v2.0.0 Status</h1>
        
        <div class="section">
          <p style="text-align: center; color: #888; margin: 0;">
            Last updated: <span id="updateTime">${new Date().toLocaleString()}</span>
          </p>
          <div style="text-align: center; margin-top: 10px;">
            <label style="margin-right: 20px;">
              <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> 
              Auto Refresh
            </label>
            <label>
              Interval: 
              <select id="refreshInterval" onchange="updateRefreshInterval()">
                <option value="5">5 seconds</option>
                <option value="10">10 seconds</option>
                <option value="30" selected>30 seconds</option>
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
                <option value="600">10 minutes</option>
              </select>
            </label>
          </div>
        </div>
        
        <div class="section">
          <h2>Configuration</h2>
          <div class="info">
            <div class="notice">
              <p><strong>Access Control:</strong> Server key is set. Provide header <code>Authorization: Bearer &lt;key&gt;</code> for all endpoints except <code>/status</code>.</p>
            </div>
            <p><strong>Round-Robin Algorithm:</strong> <code>${stats.algorithm}</code></p>
            <p><strong>Remove on 402:</strong> <code>${stats.removeOn402 ? 'Enabled' : 'Disabled'}</code></p>
            <p><strong>Active Keys:</strong> ${stats.keys.length}</p>
            <p><strong>Deprecated Keys:</strong> ${stats.deprecatedKeys ? stats.deprecatedKeys.length : 0}</p>
            <p><strong>Client-Auth (optional):</strong> If you rely on client-provided upstream tokens, send them via header <code>X-Endpoint-Authorization: Bearer &lt;upstream-token&gt;</code>. The <code>Authorization</code> header is reserved for the server access key.</p>
          </div>
        </div>
        
        ${stats.endpoints.length > 0 ? `
        <div class="section">
          <h2>Endpoint Statistics</h2>
          <table>
            <thead>
              <tr>
                <th>Endpoint</th>
                <th>Success</th>
                <th>Fail</th>
                <th>Total</th>
                <th>Success Rate</th>
              </tr>
            </thead>
            <tbody>
              ${stats.endpoints.map(ep => `
                <tr>
                  <td><code>${ep.endpoint}</code></td>
                  <td class="success">${ep.success}</td>
                  <td class="fail">${ep.fail}</td>
                  <td>${ep.total}</td>
                  <td class="rate">${ep.successRate}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        ` : ''}
        
        <div class="section">
          <h2>API Keys Statistics (Active)</h2>
          <div class="controls">
            <button id="refresh-all" class="btn btn-refresh" onclick="refreshAllBalances()">刷新所有余额</button>
            <label for="auto-toggle">自动刷新</label>
            <input type="checkbox" id="auto-toggle" />
            <label for="auto-mins">间隔(分钟)</label>
            <input type="number" id="auto-mins" min="1" step="1" placeholder="5" />
            <span id="auto-next" style="color:#666; font-size: 0.9em;">&nbsp;</span>
            <label for="skip-threshold" style="margin-left:10px;">跳过阈值(剩余≤)</label>
            <input type="number" id="skip-threshold" min="0" step="1" placeholder="0" />
            <label for="batch-size" style="margin-left:10px;">每批数量</label>
            <input type="number" id="batch-size" min="1" step="1" placeholder="5" />
            <label for="batch-delay" style="margin-left:10px;">批间隔(秒)</label>
            <input type="number" id="batch-delay" min="0" step="1" placeholder="1" />
          </div>
          <table>
            <thead>
              <tr>
                <th class="nowrap">Key</th>
                <th>Success</th>
                <th>Fail</th>
                <th>Total</th>
                <th>Success Rate</th>
                <th class="number">Usage %</th>
                <th class="number">Balance (Remaining / Total)</th>
                <th class="nowrap">Start</th>
                <th class="nowrap">End</th>
                <th class="nowrap">Updated</th>
                <th class="nowrap" style="width:120px;">Actions</th>
              </tr>
            </thead>
            <tbody>
              ${stats.keys.map(key => `
                <tr id="row-${key.index}" class="${key.depleted ? 'depleted' : ''}">
                  <td class="nowrap"><code>${key.key}${key.depleted ? '（已用尽）' : ''}</code></td>
                  <td class="success">${key.success}</td>
                  <td class="fail">${key.fail}</td>
                  <td>${key.total}</td>
                  <td class="rate">${key.successRate}</td>
                  <td id="usage-${key.index}" class="usage number">—</td>
                  <td id="balance-${key.index}" class="balance number">—</td>
                  <td id="start-${key.index}" class="date">—</td>
                  <td id="end-${key.index}" class="date">—</td>
                  <td id="updated-${key.index}" class="updated">—</td>
                  <td>
                    <button id="btn-${key.index}" class="btn btn-refresh" onclick="refreshBalance(${key.index})" ${key.depleted ? 'disabled' : ''}>刷新余额</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          <script>
            function formatNumber(num){
              if(num===undefined||num===null||Number.isNaN(num)) return '0';
              try{ return new Intl.NumberFormat('en-US').format(num);}catch(e){return String(num)}
            }
            function formatPercentage(ratio){
              if(ratio===undefined||ratio===null||Number.isNaN(ratio)) return '0.00%';
              try{ return (ratio*100).toFixed(2)+'%'; }catch(e){ return '0.00%'; }
            }
            let currentSkipThreshold = 0;
            async function refreshBalance(index){
              const bal = document.getElementById('balance-'+index);
              const upd = document.getElementById('updated-'+index);
              const usageCell = document.getElementById('usage-'+index);
              const startCell = document.getElementById('start-'+index);
              const endCell = document.getElementById('end-'+index);
              const row = document.getElementById('row-'+index);
              const btn = document.getElementById('btn-'+index);
              if(!bal){return}
              const prev = bal.textContent;
              bal.textContent = '加载中...';
              try{
                const resp = await fetch('/status/balance/'+index+'?t='+(Date.now()));
                const data = await resp.json();
                if(!resp.ok || data.error){ throw new Error(data.error || ('HTTP '+resp.status)); }
                const total = Number(data.totalAllowance||0);
                const used  = Number(data.used||0);
                const remaining = Math.max(0, total - used);
                bal.textContent = formatNumber(remaining) + ' / ' + formatNumber(total);
                if(usageCell){ usageCell.textContent = formatPercentage(total>0 ? Math.min(1, used/total) : 0); }
                if(startCell){ startCell.textContent = data.startDate ? new Date(data.startDate).toLocaleString() : '—'; }
                if(endCell){ endCell.textContent = data.endDate ? new Date(data.endDate).toLocaleString() : '—'; }
                if(upd){ upd.textContent = new Date(data.fetchedAt||Date.now()).toLocaleString(); }
                const dep = remaining <= currentSkipThreshold;
                if(row){ row.classList.toggle('depleted', dep); }
                if(btn){ btn.disabled = dep; btn.title = dep ? '已用尽，轮巡中跳过' : ''; }
              }catch(err){
                bal.textContent = '错误';
                if(usageCell){ usageCell.textContent = '—'; }
                if(startCell){ startCell.textContent = '—'; }
                if(endCell){ endCell.textContent = '—'; }
                if(upd){ upd.textContent = err.message; }
              }
            }
            // 批量刷新设置
            function getBatchSettings(){
              const size = Math.max(1, Number(localStorage.getItem('batch_size') || 5));
              const delayMs = Math.max(0, Number(localStorage.getItem('batch_delay_ms') || 1000));
              return { size, delayMs };
            }
            function saveBatchSettings(size, delaySec){
              const nSize = Math.max(1, Number(size) || 1);
              const nDelayMs = Math.max(0, Math.round((Number(delaySec) || 0) * 1000));
              localStorage.setItem('batch_size', String(nSize));
              localStorage.setItem('batch_delay_ms', String(nDelayMs));
            }
            function initBatchControls(){
              const sizeInput = document.getElementById('batch-size');
              const delayInput = document.getElementById('batch-delay');
              const { size, delayMs } = getBatchSettings();
              if(sizeInput){ sizeInput.value = String(size); sizeInput.addEventListener('change', () => {
                  const v = Math.max(1, Number(sizeInput.value)||1);
                  sizeInput.value = String(v);
                  saveBatchSettings(v, (document.getElementById('batch-delay')?.value)||Math.round(delayMs/1000));
                });
              }
              if(delayInput){ delayInput.value = String(Math.round(delayMs/1000)); delayInput.addEventListener('change', () => {
                  const v = Math.max(0, Number(delayInput.value)||0);
                  delayInput.value = String(v);
                  saveBatchSettings((document.getElementById('batch-size')?.value)||size, v);
                });
              }
            }

            let isRefreshingAll = false;
            async function refreshAllBalances(){
              if(isRefreshingAll){ return; }
              isRefreshingAll = true;
              try{
                const rows = Array.from(document.querySelectorAll('tr[id^="row-"]'));
                const indices = rows
                  .map(r => { const i = Number((r.id||'').replace('row-','')); return Number.isFinite(i) ? i : null; })
                  .filter(i => i !== null);
                const { size, delayMs } = getBatchSettings();
                for(let i = 0; i < indices.length; i += size){
                  const batch = indices.slice(i, i + size);
                  await Promise.all(batch.map(idx => refreshBalance(idx)));
                  if(i + size < indices.length && delayMs > 0){
                    await new Promise(r => setTimeout(r, delayMs));
                  }
                }
              } catch(err) {
                // ignore global error; per-row buttons still work
              } finally {
                isRefreshingAll = false;
              }
            }
            // 自动刷新设置
            let autoTimer = null;
            function getAutoSettings(){
              const enabled = localStorage.getItem('auto_enabled') === '1';
              const ms = Number(localStorage.getItem('auto_ms') || 300000); // 默认5分钟
              return { enabled, ms };
            }
            function saveAutoSettings(enabled, ms){
              localStorage.setItem('auto_enabled', enabled ? '1' : '0');
              if(ms && ms > 0){ localStorage.setItem('auto_ms', String(ms)); }
            }
            function applyAutoRefresh(){
              const hint = document.getElementById('auto-next');
              if(autoTimer){ clearInterval(autoTimer); autoTimer = null; }
              const { enabled, ms } = getAutoSettings();
              if(enabled && ms > 0){
                autoTimer = setInterval(() => {
                  refreshAllBalances();
                }, ms);
                if(hint){ hint.textContent = '每 ' + Math.round(ms/60000) + ' 分钟自动刷新'; }
              } else {
                if(hint){ hint.textContent = ''; }
              }
            }
            function initAutoControls(){
              const toggle = document.getElementById('auto-toggle');
              const minsInput = document.getElementById('auto-mins');
              const { enabled, ms } = getAutoSettings();
              if(toggle){ toggle.checked = !!enabled; toggle.addEventListener('change', () => {
                  const en = toggle.checked;
                  const mins = Math.max(1, Number(minsInput.value || Math.round(ms/60000)));
                  saveAutoSettings(en, mins*60000);
                  applyAutoRefresh();
                });
              }
              if(minsInput){ minsInput.value = Math.max(1, Math.round(ms/60000)); minsInput.addEventListener('change', () => {
                  const mins = Math.max(1, Number(minsInput.value||1));
                  const en = toggle ? toggle.checked : true;
                  saveAutoSettings(en, mins*60000);
                  applyAutoRefresh();
                });
              }
              applyAutoRefresh();
            }
            // 阈值控制：与服务器同步
            async function fetchSkipThreshold(){
              try {
                const resp = await fetch('/status/skip-threshold?t='+(Date.now()));
                const data = await resp.json();
                if(resp.ok && typeof data.skipThreshold === 'number'){
                  currentSkipThreshold = data.skipThreshold;
                }
              } catch(e) {}
            }
            async function updateSkipThreshold(n){
              currentSkipThreshold = Math.max(0, Number(n)||0);
              try {
                await fetch('/status/skip-threshold', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skipThreshold: currentSkipThreshold }) });
              } catch(e) {}
            }
            function initThresholdControls(){
              const input = document.getElementById('skip-threshold');
              fetchSkipThreshold().then(() => { if(input){ input.value = String(currentSkipThreshold); }});
              if(input){ input.addEventListener('change', () => {
                  const v = Math.max(0, Number(input.value)||0);
                  input.value = String(v);
                  updateSkipThreshold(v).then(() => refreshAllBalances());
                });
              }
            }
            // 首次进入：不自动刷新余额，仅初始化控件
            window.addEventListener('load', () => { 
              initAutoControls();
              initThresholdControls();
              initBatchControls();
            });
          </script>
        </div>
        
        ${stats.deprecatedKeys && stats.deprecatedKeys.length > 0 ? `
        <div class="section">
          <h2>Deprecated Keys (Removed due to 402)</h2>
          <table>
            <thead>
              <tr>
                <th>Key</th>
                <th>Success</th>
                <th>Fail</th>
                <th>Total</th>
                <th>Success Rate</th>
                <th>Deprecated At</th>
              </tr>
            </thead>
            <tbody>
              ${stats.deprecatedKeys.map(key => `
                <tr style="background-color: #fff3cd;">
                  <td><code>${key.key}</code></td>
                  <td class="success">${key.success}</td>
                  <td class="fail">${key.fail}</td>
                  <td>${key.total}</td>
                  <td class="rate">${key.successRate}</td>
                  <td>${new Date(key.deprecatedAt).toLocaleString()}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        ` : ''}
      </body>
      </html>
    `;
    
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.send(html);
    
  } catch (error) {
    logError('Error in GET /status', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

// First-time key setup endpoint (only works before key is set)
router.post('/status/set-key', (req, res) => {
  logInfo('POST /status/set-key');
  try {
    if (isServerKeySet()) {
      return res.status(400).send(`
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><title>Key Already Set</title></head>
        <body>
          <p>Server key has already been set. Go back to <a href="/status">/status</a>.</p>
        </body>
        </html>
      `);
    }
    const key = (req.body?.key || '').toString();
    if (!key || key.trim() === '') {
      return res.status(400).send('Key is required');
    }
    setServerKey(key);
    return res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>Key Saved</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 720px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
          .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
          code { background: #eee; padding: 2px 6px; border-radius: 3px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h3>Server key saved successfully.</h3>
          <p>Use header <code>Authorization: Bearer &lt;key&gt;</code> for all API requests (except <code>/status</code>).</p>
          <p><a href="/status">Go to Status</a></p>
        </div>
      </body>
      </html>
    `);
  } catch (error) {
    logError('Error in POST /status/set-key', error);
    res.status(500).send('Failed to set key');
  }
});

// 注册路由
router.post('/v1/chat/completions', handleChatCompletions);
router.post('/v1/responses', handleDirectResponses);
router.post('/v1/messages', handleDirectMessages);

export default router;

// ----------------------
// Balance query endpoints
// ----------------------
// Helper to fetch usage/balance for a single key
async function fetchUsageForKeyRaw(apiKey) {
  try {
    const url = 'https://app.factory.ai/api/organization/members/chat-usage';
    const configUA = (() => { try { return getConfig()?.user_agent; } catch { return null; } })();
    const headers = {
      'Authorization': `Bearer ${apiKey}`,
      'User-Agent': configUA || 'factory-cli/0.19.3'
    };
    const resp = await fetch(url, { method: 'GET', headers });
    if (!resp.ok) {
      return { error: `HTTP ${resp.status}` };
    }
    const data = await resp.json();
    const usage = data?.usage;
    const standard = usage?.standard;
    if (!standard) {
      return { error: 'Invalid API response structure' };
    }
    const totalAllowance = Number(standard.totalAllowance || 0);
    const used = Number(standard.orgTotalTokensUsed || 0);
    const usedRatio = typeof standard.usedRatio === 'number' ? standard.usedRatio : (totalAllowance > 0 ? used / totalAllowance : 0);
    const startDate = usage?.startDate ? new Date(usage.startDate).toISOString() : null;
    const endDate = usage?.endDate ? new Date(usage.endDate).toISOString() : null;
    return {
      totalAllowance,
      used,
      usedRatio,
      startDate,
      endDate
    };
  } catch (e) {
    return { error: 'Failed to fetch' };
  }
}

// GET /status/balance/:index - fetch balance for specific key by index
router.get('/status/balance/:index', async (req, res) => {
  try {
    const keyManager = getKeyManager();
    if (!keyManager) {
      return res.status(400).json({ error: 'Multi-key not configured' });
    }
    const idx = parseInt(req.params.index, 10);
    if (Number.isNaN(idx) || idx < 0 || idx >= keyManager.keys.length) {
      return res.status(400).json({ error: 'Invalid index' });
    }
    const keyObj = keyManager.keys[idx];
    if (!keyObj || keyObj.deprecated) {
      return res.status(404).json({ error: 'Key not active' });
    }
    const usage = await fetchUsageForKeyRaw(keyObj.key);
    if (usage.error) {
      return res.status(200).json({ index: idx, maskedKey: keyManager.maskKey(keyObj.key), error: usage.error, fetchedAt: new Date().toISOString() });
    }
    const total = Number(usage.totalAllowance||0);
    const used = Number(usage.used||0);
    const remaining = Math.max(0, total - used);
    // 同步到 KeyManager，余额用尽则在轮询中跳过
    try { keyManager.setBalanceByIndex(idx, { totalAllowance: total, used, remaining, fetchedAt: new Date().toISOString() }); } catch {}
    return res.json({ index: idx, maskedKey: keyManager.maskKey(keyObj.key), ...usage, fetchedAt: new Date().toISOString() });
  } catch (error) {
    logError('Error in GET /status/balance/:index', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /status/balances - fetch balances for all active keys
router.get('/status/balances', async (req, res) => {
  try {
    const keyManager = getKeyManager();
    if (!keyManager) {
      return res.status(400).json({ error: 'Multi-key not configured' });
    }
    const tasks = keyManager.keys
      .map((k, i) => ({ k, i }))
      .filter(x => !x.k.deprecated)
      .map(async ({ k, i }) => {
        const usage = await fetchUsageForKeyRaw(k.key);
        if (usage.error) {
          return { index: i, maskedKey: keyManager.maskKey(k.key), error: usage.error, fetchedAt: new Date().toISOString() };
        }
        const total = Number(usage.totalAllowance||0);
        const used = Number(usage.used||0);
        const remaining = Math.max(0, total - used);
        try { keyManager.setBalanceByIndex(i, { totalAllowance: total, used, remaining, fetchedAt: new Date().toISOString() }); } catch {}
        return { index: i, maskedKey: keyManager.maskKey(k.key), ...usage, fetchedAt: new Date().toISOString() };
      });
    const list = await Promise.all(tasks);
    return res.json(list);
  } catch (error) {
    logError('Error in GET /status/balances', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// Skip-threshold settings (in-memory)
router.get('/status/skip-threshold', (req, res) => {
  try {
    const km = getKeyManager();
    const value = km ? km.getSkipThreshold?.() ?? 0 : 0;
    res.json({ skipThreshold: value });
  } catch (e) {
    res.status(200).json({ skipThreshold: 0 });
  }
});

router.post('/status/skip-threshold', express.json(), (req, res) => {
  try {
    const km = getKeyManager();
    if (!km) return res.status(400).json({ error: 'Multi-key not configured' });
    const body = req.body || {};
    const n = body.skipThreshold ?? body.n ?? body.remaining_threshold ?? 0;
    km.setSkipThreshold?.(Number(n) || 0);
    res.json({ ok: true, skipThreshold: km.getSkipThreshold?.() ?? 0 });
  } catch (e) {
    res.status(500).json({ error: 'Failed to set threshold' });
  }
});
