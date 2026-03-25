import { logDebug } from '../logger.js';
import { getSystemPrompt, getUserAgent, getModelReasoning } from '../config.js';

export function transformToGoogle(openaiRequest) {
  logDebug('Transforming OpenAI request to Google format');

  const googleRequest = {
    model: openaiRequest.model,
    contents: []
  };

  // Collect system parts: config system prompt first, then user system messages
  let systemParts = [];
  const systemPrompt = getSystemPrompt();

  if (systemPrompt) {
    systemParts.push({ text: systemPrompt });
  }

  // Transform messages to contents
  if (openaiRequest.messages && Array.isArray(openaiRequest.messages)) {
    for (const msg of openaiRequest.messages) {
      if (msg.role === 'system') {
        if (typeof msg.content === 'string') {
          systemParts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              systemParts.push({ text: part.text });
            }
          }
        }
        continue;
      }

      // Map OpenAI "assistant" -> Google "model"
      const googleRole = msg.role === 'assistant' ? 'model' : msg.role;
      const googleMsg = {
        role: googleRole,
        parts: []
      };

      if (typeof msg.content === 'string') {
        googleMsg.parts.push({ text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === 'text') {
            googleMsg.parts.push({ text: part.text });
          } else if (part.type === 'image_url') {
            googleMsg.parts.push({
              inlineData: {
                mimeType: part.image_url.type || 'image/jpeg',
                data: part.image_url.url
              }
            });
          } else {
            googleMsg.parts.push(part);
          }
        }
      }

      googleRequest.contents.push(googleMsg);
    }
  }

  if (systemParts.length > 0) {
    googleRequest.systemInstruction = {
      parts: systemParts
    };
  }

  // Build generationConfig
  const generationConfig = {};

  if (openaiRequest.max_tokens) {
    generationConfig.maxOutputTokens = openaiRequest.max_tokens;
  } else if (openaiRequest.max_completion_tokens) {
    generationConfig.maxOutputTokens = openaiRequest.max_completion_tokens;
  }

  if (openaiRequest.temperature !== undefined) {
    generationConfig.temperature = openaiRequest.temperature;
  }
  if (openaiRequest.top_p !== undefined) {
    generationConfig.topP = openaiRequest.top_p;
  }
  if (openaiRequest.stop !== undefined) {
    generationConfig.stopSequences = Array.isArray(openaiRequest.stop)
      ? openaiRequest.stop
      : [openaiRequest.stop];
  }
  if (openaiRequest.presence_penalty !== undefined) {
    generationConfig.presencePenalty = openaiRequest.presence_penalty;
  }
  if (openaiRequest.frequency_penalty !== undefined) {
    generationConfig.frequencyPenalty = openaiRequest.frequency_penalty;
  }

  // Handle reasoning/thinking config via thinkingLevel
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === 'auto') {
    // 保持原始请求的 thinkingConfig 不变
  } else if (reasoningLevel && ['low', 'medium', 'high'].includes(reasoningLevel)) {
    const levelMap = { 'low': 'LOW', 'medium': 'MEDIUM', 'high': 'HIGH' };
    if (!generationConfig.thinkingConfig) {
      generationConfig.thinkingConfig = {};
    }
    generationConfig.thinkingConfig.thinkingLevel = levelMap[reasoningLevel];
  } else {
    // off 或无效：移除 thinkingConfig
    delete generationConfig.thinkingConfig;
  }

  if (Object.keys(generationConfig).length > 0) {
    googleRequest.generationConfig = generationConfig;
  }

  // Transform tools if present
  if (openaiRequest.tools && Array.isArray(openaiRequest.tools)) {
    googleRequest.tools = [{
      functionDeclarations: openaiRequest.tools
        .filter(tool => tool.type === 'function')
        .map(tool => ({
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters || {}
        }))
    }];
  }

  logDebug('Transformed Google request', googleRequest);
  return googleRequest;
}

export function getGoogleHeaders(authHeader, clientHeaders = {}, provider = 'google') {
  const sessionId = clientHeaders['x-session-id'] || generateUUID();
  const messageId = clientHeaders['x-assistant-message-id'] || generateUUID();

  const userAgent = getUserAgent();
  const versionMatch = userAgent.match(/\/(\d+\.\d+\.\d+)/);
  const clientVersion = versionMatch ? versionMatch[1] : '0.84.0';

  const headers = {
    'accept': '*/*',
    'content-type': 'application/json',
    'authorization': authHeader || '',
    'user-agent': userAgent,
    'x-client-version': clientVersion,
    'x-factory-client': clientHeaders['x-factory-client'] || 'cli',
    'x-api-provider': provider,
    'x-assistant-message-id': messageId,
    'x-session-id': sessionId,
    'connection': 'keep-alive'
  };

  return headers;
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
