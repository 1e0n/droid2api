import { logDebug } from '../logger.js';
import { getSystemPrompt, getModelReasoning, getModelFast, getUserAgent } from '../config.js';

export function transformToAnthropic(openaiRequest) {
  logDebug('Transforming OpenAI request to Anthropic format');
  
  const anthropicRequest = {
    model: openaiRequest.model,
    messages: []
  };

  // Only add stream parameter if explicitly provided by client
  if (openaiRequest.stream !== undefined) {
    anthropicRequest.stream = openaiRequest.stream;
  }

  // Handle max_tokens
  if (openaiRequest.max_tokens) {
    anthropicRequest.max_tokens = openaiRequest.max_tokens;
  } else if (openaiRequest.max_completion_tokens) {
    anthropicRequest.max_tokens = openaiRequest.max_completion_tokens;
  } else {
    anthropicRequest.max_tokens = 4096;
  }

  // Extract system message(s) and transform other messages
  let systemContent = [];
  
  if (openaiRequest.messages && Array.isArray(openaiRequest.messages)) {
    for (const msg of openaiRequest.messages) {
      // Handle system messages separately
      if (msg.role === 'system') {
        if (typeof msg.content === 'string') {
          systemContent.push({
            type: 'text',
            text: msg.content
          });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              systemContent.push({
                type: 'text',
                text: part.text
              });
            } else {
              systemContent.push(part);
            }
          }
        }
        continue; // Skip adding system messages to messages array
      }

      const anthropicMsg = {
        role: msg.role,
        content: []
      };

      if (typeof msg.content === 'string') {
        anthropicMsg.content.push({
          type: 'text',
          text: msg.content
        });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === 'text') {
            anthropicMsg.content.push({
              type: 'text',
              text: part.text
            });
          } else if (part.type === 'image_url') {
            anthropicMsg.content.push({
              type: 'image',
              source: part.image_url
            });
          } else {
            anthropicMsg.content.push(part);
          }
        }
      }

      anthropicRequest.messages.push(anthropicMsg);
    }
  }

  // Add system parameter with system prompt prepended
  const systemPrompt = getSystemPrompt();
  if (systemPrompt || systemContent.length > 0) {
    anthropicRequest.system = [];
    // Prepend system prompt as first element if it exists
    if (systemPrompt) {
      anthropicRequest.system.push({
        type: 'text',
        text: systemPrompt
      });
    }
    // Add user-provided system content
    anthropicRequest.system.push(...systemContent);
  }

  // Transform tools if present
  if (openaiRequest.tools && Array.isArray(openaiRequest.tools)) {
    anthropicRequest.tools = openaiRequest.tools.map(tool => {
      if (tool.type === 'function') {
        return {
          name: tool.function.name,
          description: tool.function.description,
          input_schema: tool.function.parameters || {}
        };
      }
      return tool;
    });
  }

  // Handle thinking field based on model configuration
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === 'auto') {
    // Auto mode: preserve original request's thinking field exactly as-is
    if (openaiRequest.thinking !== undefined) {
      anthropicRequest.thinking = openaiRequest.thinking;
    }
    // If original request has no thinking field, don't add one
  } else if (reasoningLevel && ['low', 'medium', 'high', 'xhigh'].includes(reasoningLevel)) {
    // Specific level: override with model configuration
    const budgetTokens = {
      'low': 4096,
      'medium': 12288,
      'high': 24576,
      'xhigh': 24576
    };
    
    anthropicRequest.thinking = {
      type: 'enabled',
      budget_tokens: budgetTokens[reasoningLevel]
    };
  } else {
    // Off or invalid: explicitly remove thinking field
    // This ensures any thinking field from the original request is deleted
    delete anthropicRequest.thinking;
  }

  // Pass through other compatible parameters
  if (openaiRequest.temperature !== undefined) {
    anthropicRequest.temperature = openaiRequest.temperature;
  }
  if (openaiRequest.top_p !== undefined) {
    anthropicRequest.top_p = openaiRequest.top_p;
  }
  if (openaiRequest.stop !== undefined) {
    anthropicRequest.stop_sequences = Array.isArray(openaiRequest.stop) 
      ? openaiRequest.stop 
      : [openaiRequest.stop];
  }

  logDebug('Transformed Anthropic request', anthropicRequest);
  return anthropicRequest;
}

export function getAnthropicHeaders(authHeader, clientHeaders = {}, isStreaming = true, modelId = null, provider = 'anthropic') {
  // Generate unique IDs if not provided
  const sessionId = clientHeaders['x-session-id'] || generateUUID();
  const messageId = clientHeaders['x-assistant-message-id'] || generateUUID();
  
  const headers = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'anthropic-version': clientHeaders['anthropic-version'] || '2023-06-01',
    'authorization': authHeader || '',
    'x-api-key': 'placeholder',
    'x-api-provider': provider,
    'x-factory-client': 'cli',
    'x-session-id': sessionId,
    'x-assistant-message-id': messageId,
    'user-agent': getUserAgent(),
    'x-stainless-timeout': '600',
    'connection': 'keep-alive'
  }

  // Handle anthropic-beta header based on reasoning configuration
  const reasoningLevel = modelId ? getModelReasoning(modelId) : null;
  let betaValues = [];
  
  // Add existing beta values from client headers
  if (clientHeaders['anthropic-beta']) {
    const existingBeta = clientHeaders['anthropic-beta'];
    betaValues = existingBeta.split(',').map(v => v.trim());
  }
  
  // Handle thinking beta based on reasoning configuration
  const thinkingBeta = 'interleaved-thinking-2025-05-14';
  if (reasoningLevel === 'auto') {
    // Auto mode: don't modify anthropic-beta header, preserve original
    // betaValues remain unchanged from client headers
  } else if (reasoningLevel && ['low', 'medium', 'high', 'xhigh'].includes(reasoningLevel)) {
    // Add thinking beta if not already present
    if (!betaValues.includes(thinkingBeta)) {
      betaValues.push(thinkingBeta);
    }
  } else {
    // Remove thinking beta if reasoning is off/invalid
    betaValues = betaValues.filter(v => v !== thinkingBeta);
  }
  
  // Add fast-mode beta if model has fast enabled
  const fastModeBeta = 'fast-mode-2026-02-01';
  if (modelId && getModelFast(modelId)) {
    if (!betaValues.includes(fastModeBeta)) {
      betaValues.push(fastModeBeta);
    }
  }

  // Set anthropic-beta header if there are any values
  if (betaValues.length > 0) {
    headers['anthropic-beta'] = betaValues.join(', ');
  }

  // Pass through Stainless SDK headers with defaults
  const stainlessDefaults = {
    'x-stainless-arch': 'x64',
    'x-stainless-lang': 'js',
    'x-stainless-os': 'MacOS',
    'x-stainless-runtime': 'node',
    'x-stainless-retry-count': '0',
    'x-stainless-package-version': '0.57.0',
    'x-stainless-runtime-version': 'v24.3.0'
  };

  // Set helper-method based on streaming
  if (isStreaming) {
    headers['x-stainless-helper-method'] = 'stream';
  }

  // Copy Stainless headers from client or use defaults
  Object.keys(stainlessDefaults).forEach(header => {
    headers[header] = clientHeaders[header] || stainlessDefaults[header];
  });

  // Override timeout from defaults if client provided
  if (clientHeaders['x-stainless-timeout']) {
    headers['x-stainless-timeout'] = clientHeaders['x-stainless-timeout'];
  }

  return headers;
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
