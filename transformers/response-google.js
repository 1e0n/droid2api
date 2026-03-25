import { logDebug } from '../logger.js';

export class GoogleResponseTransformer {
  constructor(model, requestId) {
    this.model = model;
    this.requestId = requestId || `chatcmpl-${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
    this.sentRole = false;
  }

  parseSSELine(line) {
    if (line.startsWith('data:')) {
      const dataStr = line.slice(5).trim();
      try {
        return { type: 'data', value: JSON.parse(dataStr) };
      } catch (e) {
        return { type: 'data', value: dataStr };
      }
    }
    return null;
  }

  transformEvent(eventData) {
    if (!eventData || typeof eventData !== 'object') return null;

    const candidate = eventData.candidates?.[0];
    if (!candidate) return null;

    const parts = candidate.content?.parts;
    if (!parts || parts.length === 0) {
      // Final chunk with finishReason but no new text
      if (candidate.finishReason) {
        const finishReason = this.mapFinishReason(candidate.finishReason);
        const finalChunk = this.createOpenAIChunk('', null, true, finishReason);
        const usage = this.transformUsage(eventData.usageMetadata);
        if (usage) {
          // Parse the chunk to inject usage
          const parsed = JSON.parse(finalChunk.slice(6, -2)); // strip "data: " and "\n\n"
          parsed.usage = usage;
          return `data: ${JSON.stringify(parsed)}\n\n` + this.createDoneSignal();
        }
        return finalChunk + this.createDoneSignal();
      }
      return null;
    }

    let results = '';

    for (const part of parts) {
      // Skip thinking/thought parts
      if (part.thought === true) continue;

      const text = part.text || '';
      if (!text && !candidate.finishReason) continue;

      // Send role on first non-thought content
      if (!this.sentRole) {
        this.sentRole = true;
        results += this.createOpenAIChunk('', 'assistant', false);
      }

      if (text) {
        results += this.createOpenAIChunk(text, null, false);
      }
    }

    // Handle finishReason in the same chunk that has content
    if (candidate.finishReason) {
      const finishReason = this.mapFinishReason(candidate.finishReason);
      const usage = this.transformUsage(eventData.usageMetadata);
      const finalChunk = this.createOpenAIChunk('', null, true, finishReason);
      if (usage) {
        const parsed = JSON.parse(finalChunk.slice(6, -2));
        parsed.usage = usage;
        results += `data: ${JSON.stringify(parsed)}\n\n`;
      } else {
        results += finalChunk;
      }
      results += this.createDoneSignal();
    }

    return results || null;
  }

  transformUsage(usageMetadata) {
    if (!usageMetadata) return null;
    return {
      prompt_tokens: usageMetadata.promptTokenCount || 0,
      completion_tokens: usageMetadata.candidatesTokenCount || 0,
      total_tokens: usageMetadata.totalTokenCount || 0
    };
  }

  createOpenAIChunk(content, role = null, finish = false, finishReason = null) {
    const chunk = {
      id: this.requestId,
      object: 'chat.completion.chunk',
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: finish ? finishReason : null
        }
      ]
    };

    if (role) {
      chunk.choices[0].delta.role = role;
    }
    if (content) {
      chunk.choices[0].delta.content = content;
    }

    return `data: ${JSON.stringify(chunk)}\n\n`;
  }

  createDoneSignal() {
    return 'data: [DONE]\n\n';
  }

  mapFinishReason(googleReason) {
    const mapping = {
      'STOP': 'stop',
      'MAX_TOKENS': 'length',
      'SAFETY': 'content_filter',
      'RECITATION': 'content_filter',
      'FINISH_REASON_UNSPECIFIED': 'stop'
    };
    return mapping[googleReason] || 'stop';
  }

  async *transformStream(sourceStream) {
    let buffer = '';

    try {
      for await (const chunk of sourceStream) {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          const parsed = this.parseSSELine(line);
          if (!parsed || parsed.type !== 'data') continue;
          if (typeof parsed.value !== 'object') continue;

          const transformed = this.transformEvent(parsed.value);
          if (transformed) {
            yield transformed;
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        const parsed = this.parseSSELine(buffer.trim());
        if (parsed && parsed.type === 'data' && typeof parsed.value === 'object') {
          const transformed = this.transformEvent(parsed.value);
          if (transformed) {
            yield transformed;
          }
        }
      }
    } catch (error) {
      logDebug('Error in Google stream transformation', error);
      throw error;
    }
  }
}
