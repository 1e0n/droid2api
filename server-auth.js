import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { logInfo, logError } from './logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const KEY_FILE = path.join(__dirname, 'server-key.json');

let cachedKey = null;

function loadKeyFromDisk() {
  try {
    if (fs.existsSync(KEY_FILE)) {
      const raw = fs.readFileSync(KEY_FILE, 'utf-8');
      const data = JSON.parse(raw);
      if (data && typeof data.key === 'string' && data.key.trim() !== '') {
        cachedKey = data.key.trim();
        return cachedKey;
      }
    }
  } catch (err) {
    logError('Failed to read server key from disk', err);
  }
  return null;
}

export function isServerKeySet() {
  if (cachedKey && cachedKey.length > 0) return true;
  const key = loadKeyFromDisk();
  return !!(key && key.length > 0);
}

export function setServerKey(key) {
  if (isServerKeySet()) {
    throw new Error('Server key already set');
  }
  if (!key || typeof key !== 'string' || key.trim() === '') {
    throw new Error('Invalid key');
  }
  const normalized = key.trim();
  try {
    fs.writeFileSync(KEY_FILE, JSON.stringify({ key: normalized }, null, 2), 'utf-8');
    cachedKey = normalized;
    logInfo('Server key has been set successfully');
  } catch (err) {
    logError('Failed to write server key to disk', err);
    throw err;
  }
}

export function verifyServerKey(provided) {
  if (!provided || typeof provided !== 'string') return false;
  const expected = cachedKey || loadKeyFromDisk();
  if (!expected) return false;
  return expected === provided.trim();
}

export function serverAuthMiddleware(req, res, next) {
  // Allow status and its subpaths without key
  const path = req.path || req.originalUrl || '';
  if (path === '/status' || path.startsWith('/status/')) {
    return next();
  }

  // If key is not set yet, block all other routes and instruct to visit /status
  if (!isServerKeySet()) {
    return res.status(503).json({
      error: 'Server key not set',
      message: 'Visit /status to set the initial access key.'
    });
  }

  // Accept key via header or query
  const provided = req.headers['x-server-key'] || req.query.key || req.query.server_key;
  if (!verifyServerKey(typeof provided === 'string' ? provided : '')) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Missing or invalid X-Server-Key (or ?key=)'
    });
  }

  return next();
}

