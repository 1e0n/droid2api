import { logInfo, logDebug, logError } from './logger.js';

/**
 * Key Manager - 管理多个API key的选择和统计
 */
class KeyManager {
  constructor(keys, algorithm = 'weighted', removeOn402 = true) {
    // 初始化key数组，每个key有自己的统计信息
    this.keys = keys.map(key => ({
      key: key,
      success: 0,
      fail: 0,
      deprecated: false, // 是否被废弃
      depleted: false,   // 是否余额用尽
      balance: null      // 最近一次余额信息 { totalAllowance, used, remaining, fetchedAt }
    }));
    
    this.algorithm = algorithm; // 'weighted' or 'simple'
    this.simpleIndex = 0; // 用于simple算法的当前索引
    this.endpointStats = {}; // 端点统计 { endpoint: { success: 0, fail: 0 } }
    this.removeOn402 = removeOn402; // 是否在402时移除key
    this.deprecatedKeys = []; // 已废弃的key列表
    this.skipThreshold = 0; // 剩余≤阈值则视为用尽
    
    logInfo(`KeyManager initialized with ${this.keys.length} keys, algorithm: ${this.algorithm}, removeOn402: ${this.removeOn402}`);
  }
  
  /**
   * 根据算法选择一个key
   */
  selectKey() {
    // 获取未废弃的key
    const activeKeys = this.keys.filter(k => !k.deprecated && !k.depleted);
    
    if (activeKeys.length === 0) {
      throw new Error('No active keys available - all keys have been deprecated');
    }
    
    if (activeKeys.length === 1) {
      return activeKeys[0].key;
    }
    
    if (this.algorithm === 'simple') {
      return this.simpleSelect(activeKeys);
    } else {
      return this.weightedSelect(activeKeys);
    }
  }
  
  /**
   * 简单轮询算法 - 按顺序循环选择key
   */
  simpleSelect(activeKeys) {
    // 使用activeKeys数组进行轮询
    const key = activeKeys[this.simpleIndex % activeKeys.length].key;
    this.simpleIndex = (this.simpleIndex + 1) % activeKeys.length;
    logDebug(`Simple select: key index ${this.simpleIndex - 1 >= 0 ? this.simpleIndex - 1 : activeKeys.length - 1}`);
    return key;
  }
  
  /**
   * 基于健康度的加权轮询算法
   * 健康度 = 成功次数 / (成功次数 + 失败次数)
   * 使用加权随机选择，健康度高的key被选中概率更大
   */
  weightedSelect(activeKeys) {
    // 计算每个key的健康度权重
    const weights = activeKeys.map(keyObj => {
      const total = keyObj.success + keyObj.fail;
      if (total === 0) {
        // 没有历史记录的key给予默认权重1
        return 1;
      }
      // 健康度 = 成功率 + 0.1 (确保失败的key也有被选中的机会)
      const healthScore = (keyObj.success / total) + 0.1;
      return healthScore;
    });
    
    // 计算总权重
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    
    // 加权随机选择
    let random = Math.random() * totalWeight;
    for (let i = 0; i < weights.length; i++) {
      random -= weights[i];
      if (random <= 0) {
        logDebug(`Weighted select: key index ${i}, health: ${(weights[i] - 0.1).toFixed(2)}`);
        return activeKeys[i].key;
      }
    }
    
    // 兜底：返回最后一个活跃key
    return activeKeys[activeKeys.length - 1].key;
  }
  
  /**
   * 记录请求结果
   * @param {string} key - 使用的key
   * @param {string} endpoint - 端点URL
   * @param {boolean} success - 是否成功
   * @param {number} statusCode - HTTP状态码
   */
  recordResult(key, endpoint, success, statusCode = null) {
    // 记录key的统计
    const keyObj = this.keys.find(k => k.key === key);
    if (keyObj) {
      if (success) {
        keyObj.success++;
      } else {
        keyObj.fail++;
      }
      logDebug(`Key stats updated: success=${keyObj.success}, fail=${keyObj.fail}`);
      
      // 检查是否需要废弃key（402状态码）
      if (this.removeOn402 && statusCode === 402 && !keyObj.deprecated) {
        this.deprecateKey(key);
      }
    }
    
    // 记录端点的统计
    if (!this.endpointStats[endpoint]) {
      this.endpointStats[endpoint] = { success: 0, fail: 0 };
    }
    if (success) {
      this.endpointStats[endpoint].success++;
    } else {
      this.endpointStats[endpoint].fail++;
    }
    logDebug(`Endpoint stats updated: ${endpoint}, success=${this.endpointStats[endpoint].success}, fail=${this.endpointStats[endpoint].fail}`);
  }
  
  /**
   * 更新余额与用尽状态（按索引）
   * @param {number} index
   * @param {object} balanceInfo { totalAllowance, used, remaining, fetchedAt }
   */
  setBalanceByIndex(index, balanceInfo) {
    if (typeof index !== 'number' || index < 0 || index >= this.keys.length) return;
    const k = this.keys[index];
    if (!k) return;
    k.balance = balanceInfo || null;
    if (balanceInfo && typeof balanceInfo.remaining === 'number') {
      k.depleted = balanceInfo.remaining <= this.skipThreshold;
    }
  }

  /**
   * 设置跳过阈值（剩余≤n 视为用尽）并重算状态
   */
  setSkipThreshold(n) {
    const val = Number(n);
    this.skipThreshold = Number.isFinite(val) && val >= 0 ? val : 0;
    for (const k of this.keys) {
      if (k.balance && typeof k.balance.remaining === 'number') {
        k.depleted = k.balance.remaining <= this.skipThreshold;
      }
    }
  }

  getSkipThreshold() {
    return this.skipThreshold;
  }

  /**
   * 废弃一个key
   * @param {string} key - 要废弃的key
   */
  deprecateKey(key) {
    const keyObj = this.keys.find(k => k.key === key);
    if (keyObj && !keyObj.deprecated) {
      keyObj.deprecated = true;
      // 移动到废弃列表
      this.deprecatedKeys.push({
        key: keyObj.key,
        success: keyObj.success,
        fail: keyObj.fail,
        deprecatedAt: new Date().toISOString()
      });
      logInfo(`Key deprecated due to 402 response: ${this.maskKey(key)}`);
      
      // 检查是否还有活跃的key
      const activeCount = this.keys.filter(k => !k.deprecated).length;
      if (activeCount === 0) {
        logError('All keys have been deprecated!', new Error('No active keys available'));
      } else {
        logInfo(`Remaining active keys: ${activeCount}`);
      }
    }
  }
  
  /**
   * 掩码key，只显示前6位和后6位
   */
  maskKey(key) {
    if (key.length <= 12) {
      return key; // key太短，不掩码
    }
    return `${key.substring(0, 6)}******${key.substring(key.length - 6)}`;
  }
  
  /**
   * 获取统计信息
   */
  getStats() {
    // 分离活跃的key和废弃的key（保留原始索引以便在UI中定位）
    const indexedKeys = this.keys.map((k, idx) => ({ ...k, __index: idx }));
    const activeKeys = indexedKeys.filter(k => !k.deprecated);
    const deprecatedKeys = indexedKeys.filter(k => k.deprecated);
    
    return {
      algorithm: this.algorithm,
      removeOn402: this.removeOn402,
      keys: activeKeys.map(keyObj => ({
        key: this.maskKey(keyObj.key),
        success: keyObj.success,
        fail: keyObj.fail,
        total: keyObj.success + keyObj.fail,
        successRate: keyObj.success + keyObj.fail > 0 
          ? ((keyObj.success / (keyObj.success + keyObj.fail)) * 100).toFixed(2) + '%'
          : 'N/A',
        index: keyObj.__index,
        depleted: !!keyObj.depleted
      })),
      deprecatedKeys: deprecatedKeys.map(keyObj => ({
        key: this.maskKey(keyObj.key),
        success: keyObj.success,
        fail: keyObj.fail,
        total: keyObj.success + keyObj.fail,
        successRate: keyObj.success + keyObj.fail > 0 
          ? ((keyObj.success / (keyObj.success + keyObj.fail)) * 100).toFixed(2) + '%'
          : 'N/A',
        deprecatedAt: this.deprecatedKeys.find(dk => dk.key === keyObj.key)?.deprecatedAt || 'Unknown',
        index: keyObj.__index
      })),
      endpoints: Object.entries(this.endpointStats)
        .filter(([_, stats]) => stats.success > 0 || stats.fail > 0)
        .map(([endpoint, stats]) => ({
          endpoint,
          success: stats.success,
          fail: stats.fail,
          total: stats.success + stats.fail,
          successRate: stats.success + stats.fail > 0
            ? ((stats.success / (stats.success + stats.fail)) * 100).toFixed(2) + '%'
            : 'N/A'
        }))
    };
  }
}

// 全局实例
let keyManagerInstance = null;

/**
 * 初始化KeyManager
 */
export function initializeKeyManager(keys, algorithm, removeOn402 = true) {
  if (keys && keys.length > 0) {
    keyManagerInstance = new KeyManager(keys, algorithm, removeOn402);
    return keyManagerInstance;
  }
  return null;
}

/**
 * 获取KeyManager实例
 */
export function getKeyManager() {
  return keyManagerInstance;
}

/**
 * 重置KeyManager实例
 */
export function resetKeyManager() {
  keyManagerInstance = null;
}

export default KeyManager;
