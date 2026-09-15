/**
 * 精准Token统计工具
 * 使用tiktoken进行准确的token计数
 */

const tiktoken = require('tiktoken')

/**
 * 使用tiktoken进行精准token计数
 * @param {string} text - 要计数的文本
 * @param {string} model - 模型名称，默认为gpt-3.5-turbo
 * @returns {number} 精确的token数量
 */
function countTokens(text, model = 'gpt-3.5-turbo') {
  if (!text || typeof text !== 'string') return 0

  const encoding = tiktoken.encoding_for_model(model)
  const tokens = encoding.encode(text)
  encoding.free() // 释放内存
  return tokens.length
}



/**
 * 计算消息数组的token数量
 * @param {Array} messages - 消息数组
 * @param {string} model - 模型名称
 * @returns {number} 总token数量
 */
function countMessagesTokens(messages, model = 'gpt-3.5-turbo') {
  if (!Array.isArray(messages)) return 0

  let totalTokens = 0

  // 每条消息的基础开销（根据OpenAI文档）
  const messageOverhead = 4 // 每条消息约4个token的格式开销

  for (const message of messages) {
    totalTokens += messageOverhead

    // 角色token
    if (message.role) {
      totalTokens += countTokens(message.role, model)
    }

    // 内容token
    if (typeof message.content === 'string') {
      totalTokens += countTokens(message.content, model)
    } else if (Array.isArray(message.content)) {
      for (const item of message.content) {
        if (item.text) {
          totalTokens += countTokens(item.text, model)
        }
      }
    }

    // 函数调用等其他字段的token计算
    if (message.function_call) {
      totalTokens += countTokens(JSON.stringify(message.function_call), model)
    }
  }

  // 对话的额外开销
  totalTokens += 2 // 对话开始和结束的token

  return totalTokens
}

/**
 * 创建精准的usage对象
 * @param {Array|string} promptMessages - 提示消息或文本
 * @param {string} completionText - 完成文本
 * @param {object} realUsage - 真实的usage数据（如果有）
 * @param {string} model - 模型名称
 * @returns {object} usage对象
 */
function createUsageObject(promptMessages, completionText = '', realUsage = null, model = 'gpt-3.5-turbo') {
  // 如果有真实的usage数据，优先使用
  if (realUsage && realUsage.prompt_tokens && realUsage.completion_tokens) {
    return {
      prompt_tokens: realUsage.prompt_tokens,
      completion_tokens: realUsage.completion_tokens,
      total_tokens: realUsage.total_tokens || (realUsage.prompt_tokens + realUsage.completion_tokens)
    }
  }

  // 计算prompt tokens
  let promptTokens = 0
  if (Array.isArray(promptMessages)) {
    promptTokens = countMessagesTokens(promptMessages, model)
  } else if (typeof promptMessages === 'string') {
    promptTokens = countTokens(promptMessages, model)
  }

  // 计算completion tokens
  const completionTokens = countTokens(completionText, model)

  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens
  }
}

/**
 * 把上游帧里的一个计数字段转成有效数字：数字串也接受；负数、NaN、非数字、
 * 以及 0（上游"没数"时也发 0）都当作"没报"→ null。
 */
function toReportedCount(value) {
  if (typeof value === 'string' && value.trim() !== '') value = Number(value)
  return (typeof value === 'number' && Number.isFinite(value) && value > 0) ? value : null
}

function firstReportedCount(raw, keys) {
  for (const key of keys) {
    const count = toReportedCount(raw[key])
    if (count !== null) return count
  }
  return null
}

/**
 * 上游 usage 归一化。Qwen（DashScope 命名）发 input_tokens / output_tokens，
 * OpenAI 命名发 prompt_tokens / completion_tokens；统一成 OpenAI 命名。
 * 没报的字段为 null，让调用方只补估算那一个字段。
 * @param {*} raw - 上游帧里的 usage 对象
 * @returns {{prompt_tokens: number|null, completion_tokens: number|null}|null} 一个可用字段都没有时返回 null
 */
function normalizeUpstreamUsage(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const prompt_tokens = firstReportedCount(raw, ['input_tokens', 'prompt_tokens'])
  const completion_tokens = firstReportedCount(raw, ['output_tokens', 'completion_tokens'])
  if (prompt_tokens === null && completion_tokens === null) return null
  return { prompt_tokens, completion_tokens }
}

/**
 * 逐帧累积上游 usage。Qwen 每个 typing 帧都带累计值，最后的 finished 帧不带：
 * 报了的字段以最后一次为准，没报的保持已累积的值。
 * @param {{prompt_tokens: number|null, completion_tokens: number|null}|null} acc - 累积值（初始 null）
 * @param {*} rawFrameUsage - 当前帧的 usage
 */
function mergeUpstreamUsage(acc, rawFrameUsage) {
  const frame = normalizeUpstreamUsage(rawFrameUsage)
  if (!frame) return acc
  return {
    prompt_tokens: frame.prompt_tokens ?? acc?.prompt_tokens ?? null,
    completion_tokens: frame.completion_tokens ?? acc?.completion_tokens ?? null
  }
}

/**
 * 只对上游没报的字段补本地估算；两项都有时不调用估算（tiktoken 有成本）。
 * @param {{prompt_tokens: number|null, completion_tokens: number|null}|null} acc - 累积的上游 usage
 * @param {() => {prompt_tokens: number, completion_tokens: number}} estimate - 惰性本地估算
 * @returns {{prompt_tokens: number, completion_tokens: number, total_tokens: number}}
 */
function resolveUsage(acc, estimate) {
  const upstreamPrompt = acc?.prompt_tokens ?? null
  const upstreamCompletion = acc?.completion_tokens ?? null
  const estimated = (upstreamPrompt === null || upstreamCompletion === null) ? estimate() : null
  const prompt_tokens = upstreamPrompt ?? (estimated.prompt_tokens || 0)
  const completion_tokens = upstreamCompletion ?? (estimated.completion_tokens || 0)
  return { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens }
}

/**
 * 给日志用：这次响应的 usage 来源。两项都来自上游是 "upstream"；
 * 哪怕只有一项是本地估算的也算 "estimated"。
 * @param {{prompt_tokens: number|null, completion_tokens: number|null}|null} acc
 * @returns {'upstream'|'estimated'}
 */
function describeUsageSource(acc) {
  const complete = (acc?.prompt_tokens ?? null) !== null && (acc?.completion_tokens ?? null) !== null
  return complete ? 'upstream' : 'estimated'
}

module.exports = {
  countTokens,
  countMessagesTokens,
  createUsageObject,
  normalizeUpstreamUsage,
  mergeUpstreamUsage,
  resolveUsage,
  describeUsageSource
}
