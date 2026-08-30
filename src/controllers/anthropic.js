const { isJson, generateUUID } = require('../utils/tools.js');
const { createUsageObject } = require('../utils/precise-tokenizer.js');
const { sendChatRequest } = require('../utils/request.js');
const accountManager = require('../utils/account.js');
const { isChatType, isThinkingEnabled, parserModel, parserMessages, createUpstreamDeltaNormalizer } = require('../utils/chat-helpers.js');
const {
  buildToolSystemPrompt,
  foldToolMessages,
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator,
  looksLikeUnexecutedToolAction
} = require('../utils/tool-prompt.js');
const { createAgentTagStripper, stripAgentTags, buildAgentRetryHint } = require('../utils/agent-turn.js');
const { consumeSSEStream, createUpstreamResponseFilter } = require('../utils/sse.js');
const { logger } = require('../utils/logger');
const { assertNoUpstreamFailure } = require('../utils/upstream-error.js');
const {
  analyzeAnthropicCompatibility,
  buildAnthropicCompatibilityHeaders
} = require('./anthropic.compatibility.js');

const mapAnthropicStopReason = (upstreamReason, hasToolCalls, upstreamCompleted) => {
  if (hasToolCalls) return 'tool_use';
  if (upstreamReason === 'length' || upstreamReason === 'max_tokens') return 'max_tokens';
  if (upstreamReason === 'stop_sequence') return 'stop_sequence';
  if (upstreamReason === 'content_filter' || upstreamReason === 'refusal') return 'refusal';
  if (upstreamReason === 'stop' || upstreamReason === 'end_turn') return 'end_turn';
  if (!upstreamReason && upstreamCompleted) return 'end_turn';
  return null;
};

const writeAnthropicError = (res, message, errorType = 'api_error') => {
  writeAnthropicEvent(res, 'error', {
    type: 'error',
    error: { type: errorType, message }
  });
  res.end();
};

/**
 * 安全累计 chat stats（与 chat.js attributeChatUsage 共享语义）
 * 静默吞掉异常——stats 累计失败不应中断响应
 * 同 epic notes: tool-retry 全归属主账户（精度损失可接受）
 * @param {Object} account - 主请求账户对象
 * @param {number} promptTokens - 输入 tokens
 * @param {number} completionTokens - 输出 tokens
 */
const attributeChatUsage = (account, promptTokens, completionTokens) => {
  if (!account || !account.email) return;
  try {
    accountManager.accumulateStats(account.email, 'chat', {
      input: Number(promptTokens) || 0,
      output: Number(completionTokens) || 0
    });
  } catch (e) {
    // 静默
  }
};

/**
 * Anthropic stop_reason 枚举
 * @typedef {('end_turn'|'tool_use'|'max_tokens'|'stop_sequence')} AnthropicStopReason
 */

/**
 * 将 Anthropic system 字段规范为字符串
 * @param {string|Array<Object>} system - Anthropic system
 * @returns {string} 合并后的 system 文本
 */
const normalizeAnthropicSystem = (system) => {
  if (!system) return '';
  if (typeof system === 'string') return system;
  if (Array.isArray(system)) {
    return system
      .filter(b => b && b.type === 'text')
      .map(b => b.text || '')
      .join('\n');
  }
  return '';
};

/**
 * 将 Anthropic tools 列表转为 OpenAI 风格供 buildToolSystemPrompt 使用
 * @param {Array<Object>} tools - Anthropic 工具定义
 * @returns {Array<Object>} OpenAI 风格工具定义
 */
const normalizeAnthropicTools = (tools) => {
  if (!Array.isArray(tools)) return [];
  return tools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema || { type: 'object', properties: {} }
    }
  }));
};

/**
 * 将 Anthropic tool_choice 转为内部统一形式
 * @param {Object} toolChoice - Anthropic tool_choice
 * @returns {string|Object|undefined} OpenAI 风格 tool_choice
 */
const normalizeAnthropicToolChoice = (toolChoice) => {
  if (!toolChoice || typeof toolChoice !== 'object') return undefined;
  if (toolChoice.type === 'auto') return 'auto';
  if (toolChoice.type === 'any') return 'required';
  if (toolChoice.type === 'tool' && toolChoice.name) {
    return { type: 'function', function: { name: toolChoice.name } };
  }
  if (toolChoice.type === 'none') return 'none';
  return undefined;
};

/**
 * 把 Anthropic 风格的消息（含 content blocks 与 tool_use/tool_result）展开为
 * OpenAI 风格消息列表。tool_use 转为 assistant.tool_calls；tool_result 转为
 * role=tool 消息（保留 tool_call_id），后续由 foldToolMessages 折叠。
 * @param {Array<Object>} messages - Anthropic messages
 * @returns {Array<Object>} OpenAI 风格 messages
 */
const flattenAnthropicMessages = (messages) => {
  if (!Array.isArray(messages)) return [];
  const out = [];

  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue;
    const role = msg.role;

    if (typeof msg.content === 'string') {
      out.push({ role, content: msg.content });
      continue;
    }

    if (!Array.isArray(msg.content)) continue;

    if (role === 'assistant') {
      const textParts = [];
      const toolCalls = [];
      for (const block of msg.content) {
        if (block?.type === 'text' && typeof block.text === 'string') {
          textParts.push(block.text);
        } else if (block?.type === 'tool_use') {
          toolCalls.push({
            id: block.id || `toolu_${generateUUID().replace(/-/g, '').slice(0, 24)}`,
            type: 'function',
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input ?? {})
            }
          });
        }
      }
      const out_msg = { role: 'assistant', content: textParts.join('') };
      if (toolCalls.length > 0) out_msg.tool_calls = toolCalls;
      out.push(out_msg);
      continue;
    }

    // user 角色：tool_result 拆为独立 role=tool 消息，普通文本/图片合并保留
    const collectedTextParts = [];
    const flushCollectedText = () => {
      if (collectedTextParts.length === 0) return;
      out.push({ role: 'user', content: collectedTextParts.join('') });
      collectedTextParts.length = 0;
    };
    for (const block of msg.content) {
      if (block?.type === 'tool_result') {
        flushCollectedText();
        const resultContent = typeof block.content === 'string'
          ? block.content
          : Array.isArray(block.content)
            ? block.content.filter(b => b?.type === 'text').map(b => b.text || '').join('\n')
            : JSON.stringify(block.content ?? '');
        out.push({
          role: 'tool',
          tool_call_id: block.tool_use_id || '',
          content: resultContent
        });
      } else if (block?.type === 'text' && typeof block.text === 'string') {
        collectedTextParts.push(block.text);
      } else if (block?.type === 'image') {
        // 透传 image 块给现有 parserMessages 处理（OpenAI image_url 形态）
        const src = block.source || {};
        const url = src.type === 'base64' && src.data
          ? `data:${src.media_type || 'image/png'};base64,${src.data}`
          : (src.url || '');
        if (url) {
          if (collectedTextParts.length > 0) {
            out.push({
              role: 'user',
              content: [
                { type: 'text', text: collectedTextParts.join('') },
                { type: 'image_url', image_url: { url } }
              ]
            });
            collectedTextParts.length = 0;
          } else {
            out.push({ role: 'user', content: [{ type: 'image_url', image_url: { url } }] });
          }
        }
      }
    }
    flushCollectedText();
  }

  return out;
};

/**
 * 构造内部 Qwen 上游请求体
 * @param {Object} anthropicReq - Anthropic 风格请求体
 * @returns {Promise<{body: Object, hasTools: boolean, toolChoice: any, allowedToolNames: string[], enable_thinking: boolean, model: string}>} 转换结果
 */
const buildInternalRequest = async (anthropicReq) => {
  const { model, messages, system, tools, tool_choice, stream, thinking } = anthropicReq;

  const normalizedTools = normalizeAnthropicTools(tools);
  const internalToolChoice = normalizeAnthropicToolChoice(tool_choice);

  // 1. 展开 Anthropic 消息（tool_use/tool_result 折叠由 foldToolMessages 完成）
  let flat = flattenAnthropicMessages(messages);
  const systemText = normalizeAnthropicSystem(system);

  // 2. system 文本拼到首条用户消息内容前缀（不要作为独立 system 消息，
  //    否则会被 parserMessages 折叠为 "system:..." 文字前缀污染模型理解）
  const hasTools = normalizedTools.length > 0;
  const toolPrompt = hasTools ? buildToolSystemPrompt(normalizedTools, { tool_choice: internalToolChoice }) : '';

  if (hasTools) {
    flat = foldToolMessages(flat);
  }

  // 3. 走现有 parserMessages 复用图片上传与 thinking 配置
  const enable_thinking = !!(thinking && thinking.type === 'enabled');
  const thinkingCfg = await isThinkingEnabled(model, enable_thinking, thinking?.budget_tokens);
  const chatType = isChatType(model);
  const parsedMessages = await parserMessages(flat, thinkingCfg, chatType);
  const parsedModel = await parserModel(model);

  // 4. 合并 system 文本与工具提示词到最终用户消息开头
  const prefixParts = [systemText, toolPrompt].filter(Boolean);
  if (prefixParts.length > 0 && Array.isArray(parsedMessages) && parsedMessages.length > 0) {
    const prefix = prefixParts.join('\n\n');
    const last = parsedMessages[parsedMessages.length - 1];
    if (typeof last.content === 'string') {
      last.content = `${prefix}\n\n${last.content}`;
    } else if (Array.isArray(last.content)) {
      const textIdx = last.content.findIndex(c => c && c.type === 'text');
      if (textIdx >= 0) {
        last.content[textIdx].text = `${prefix}\n\n${last.content[textIdx].text || ''}`;
      } else {
        last.content.unshift({
          type: 'text',
          text: prefix,
          chat_type: 't2t',
          feature_config: { output_schema: 'phase', thinking_enabled: false }
        });
      }
    }
  }

  // Align with React UI envelope format (chat-middleware.js lines 63-100)
  // to avoid WAF/captcha rejection (FAIL_SYS_USER_VALIDATE).
  const now = Math.floor(Date.now() / 1000);
  const fid = generateUUID();
  const lastParsed = Array.isArray(parsedMessages) && parsedMessages.length > 0
    ? parsedMessages[parsedMessages.length - 1]
    : { role: 'user', content: '' };

  const envelopeMessage = {
    id: null,
    fid: fid,
    parentId: null,
    parent_id: null,
    childrenIds: [generateUUID()],
    role: lastParsed.role || 'user',
    content: lastParsed.content || '',
    user_action: 'chat',
    files: [],
    timestamp: now,
    models: [parsedModel],
    model: '',
    chat_type: chatType,
    feature_config: {
      output_schema: 'phase',
      thinking_enabled: thinkingCfg.thinking_enabled,
      research_mode: 'normal',
      auto_thinking: true,
      thinking_mode: 'Auto',
      thinking_format: 'summary',
      auto_search: true
    },
    extra: { meta: { subChatType: chatType } },
    sub_chat_type: chatType
  };

  const body = {
    stream: !!stream,
    version: '2.1',
    incremental_output: true,
    chat_id: null,
    chatId: null,
    chat_mode: 'normal',
    model: parsedModel,
    parent_id: null,
    parentId: null,
    messages: [envelopeMessage],
    timestamp: now,
    chat_type: chatType,
    sub_chat_type: chatType,
    session_id: generateUUID(),
    id: generateUUID()
  };

  // Pass max_tokens to upstream if provided (guard against NaN/Infinity)
  if (anthropicReq.max_tokens != null) {
    const mt = Number(anthropicReq.max_tokens);
    if (Number.isFinite(mt) && mt > 0) {
      body.max_tokens = mt;
    }
  }

  return {
    body,
    hasTools,
    toolChoice: internalToolChoice,
    allowedToolNames: normalizedTools.map(tool => tool.function.name).filter(Boolean),
    enable_thinking: thinkingCfg.thinking_enabled,
    model: parsedModel
  };
};

/**
 * 在请求体中追加用于 required 重试的强制提示
 * @param {Object} body - 内部请求体
 * @param {string} hint - 重试提示词
 * @returns {Object} 新请求体
 */
const appendRetryHint = (body, hint) => {
  const messages = Array.isArray(body.messages)
    ? body.messages.map(message => ({ ...message }))
    : [];
  if (messages.length === 0) {
    messages.push({ role: 'user', content: hint });
  } else {
    const last = messages[messages.length - 1];
    if (typeof last.content === 'string') {
      last.content = `${last.content}\n\n# Tool-call retry\n${hint}`;
    } else if (Array.isArray(last.content)) {
      const textPart = last.content.find(part => part?.type === 'text');
      if (textPart) {
        textPart.text = `${textPart.text || ''}\n\n# Tool-call retry\n${hint}`;
      } else {
        last.content = [{ type: 'text', text: hint }, ...last.content];
      }
    }
  }
  return { ...body, messages };
};

/**
 * 判断 tool_choice 是否需要强制调用
 * @param {string|Object} toolChoice - 内部 tool_choice
 * @returns {boolean} 是否要求至少一次工具调用
 */
const requiresToolCall = (toolChoice) => {
  if (toolChoice === 'required') return true;
  if (toolChoice && typeof toolChoice === 'object' && toolChoice.function?.name) return true;
  return false;
};

/**
 * 构建 required 重试提示
 * @param {string|Object} toolChoice - 内部 tool_choice
 * @returns {string} 提示文本
 */
const buildRetryHint = (toolChoice) => {
  if (toolChoice && typeof toolChoice === 'object' && toolChoice.function?.name) {
    return `You did not call any tool. You MUST now call \`${toolChoice.function.name}\` using the <tool_call>...</tool_call> format.`;
  }
  return 'You did not call any tool. You MUST now call exactly one tool using the <tool_call>...</tool_call> format.';
};

const buildEmptyOutputRetryHint = () => [
  'Your previous reply produced no visible final answer or executable tool call.',
  'Continue the Agent task now. If any action remains, emit the required `<tool_call>` block immediately with no preamble.',
  'Only give a normal final answer when the task is actually complete; do not repeat hidden reasoning.'
].join(' ');

const buildMissingToolRetryHint = () => [
  'Your previous reply described an action but did not execute any tool call.',
  'Perform that action now by emitting the real `<tool_call>` block immediately with no preamble.',
  'Do not describe the action again or claim completion without a tool result.'
].join(' ');

/**
 * 把解析器的错误列表压成一行可读的诊断串。
 * @param {Array<Object>} errors - parser/native accumulator 的 getErrors()
 * @returns {string} 形如 `unknown_tool: Bash, Read; invalid_json ×2`
 */
const describeToolErrors = (errors) => {
  const unknown = [...new Set(
    errors.filter(e => e?.type === 'unknown_tool').map(e => e.name).filter(Boolean)
  )];
  const parts = [];
  if (unknown.length) parts.push(`unknown_tool: ${unknown.join(', ')}`);
  for (const type of ['invalid_json', 'truncated_tool_call']) {
    const count = errors.filter(e => e?.type === type).length;
    if (count) parts.push(`${type} ×${count}`);
  }
  return parts.join('; ') || 'unspecified';
};

/**
 * 工具错误的重试提示。基础文本复用 agent-turn.js 的通用提示；当错误是编造的工具名时，
 * 补上真实的名字 —— 那是让这类错误可恢复的唯一信息。
 * @param {Array<Object>} errors - 本轮的工具错误
 * @param {Array<string>} allowedToolNames - 本次请求真正提供的工具名
 * @returns {string} 提示文本
 */
const buildToolErrorRetryHint = (errors, allowedToolNames) => {
  const base = buildAgentRetryHint('invalid_tool_call');
  const unknown = [...new Set(
    errors.filter(e => e?.type === 'unknown_tool').map(e => e.name).filter(Boolean)
  )];
  if (!unknown.length || !allowedToolNames?.length) return base;
  return [
    base,
    `The tool name(s) ${unknown.join(', ')} do not exist.`,
    `Use ONLY these exact tool names: ${allowedToolNames.join(', ')}.`
  ].join('\n');
};

/**
 * 异步迭代上游 axios 流，按 SSE 段切分回调内部 delta JSON
 * @param {object} upstream - axios stream 响应
 * @param {(json: Object) => Promise<void>|void} onDelta - 单个 delta 回调
 * @returns {Promise<void>} 完成 Promise
 */
const consumeUpstream = async (upstream, onDelta) => consumeSSEStream(upstream, async (frame) => {
  const payload = frame.data;
  if (!payload || payload.trim() === '[DONE]') return;
  if (!isJson(payload)) return;
  const parsed = JSON.parse(payload);
  assertNoUpstreamFailure(parsed);
  await onDelta(parsed);
});

/**
 * 把工具调用的 arguments JSON 字符串切成 input_json_delta 切片
 * @param {string} argsJson - 完整 JSON 字符串
 * @param {number} chunkSize - 单片大小
 * @returns {Array<string>} 切片列表
 */
const sliceArgsJson = (argsJson, chunkSize = 32) => {
  const out = [];
  for (let i = 0; i < argsJson.length; i += chunkSize) {
    out.push(argsJson.slice(i, i + chunkSize));
  }
  return out;
};

/**
 * 写入一个 Anthropic SSE 事件
 * @param {object} res - Express 响应
 * @param {string} event - 事件名
 * @param {Object} data - 事件 payload
 */
const writeAnthropicEvent = (res, event, data) => {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
};

// SSE 保活间隔。上游长时间静默的两个来源：首轮 thinking，以及门禁拒绝后的补偿重试
// —— 后者要整段重新生成，客户端在此期间看不到任何内容。
// 延迟读取：本文件没有在模块作用域引入 config，顶层读取会在加载时抛错。
const pingIntervalMs = () => require('../config/index.js').anthropicPingIntervalMs;

/**
 * 在 work 执行期间按间隔发送 Anthropic `ping` 事件，避免客户端把流判为卡死。
 *
 * 必须用协议内的 `ping` 事件，不能用 SSE 注释（`: keepalive`）：注释的字节能重置
 * 反向代理的空闲计时器，但 SDK 会在读取行时直接丢弃以 `:` 开头的行，客户端因此
 * 什么都收不到。ccproxy 网桥当初正是靠改发真正的 ping 事件才消除同样的假死。
 *
 * 只能在 message_start 之后调用——此时响应头已提交，ping 是合法的流内事件。
 * @param {object} res - Express 响应
 * @param {Function} work - 被包裹的异步任务
 * @param {number} [intervalMs] - 发送间隔，缺省取 config.anthropicPingIntervalMs
 * @returns {Promise<*>} work 的返回值
 */
const runWithAnthropicPing = async (res, work, intervalMs) => {
  const everyMs = Math.max(1, Number(intervalMs) || pingIntervalMs());
  const timer = setInterval(() => {
    if (res.writableEnded || res.destroyed) return;
    try {
      writeAnthropicEvent(res, 'ping', { type: 'ping' });
      if (typeof res.flush === 'function') res.flush();
    } catch (_) {
      // 客户端断开由后续流消费/写入路径统一收敛。
    }
  }, everyMs);
  timer.unref?.();
  try {
    return await work();
  } finally {
    clearInterval(timer);
  }
};

/**
 * 处理流式 Anthropic 响应
 * @param {object} res - Express 响应
 * @param {Object} ctx - 处理上下文
 * @param {object} upstream - 上游 axios 响应
 * @param {string} ctx.message_id - 消息 ID
 * @param {string} ctx.model - 模型名
 * @param {boolean} ctx.hasTools - 是否启用工具
 * @param {string|Object} ctx.toolChoice - 内部 tool_choice
 * @param {Object} ctx.requestBody - 内部请求体（用于重试）
 * @returns {Promise<void>} 完成 Promise
 */
const handleAnthropicStream = async (res, ctx, upstream) => {
  const {
    message_id, model, hasTools, toolChoice, requestBody, allowedToolNames = [],
    sendRequest = sendChatRequest
  } = ctx;

  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  const createdAt = new Date().toISOString();

  // message_start
  writeAnthropicEvent(res, 'message_start', {
    type: 'message_start',
    message: {
      id: message_id,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      created_at: createdAt,
      metadata: {},
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null
      }
    }
  });

  let blockIndex = -1;
  let textBlockOpen = false;
  let thinkingBlockOpen = false;
  let thinkingSignature = null;
  let promptTokens = 0;
  let completionTokens = 0;
  let upstreamFinishReason = null;
  let upstreamCompleted = false;
  let upstreamEventCount = 0;
  let visibleText = '';

  // 每个 attempt 都必须拿到全新的解析器。旧代码只建一次，于是补偿重试会继承上一轮的
  // 错误列表（hasParseError 永远为真，即使重试本身成功），而一个被截断的 <tool_call>
  // 还会让 inToolCall 保持打开，把下一轮的正文灌进上一轮的缓冲区。
  // OpenAI 路径正是为此每轮新建（openai-agent-runtime.js 顶部注释）。
  let parser = null;
  let nativeToolAccumulator = null;
  let agentTagStripper = null;
  let normalizeDelta = null;
  let acceptUpstreamFrame = null;

  const startAttempt = () => {
    parser = hasTools ? createToolCallStreamParser({ allowedToolNames }) : null;
    nativeToolAccumulator = hasTools
      ? createNativeToolCallAccumulator({ allowedToolNames })
      : null;
    // buildToolSystemPrompt 让模型把最终答复包进 <agent_final>...</agent_final>，
    // 但本控制器没有 Agent 回合门禁去解包，标签会原样发给客户端。剥掉它们。
    agentTagStripper = createAgentTagStripper();
    normalizeDelta = createUpstreamDeltaNormalizer();
    acceptUpstreamFrame = createUpstreamResponseFilter();
    upstreamFinishReason = null;
  };

  /**
   * 关闭当前打开的文本块
   */
  const closeTextBlockIfOpen = () => {
    if (textBlockOpen) {
      writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: blockIndex });
      textBlockOpen = false;
    }
  };

  /**
   * 关闭当前打开的思维块
   */
  const closeThinkingBlockIfOpen = () => {
    if (thinkingBlockOpen) {
      writeAnthropicEvent(res, 'content_block_delta', {
        type: 'content_block_delta',
        index: blockIndex,
        delta: { type: 'signature_delta', signature: thinkingSignature }
      });
      writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: blockIndex });
      thinkingBlockOpen = false;
      thinkingSignature = null;
    }
  };

  /**
   * 输出一段思维增量；按需打开新思维块
   * @param {string} thinking - 思维增量
   */
  const emitThinkingDelta = (thinking) => {
    if (!thinking) return;
    if (!thinkingBlockOpen) {
      closeTextBlockIfOpen();
      blockIndex += 1;
      thinkingSignature = `qwen2api_${generateUUID().replace(/-/g, '')}`;
      writeAnthropicEvent(res, 'content_block_start', {
        type: 'content_block_start',
        index: blockIndex,
        content_block: { type: 'thinking', thinking: '' }
      });
      thinkingBlockOpen = true;
    }
    writeAnthropicEvent(res, 'content_block_delta', {
      type: 'content_block_delta',
      index: blockIndex,
      delta: { type: 'thinking_delta', thinking }
    });
  };

  /**
   * 输出一段文本增量；按需打开新文本块
   * @param {string} text - 文本增量
   */
  const emitTextDelta = (text) => {
    if (!text) return;
    visibleText += text;
    if (!textBlockOpen) {
      closeThinkingBlockIfOpen();
      blockIndex += 1;
      writeAnthropicEvent(res, 'content_block_start', {
        type: 'content_block_start',
        index: blockIndex,
        content_block: { type: 'text', text: '' }
      });
      textBlockOpen = true;
    }
    writeAnthropicEvent(res, 'content_block_delta', {
      type: 'content_block_delta',
      index: blockIndex,
      delta: { type: 'text_delta', text }
    });
  };

  /**
   * 输出一个完整的 tool_use 块（按 input_json_delta 切片）
   * @param {Object} call - 工具调用
   */
  const emitToolUse = (call) => {
    closeThinkingBlockIfOpen();
    closeTextBlockIfOpen();
    blockIndex += 1;
    writeAnthropicEvent(res, 'content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'tool_use', id: call.id, name: call.function.name, input: {} }
    });
    const args = call.function.arguments || '{}';
    for (const piece of sliceArgsJson(args)) {
      writeAnthropicEvent(res, 'content_block_delta', {
        type: 'content_block_delta',
        index: blockIndex,
        delta: { type: 'input_json_delta', partial_json: piece }
      });
    }
    writeAnthropicEvent(res, 'content_block_stop', { type: 'content_block_stop', index: blockIndex });
  };

  let completionContent = '';
  let webSearchInfo = null;
  let thinkingStarted = false;

  /**
   * 处理一个上游 delta JSON
   * @param {Object} json - 上游 SSE delta
   */
  const onUpstreamDelta = async (json) => {
    // 丢弃其余候选回答的帧：上游多路并发会让内容重复
    if (!acceptUpstreamFrame(json)) return;
    if (json.usage) {
      promptTokens = json.usage.prompt_tokens || promptTokens;
      completionTokens = json.usage.completion_tokens || completionTokens;
    }
    if (!json.choices || json.choices.length === 0) return;
    const choice = json.choices[0];
    const reportedFinishReason = choice.finish_reason ?? choice.delta?.finish_reason;
    if (reportedFinishReason !== undefined && reportedFinishReason !== null) {
      upstreamFinishReason = reportedFinishReason;
    }
    const delta = choice.delta || {};
    if (nativeToolAccumulator && Array.isArray(delta.tool_calls)) {
      nativeToolAccumulator.push(delta.tool_calls);
    } else if (nativeToolAccumulator && delta.function_call) {
      nativeToolAccumulator.push([{ index: 0, type: 'function', function: delta.function_call }]);
    }
    if (delta && delta.name === 'web_search') {
      webSearchInfo = delta.extra?.web_search_info;
    }
    const normalized = normalizeDelta(delta);
    if (!normalized) return;
    delta.phase = normalized.phase;
    let content = normalized.content;
    completionContent += content;

    if (delta.phase === 'think') {
      if (!thinkingStarted) {
        thinkingStarted = true;
        if (webSearchInfo) {
          const config = require('../config/index.js');
          try {
            const searchTable = await accountManager.generateMarkdownTable(webSearchInfo, config.searchInfoMode);
            emitThinkingDelta(searchTable + '\n\n');
          } catch (_) {}
        }
      }
      emitThinkingDelta(content);
    } else if (delta.phase === 'answer') {
      if (parser) {
        const parsed = parser.push(content);
        if (parsed.textDelta) emitTextDelta(agentTagStripper.push(parsed.textDelta));
        for (const call of parsed.completedCalls) emitToolUse(call);
      } else {
        emitTextDelta(agentTagStripper.push(content));
      }
    }
  };

  const terminalFinish = () =>
    ['length', 'max_tokens', 'content_filter', 'refusal'].includes(upstreamFinishReason);

  const currentToolErrors = () => [
    ...(parser?.getErrors() || []),
    ...(nativeToolAccumulator?.getErrors() || [])
  ];

  /**
   * 判断本轮是否需要重试；返回 null 表示接受本轮。
   * 只在 flush 之后调用：flush 会结算挂起的工具调用，此后 hasPendingCall() 恒为假。
   */
  const decideRetryReason = (emittedCalls) => {
    if (emittedCalls) return null;
    if (parser && requiresToolCall(toolChoice)) return 'required';
    // 以前任何一个工具错误都会让全部补偿失效并直接 502。可是被编造的工具名恰恰是
    // 最容易纠正的错误：把允许的名字摆在模型面前即可。
    if (currentToolErrors().length > 0) return 'tool_error';
    if (hasTools && looksLikeUnexecutedToolAction(visibleText) && !terminalFinish()) {
      return 'missing_tool';
    }
    if (!visibleText.trim() && !terminalFinish()) return 'empty';
    return null;
  };

  const retryHintFor = (reason) => {
    if (reason === 'required') return buildRetryHint(toolChoice);
    if (reason === 'missing_tool') return buildMissingToolRetryHint();
    if (reason === 'empty') return buildEmptyOutputRetryHint();
    return buildToolErrorRetryHint(currentToolErrors(), allowedToolNames);
  };

  const config = require('../config/index.js');
  const maxAttempts = Math.max(1, Number(config.agentTurnMaxAttempts) || 1);

  let currentUpstream = upstream;
  let attemptsMade = 0;
  let retriedAfterVisibleText = false;
  let nativeToolCalls = [];
  let hasEmittedToolCalls = false;

  for (;;) {
    attemptsMade += 1;
    startAttempt();

    try {
      const result = await runWithAnthropicPing(
        res,
        () => consumeUpstream(currentUpstream, onUpstreamDelta)
      );
      upstreamCompleted = result.completed;
      upstreamEventCount = result.eventCount;
    } catch (e) {
      logger.error('Anthropic 流式心跳包装失败', 'ANTHROPIC', '', e);
      throw e;
    }

    // 本轮收尾。解析器的尾巴属于这一轮，必须在判定之前放出来。
    if (parser) {
      const tail = parser.flush();
      if (tail.textDelta) emitTextDelta(agentTagStripper.push(tail.textDelta));
      for (const call of tail.completedCalls) emitToolUse(call);
    }
    // 缓冲区里可能压着一个最终没能凑成标签的前缀，它是正文，必须放出来。
    emitTextDelta(agentTagStripper.flush());

    nativeToolCalls = nativeToolAccumulator?.hasAny()
      ? nativeToolAccumulator.finalize()
      : [];
    for (const call of nativeToolCalls) emitToolUse(call);
    hasEmittedToolCalls = !!(nativeToolCalls.length > 0 || parser?.hasEmittedAnyCall());

    const retryReason = decideRetryReason(hasEmittedToolCalls);
    if (!retryReason || attemptsMade >= maxAttempts) break;

    // 本控制器是边收边发的：正文一产生就写进客户端的流（OpenAI 路径把裸正文扣在门禁
    // 内，所以它可以随便重试）。因此一旦写过正文，再重试就会把两段输出拼在一起。
    //
    // 已经写过正文时只允许一次补偿 —— 这正是改动之前的行为，required 和 missing_tool
    // 都依赖它。还没写过正文时才放开到 maxAttempts，而上报的故障恰好是这种形状：
    // 一轮纯 <tool_call> 且工具名无效不产生任何可见正文，所以 6 次尝试都够得着。
    if (visibleText.trim()) {
      if (retriedAfterVisibleText) break;
      retriedAfterVisibleText = true;
    }

    logger.warning?.(
      `Anthropic Agent attempt ${attemptsMade}/${maxAttempts} 被拒绝 (${retryReason})`,
      'ANTHROPIC'
    );

    let retryResp = null;
    try {
      await runWithAnthropicPing(res, async () => {
        retryResp = await sendRequest(appendRetryHint(requestBody, retryHintFor(retryReason)));
      });
    } catch (e) {
      logger.error('Anthropic 流式重试失败', 'ANTHROPIC', '', e);
      if (e.publicMessage) throw e;
      break;
    }
    if (!retryResp?.status || !retryResp.response) break;
    currentUpstream = retryResp.response;
  }

  const finalToolErrors = currentToolErrors();
  const hasToolProtocolError = !!(
    !hasEmittedToolCalls &&
    (requiresToolCall(toolChoice) || finalToolErrors.length > 0)
  );

  if (hasToolProtocolError) {
    // 这个细节以前存在于 getErrors() 里却被丢掉，于是三种截然不同的原因
    // （非法 JSON / 未知工具名 / 被截断）挤进同一句不透明的报错，而 unknown_tool
    // 连一行日志都不留。诊断只能靠读源码。
    const detail = finalToolErrors.length
      ? describeToolErrors(finalToolErrors)
      : 'tool_choice=required 未触发任何工具调用';
    logger.warning?.(
      `Anthropic Agent 工具协议失败，${attemptsMade}/${maxAttempts} 次尝试后放弃 (${detail})`,
      'ANTHROPIC'
    );
    closeThinkingBlockIfOpen();
    closeTextBlockIfOpen();
    writeAnthropicError(
      res,
      attemptsMade > 1
        ? `上游连续 ${attemptsMade} 次返回了残缺、非法或不存在的工具调用 (${detail})`
        : `上游返回了残缺、非法或不存在的工具调用 (${detail})`,
      'invalid_tool_call_error'
    );
    return;
  }

  if (!visibleText.trim() && !hasEmittedToolCalls &&
      !['length', 'max_tokens', 'content_filter', 'refusal'].includes(upstreamFinishReason)) {
    closeThinkingBlockIfOpen();
    closeTextBlockIfOpen();
    writeAnthropicError(res, '上游重试后仍未返回正文或工具调用', 'api_error');
    return;
  }

  closeThinkingBlockIfOpen();
  closeTextBlockIfOpen();

  const stopReason = mapAnthropicStopReason(
    upstreamFinishReason,
    hasEmittedToolCalls,
    upstreamCompleted
  );
  if (!stopReason) {
    const detail = upstreamEventCount === 0 ? '上游未返回任何 SSE 事件' : '上游流在结束标记前断开';
    writeAnthropicError(res, detail, 'api_error');
    return;
  }

  if (promptTokens === 0 && completionTokens === 0) {
    const usage = createUsageObject(requestBody?.messages || '', completionContent, null);
    promptTokens = usage.prompt_tokens || 0;
    completionTokens = usage.completion_tokens || 0;
  }

  // Daily stats 累计——一次性归属主账户（见模块顶部 attributeChatUsage 注释）
  attributeChatUsage(ctx.currentAccount, promptTokens, completionTokens);

  writeAnthropicEvent(res, 'message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null
    }
  });
  writeAnthropicEvent(res, 'message_stop', { type: 'message_stop' });
  res.end();
};

/**
 * 处理非流式 Anthropic 响应
 * @param {object} res - Express 响应
 * @param {Object} ctx - 处理上下文
 * @param {object} upstream - 上游 axios 响应
 * @returns {Promise<void>} 完成 Promise
 */
const handleAnthropicNonStream = async (res, ctx, upstream) => {
  const {
    message_id, model, hasTools, toolChoice, requestBody, allowedToolNames = [],
    sendRequest = sendChatRequest
  } = ctx;

  let thinkingContent = '';
  let answerContent = '';
  let promptTokens = 0;
  let completionTokens = 0;
  let webSearchInfo = null;
  let upstreamFinishReason = null;
  let upstreamCompleted = false;
  let upstreamEventCount = 0;
  let nativeToolAccumulator = hasTools
    ? createNativeToolCallAccumulator({ allowedToolNames })
    : null;
  const normalizeDelta = createUpstreamDeltaNormalizer();
  const acceptUpstreamFrame = createUpstreamResponseFilter();

  /**
   * 处理一个上游 delta JSON
   * @param {Object} json - 上游 SSE delta
   */
  const onUpstreamDelta = async (json) => {
    // 丢弃其余候选回答的帧：上游多路并发会让内容重复
    if (!acceptUpstreamFrame(json)) return;
    if (json.usage) {
      promptTokens = json.usage.prompt_tokens || promptTokens;
      completionTokens = json.usage.completion_tokens || completionTokens;
    }
    if (!json.choices || json.choices.length === 0) return;
    const choice = json.choices[0];
    const reportedFinishReason = choice.finish_reason ?? choice.delta?.finish_reason;
    if (reportedFinishReason !== undefined && reportedFinishReason !== null) {
      upstreamFinishReason = reportedFinishReason;
    }
    const delta = choice.delta || {};
    if (nativeToolAccumulator && Array.isArray(delta.tool_calls)) {
      nativeToolAccumulator.push(delta.tool_calls);
    } else if (nativeToolAccumulator && delta.function_call) {
      nativeToolAccumulator.push([{ index: 0, type: 'function', function: delta.function_call }]);
    }
    if (delta && delta.name === 'web_search') {
      webSearchInfo = delta.extra?.web_search_info;
    }
    const normalized = normalizeDelta(delta);
    if (!normalized) return;
    delta.phase = normalized.phase;
    const content = normalized.content;
    if (delta.phase === 'think') {
      thinkingContent += content;
    } else if (delta.phase === 'answer') {
      answerContent += content;
    }
  };

  const initialStreamResult = await consumeUpstream(upstream, onUpstreamDelta);
  upstreamCompleted = initialStreamResult.completed;
  upstreamEventCount = initialStreamResult.eventCount;

  if (!upstreamCompleted && !upstreamFinishReason) {
    const detail = upstreamEventCount === 0 ? '上游未返回任何 SSE 事件' : '上游流在结束标记前断开';
    return res.status(502).json({
      type: 'error',
      error: { type: 'api_error', message: detail }
    });
  }

  if (webSearchInfo) {
    const config = require('../config/index.js');
    try {
      const searchTable = await accountManager.generateMarkdownTable(webSearchInfo, config.searchInfoMode);
      if (thinkingContent) {
        thinkingContent = searchTable + '\n\n' + thinkingContent;
      } else {
        answerContent = searchTable + '\n\n' + answerContent;
      }
    } catch (_) {}
  }

  let parsedTools = hasTools
    ? parseToolCallsFromText(answerContent, { allowedToolNames })
    : { cleanedText: answerContent, toolCalls: [], errors: [] };
  let cleanedText = stripAgentTags(parsedTools.cleanedText);
  let nativeToolCalls = nativeToolAccumulator?.hasAny()
    ? nativeToolAccumulator.finalize()
    : [];
  let toolCalls = [...nativeToolCalls, ...parsedTools.toolCalls]
    .map((call, index) => ({ ...call, index }));
  let toolErrors = [
    ...parsedTools.errors,
    ...(nativeToolAccumulator?.getErrors() || [])
  ];

  // 非流式没有"已经写到线上"的问题：什么都还没发出去，所以每一轮都可以重试。
  const terminalFinish = () =>
    ['length', 'max_tokens', 'content_filter', 'refusal'].includes(upstreamFinishReason);

  const decideRetryReason = () => {
    if (toolCalls.length > 0) return null;
    if (hasTools && requiresToolCall(toolChoice)) return 'required';
    // 以前任何一个工具错误都会让全部补偿失效并直接 502。被编造的工具名恰恰是最容易
    // 纠正的错误：把允许的名字摆在模型面前即可。
    if (toolErrors.length > 0) return 'tool_error';
    if (hasTools && looksLikeUnexecutedToolAction(cleanedText) && !terminalFinish()) {
      return 'missing_tool';
    }
    if (!cleanedText.trim() && !terminalFinish()) return 'empty';
    return null;
  };

  const config = require('../config/index.js');
  const maxAttempts = Math.max(1, Number(config.agentTurnMaxAttempts) || 1);
  let attemptsMade = 1;
  let streamBrokeOnRetry = false;

  while (attemptsMade < maxAttempts) {
    const retryReason = decideRetryReason();
    if (!retryReason) break;

    logger.warning?.(
      `Anthropic 非流式 Agent attempt ${attemptsMade}/${maxAttempts} 被拒绝 (${retryReason})`,
      'ANTHROPIC'
    );

    const hint = retryReason === 'required'
      ? buildRetryHint(toolChoice)
      : (retryReason === 'missing_tool'
        ? buildMissingToolRetryHint()
        : (retryReason === 'empty'
          ? buildEmptyOutputRetryHint()
          : buildToolErrorRetryHint(toolErrors, allowedToolNames)));

    let retryResp = null;
    try {
      retryResp = await sendRequest(appendRetryHint(requestBody, hint));
    } catch (e) {
      logger.error('Anthropic 非流式重试失败', 'ANTHROPIC', '', e);
      if (e.publicMessage) throw e;
      break;
    }
    if (!retryResp?.status || !retryResp.response) break;

    attemptsMade += 1;
    const before = answerContent;
    // 每轮全新的累加器，否则上一轮的错误会一直跟着走。
    nativeToolAccumulator = createNativeToolCallAccumulator({ allowedToolNames });
    upstreamFinishReason = null;
    const retryResult = await consumeUpstream(retryResp.response, onUpstreamDelta);
    upstreamCompleted = retryResult.completed;
    upstreamEventCount = retryResult.eventCount;
    if (!upstreamCompleted && !upstreamFinishReason) {
      streamBrokeOnRetry = true;
      break;
    }
    const retried = answerContent.slice(before.length);
    const parsedRetry = parseToolCallsFromText(retried, { allowedToolNames });
    nativeToolCalls = nativeToolAccumulator.hasAny()
      ? nativeToolAccumulator.finalize()
      : [];
    toolCalls = [...nativeToolCalls, ...parsedRetry.toolCalls]
      .map((call, index) => ({ ...call, index }));
    cleanedText = stripAgentTags(parsedRetry.cleanedText);
    toolErrors = [...parsedRetry.errors, ...nativeToolAccumulator.getErrors()];
  }

  if (streamBrokeOnRetry) {
    return res.status(502).json({
      type: 'error',
      error: { type: 'api_error', message: '工具调用重试流在结束标记前断开' }
    });
  }

  if (hasTools && toolCalls.length === 0 && (toolErrors.length > 0 || requiresToolCall(toolChoice))) {
    // 这个细节以前存在于 errors 里却被丢掉，于是三种截然不同的原因挤进同一句
    // 不透明的报错，而 unknown_tool 连一行日志都不留。
    const detail = toolErrors.length
      ? describeToolErrors(toolErrors)
      : 'tool_choice=required 未触发任何工具调用';
    logger.warning?.(
      `Anthropic 非流式工具协议失败，${attemptsMade}/${maxAttempts} 次尝试后放弃 (${detail})`,
      'ANTHROPIC'
    );
    return res.status(502).json({
      type: 'error',
      error: {
        type: 'invalid_tool_call_error',
        message: attemptsMade > 1
          ? `上游连续 ${attemptsMade} 次返回了残缺、非法或不存在的工具调用 (${detail})`
          : `上游返回了残缺、非法或不存在的工具调用 (${detail})`
      }
    });
  }

  if (toolCalls.length === 0 && !cleanedText.trim() &&
      !['length', 'max_tokens', 'content_filter', 'refusal'].includes(upstreamFinishReason)) {
    return res.status(502).json({
      type: 'error',
      error: { type: 'api_error', message: '上游重试后仍未返回正文或工具调用' }
    });
  }

  const stopReason = mapAnthropicStopReason(
    upstreamFinishReason,
    toolCalls.length > 0,
    upstreamCompleted
  );
  if (!stopReason) {
    return res.status(502).json({
      type: 'error',
      error: { type: 'api_error', message: '上游流在结束标记前断开' }
    });
  }

  if (promptTokens === 0 && completionTokens === 0) {
    const usage = createUsageObject(requestBody?.messages || '', thinkingContent + answerContent, null);
    promptTokens = usage.prompt_tokens || 0;
    completionTokens = usage.completion_tokens || 0;
  }

  const contentBlocks = [];
  if (thinkingContent && thinkingContent.trim()) {
    contentBlocks.push({
      type: 'thinking',
      thinking: thinkingContent,
      signature: `qwen2api_${generateUUID().replace(/-/g, '')}`
    });
  }
  if (cleanedText && cleanedText.trim()) {
    contentBlocks.push({ type: 'text', text: cleanedText });
  }
  for (const call of toolCalls) {
    let input = {};
    try { input = JSON.parse(call.function.arguments || '{}'); } catch (_) { input = {}; }
    contentBlocks.push({
      type: 'tool_use',
      id: call.id,
      name: call.function.name,
      input
    });
  }

  // Daily stats 累计——一次性归属主账户（同 stream 分支注释）
  attributeChatUsage(ctx.currentAccount, promptTokens, completionTokens);

  const createdAt = new Date().toISOString();
  res.set({ 'Content-Type': 'application/json' });
  res.json({
    id: message_id,
    type: 'message',
    role: 'assistant',
    model,
    content: contentBlocks,
    stop_reason: stopReason,
    stop_sequence: null,
    created_at: createdAt,
    metadata: {},
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null
    }
  });
};

/**
 * Anthropic /v1/messages 主入口
 * @param {object} req - Express 请求
 * @param {object} res - Express 响应
 */
const handleAnthropicMessages = async (req, res) => {
  try {
    const compatibility = analyzeAnthropicCompatibility(req.body || {});
    const compatibilityHeaders = buildAnthropicCompatibilityHeaders(compatibility);
    if (Object.keys(compatibilityHeaders).length > 0) {
      res.set(compatibilityHeaders);
      logger.warn(
        `Anthropic compatibility notice: ${compatibility.summary}`,
        'ANTHROPIC'
      );
    }

    const built = await buildInternalRequest(req.body || {});
    const { body, hasTools, toolChoice, allowedToolNames, model } = built;

    const upstreamResp = await sendChatRequest(body);
    if (!upstreamResp.status || !upstreamResp.response) {
      return res.status(500).json({
        type: 'error',
        error: { type: 'api_error', message: upstreamResp.message || 'Request failed' }
      });
    }

    const message_id = `msg_${generateUUID().replace(/-/g, '').slice(0, 24)}`;
    const ctx = {
      message_id,
      model,
      hasTools,
      toolChoice,
      allowedToolNames,
      requestBody: body,
      currentAccount: upstreamResp.currentAccount
    };

    if (req.body?.stream) {
      await handleAnthropicStream(res, ctx, upstreamResp.response);
    } else {
      await handleAnthropicNonStream(res, ctx, upstreamResp.response);
    }
  } catch (error) {
    logger.error('Anthropic Messages 处理错误', 'ANTHROPIC', '', error);
    if (!res.headersSent) {
      res.status(500).json({
        type: 'error',
        error: { type: 'api_error', message: error.publicMessage || 'Service error' }
      });
    } else {
      if (!res.writableEnded) {
        try { writeAnthropicError(res, error.publicMessage || '上游响应处理失败', 'api_error'); } catch (_) { /* ignore */ }
      }
    }
  }
};

module.exports = {
  handleAnthropicMessages,
  analyzeAnthropicCompatibility,
  buildAnthropicCompatibilityHeaders,
  // 暴露内部辅助以便测试
  flattenAnthropicMessages,
  normalizeAnthropicTools,
  normalizeAnthropicToolChoice,
  normalizeAnthropicSystem,
  mapAnthropicStopReason,
  consumeUpstream,
  runWithAnthropicPing,
  handleAnthropicStream,
  handleAnthropicNonStream
};
