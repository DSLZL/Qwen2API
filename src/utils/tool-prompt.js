const { generateUUID } = require('./tools.js');
const { logger } = require('./logger');
const {
  AGENT_FINAL_OPEN,
  AGENT_FINAL_CLOSE,
  AGENT_BLOCKED_OPEN,
  AGENT_BLOCKED_CLOSE
} = require('./agent-turn.js');

/**
 * 工具调用 XML 起始标签
 * @type {string}
 */
const TOOL_CALL_OPEN = '<tool_call>';

/**
 * 工具调用 XML 结束标签
 * @type {string}
 */
const TOOL_CALL_CLOSE = '</tool_call>';

/**
 * 宽松的标签识别。模型偶尔写成 `<TOOL_CALL>`、`<tool_call >` 或 `<tool_calls>`；
 * 这些都不等于字面量，于是整段 XML 作为正文泄漏给客户端，而且**不记录任何错误** ——
 * 既不触发 502 也不触发补偿重试，调用方只看到一段裸 XML。
 *
 * 只放宽三点：大小写、标签内空白、复数 `s`。故意不接受任意属性，好让标签长度有上界；
 * 流式解析要在 chunk 边界上暂存可能被切断的标签，无界的标签会让缓冲区也无界。
 * 因此 `<tool_call id="1">` 仍然会泄漏，是已知且有意的缺口。
 *
 * 只影响**读取**。foldToolMessages 回写历史时仍然使用上面的规范形式。
 */
const TOOL_CALL_OPEN_RE = /<[ \t]{0,4}tool_calls?[ \t]{0,4}>/i;
const TOOL_CALL_CLOSE_RE = /<[ \t]{0,4}\/[ \t]{0,4}tool_calls?[ \t]{0,4}>/i;

/** 上面两个正则能匹配的最长标签，用作 chunk 边界暂存区的上界。 */
const TOOL_CALL_TAG_MAX = '</    tool_calls    >'.length;

const normalizeAllowedToolNames = (allowedToolNames) => {
  if (!allowedToolNames) return null;
  const names = allowedToolNames instanceof Set ? allowedToolNames : new Set(allowedToolNames);
  return names.size > 0 ? names : null;
};

const serializeToolArguments = (args) => {
  if (typeof args === 'string') {
    try {
      JSON.parse(args);
      return args;
    } catch (_) {
      return JSON.stringify(args);
    }
  }
  return JSON.stringify(args ?? {});
};

const compactDescription = (value, maxLength = 320) => {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
};

/**
 * 识别模型用“我将执行/Let me inspect”代替真实工具调用的占位回复。
 * 仅匹配明确的动作动词，避免把普通解释或建议误判成工具回合。
 */
const looksLikeUnexecutedToolAction = (value) => {
  const text = String(value || '').trim().replace(/^[#>*\-\s]+/, '');
  const english = /^(?:i(?:['’]ll| will)|let me|i need to|next,?\s+i(?:['’]ll| will))\s+(?:now\s+)?(?:run|execute|check|inspect|read|edit|write|search|open|call|use|look|test|verify|build|deploy|create|update|fetch)\b/i;
  const chinese = /^(?:我(?:将|会|先|需要|正在)|让我|接下来(?:我)?(?:将|会|先)?|现在(?:我)?(?:将|会|先|来)?|下面(?:我)?(?:将|会|先)?|正在)(?:立即|马上|先|来)?(?:运行|执行|检查|查看|读取|编辑|修改|写入|搜索|打开|调用|使用|测试|验证|构建|部署|创建|更新|获取)/;
  return english.test(text) || chinese.test(text);
};

const createToolCallObject = (payload, index = 0, id = null) => ({
  index,
  id: id || `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`,
  type: 'function',
  function: {
    name: payload.name,
    arguments: serializeToolArguments(payload.arguments)
  }
});

/**
 * 将 JSON Schema 类型压缩为简短 TypeScript 风格签名
 * @param {Object} schema - JSON Schema 节点
 * @returns {string} TS 风格类型表示
 */
const compressSchemaType = (schema) => {
  if (!schema || typeof schema !== 'object') {
    return 'any';
  }

  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return schema.enum.map(value => JSON.stringify(value)).join(' | ');
  }

  const type = schema.type;

  if (type === 'array') {
    const itemType = compressSchemaType(schema.items);
    return `${itemType}[]`;
  }

  if (type === 'object') {
    if (!schema.properties || typeof schema.properties !== 'object') {
      return 'object';
    }
    const requiredKeys = new Set(Array.isArray(schema.required) ? schema.required : []);
    const fields = Object.entries(schema.properties).map(([key, value]) => {
      const optional = requiredKeys.has(key) ? '' : '?';
      const description = compactDescription(value?.description, 180);
      return `${key}${optional}: ${compressSchemaType(value)}${description ? ` /* ${description.replace(/\*\//g, '* /')} */` : ''}`;
    });
    return `{ ${fields.join('; ')} }`;
  }

  if (Array.isArray(type)) {
    return type.map(t => compressSchemaType({ ...schema, type: t })).join(' | ');
  }

  return type || 'any';
};

/**
 * 将单个工具定义压缩为 TS 风格签名
 * @param {Object} tool - OpenAI 工具定义
 * @returns {string} 压缩后的工具描述
 */
const compressToolDefinition = (tool) => {
  const fn = tool?.function || tool;
  const name = fn?.name || 'unknown';
  const description = compactDescription(fn?.description);
  const params = fn?.parameters || { type: 'object', properties: {} };
  const signature = compressSchemaType(params);

  if (description) {
    return `- ${name}${signature}\n  ${description}`;
  }
  return `- ${name}${signature}`;
};

/**
 * 构建用于注入 system 消息的工具调用提示词
 * @param {Array<Object>} tools - OpenAI 风格工具定义列表
 * @param {Object} [options] - 可选参数
 * @param {string|Object} [options.tool_choice] - OpenAI tool_choice 参数
 * @returns {string} 完整的工具调用系统提示词
 */
const buildToolSystemPrompt = (tools, options = {}) => {
  if (!Array.isArray(tools) || tools.length === 0) {
    return '';
  }

  const compressed = tools
    .map(compressToolDefinition)
    .filter(Boolean)
    .join('\n');

  const lines = [
    '# Tools',
    '',
    'You have access to the following tools. This is an Agent tool protocol, not a suggestion.',
    '',
    '## Available tools',
    compressed,
    '',
    '## Output format',
    'Emit each tool invocation as:',
    '',
    '<tool_call>',
    '{"name": "<tool_name>", "arguments": {<json_arguments>}}',
    '</tool_call>',
    '',
    'Tool results are delivered back to you as user messages wrapped like this:',
    '',
    '<tool_response tool_call_id="<id>" name="<tool_name>">',
    '<result text or JSON>',
    '</tool_response>',
    '',
    'Rules:',
    '- If the task requires reading, writing, editing, searching, shell execution, browser use, or any action covered by an available tool, your visible response MUST be a `<tool_call>` block. Call the tool instead of describing the action.',
    '- A tool call must be the first non-whitespace content of the visible answer. Do not write “I will…”, “Let me…”, “我将…”, “正在…”, a plan, or a completion claim before it.',
    '- The JSON inside `<tool_call>` must be valid and on a single logical block.',
    '- Use the exact tool name listed above.',
    '- Provide all required arguments; omit unknown ones.',
    '- You may emit multiple `<tool_call>` blocks back-to-back when more than one tool is needed.',
    '- After every tool result, evaluate the actual task state. If work remains, emit the next tool call. Only return a normal-language final answer after the requested task is genuinely complete or you are blocked on user input.',
    '- Never claim that a file was changed, a command succeeded, or a result was verified unless the corresponding tool result proves it.',
    '- Do not call nonexistent tools, fabricate tool results, wrap `<tool_call>` in code fences, or mix extra commentary into a tool-call turn.',
    '- A non-tool response is valid only when it explicitly declares its state: use the completion or blocked wrapper below. Bare prose is invalid.',
    `- Verified completion: ${AGENT_FINAL_OPEN}final report${AGENT_FINAL_CLOSE}`,
    `- Requires user input/authority: ${AGENT_BLOCKED_OPEN}exact blocker${AGENT_BLOCKED_CLOSE}`,
    '- Never emit the completion wrapper after merely finishing one intermediate tool action; continue with another tool call until every requested outcome is verified.'
  ];

  const choice = options.tool_choice;
  if (choice === 'required') {
    lines.push('- You MUST call at least one tool before answering.');
  } else if (choice && typeof choice === 'object' && choice.function?.name) {
    lines.push(`- You MUST call the tool \`${choice.function.name}\` first.`);
  } else if (choice === 'none') {
    lines.push('- Do NOT call any tool for this turn; respond as plain text.');
  }

  return lines.join('\n');
};

/**
 * 将历史中的 assistant tool_calls / tool 角色消息折叠成纯文本，
 * 以便上游网页接口（仅识别 user/assistant/system）能正确接收上下文。
 * 折叠时保留原始 tool_call_id，并将后续 role=tool 消息按 id 精确回链。
 * @param {Array<Object>} messages - 原始 OpenAI 风格消息数组
 * @returns {Array<Object>} 折叠后的消息数组
 */
const foldToolMessages = (messages) => {
  if (!Array.isArray(messages)) return messages;

  const callIdToName = new Map();

  return messages.map((message) => {
    if (!message || typeof message !== 'object') return message;

    const assistantCalls = message.role === 'assistant'
      ? (Array.isArray(message.tool_calls) && message.tool_calls.length > 0
        ? message.tool_calls
        : (message.function_call?.name ? [message.function_call] : []))
      : [];
    if (assistantCalls.length > 0) {
      const blocks = assistantCalls.map((call) => {
        const fn = call?.function || call;
        let args = fn?.arguments;
        if (typeof args === 'string') {
          try {
            args = JSON.parse(args);
          } catch (_) {
            // 保留原始字符串形式
          }
        }
        const name = fn?.name || 'unknown';
        const id = call?.id || `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`;
        callIdToName.set(id, name);
        const payload = { id, name, arguments: args ?? {} };
        return `${TOOL_CALL_OPEN}\n${JSON.stringify(payload)}\n${TOOL_CALL_CLOSE}`;
      });
      const original = typeof message.content === 'string' ? message.content : '';
      return {
        role: 'assistant',
        content: [original, blocks.join('\n')].filter(Boolean).join('\n')
      };
    }

    if (message.role === 'tool' || message.role === 'function') {
      const callId = message.tool_call_id || '';
      const name = message.name || callIdToName.get(callId) || (message.role === 'function' ? 'function' : 'tool');
      const content = typeof message.content === 'string'
        ? (message.content || 'null')
        : JSON.stringify(message.content ?? null);
      const idAttr = callId ? ` tool_call_id="${escapeAttr(callId)}"` : '';
      return {
        role: 'user',
        content: `<tool_response${idAttr} name="${escapeAttr(name)}">\n${content}\n</tool_response>`
      };
    }

    return message;
  });
};

/**
 * 转义 XML 属性中的特殊字符
 * @param {string} value - 原始字符串
 * @returns {string} 转义后的字符串
 */
const escapeAttr = (value) => String(value || '')
  .replace(/&/g, '&amp;')
  .replace(/"/g, '&quot;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

/**
 * 解析单段 `<tool_call>...</tool_call>` 内的 JSON 负载
 * @param {string} raw - 标签内的原始字符串
 * @returns {{ name: string, arguments: Object }|null} 解析结果
 */
const parseToolCallPayload = (raw) => {
  if (!raw) return null;

  let text = raw.trim();
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) {
    text = fenceMatch[1].trim();
  }

  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== 'object') return null;
    const name = parsed.name || parsed.tool || parsed.function;
    const args = parsed.arguments ?? parsed.parameters ?? parsed.args ?? {};
    if (!name) return null;
    return { name: String(name), arguments: args };
  } catch (error) {
    logger.warning?.('解析 tool_call 负载失败', 'TOOL', text, error?.message);
    return null;
  }
};

/**
 * 从完整文本中提取所有工具调用块
 * @param {string} fullText - 模型完整输出
 * @param {Object} [options]
 * @param {Set<string>|Array<string>} [options.allowedToolNames]
 * @returns {{ cleanedText: string, toolCalls: Array<Object>, errors: Array<Object> }} 抽取结果
 */
const parseToolCallsFromText = (fullText, options = {}) => {
  if (typeof fullText !== 'string' || !TOOL_CALL_OPEN_RE.test(fullText)) {
    return { cleanedText: fullText || '', toolCalls: [], errors: [] };
  }

  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  const toolCalls = [];
  const errors = [];
  // 每次调用新建：带 /g 的正则会保留 lastIndex，模块级共享会让第二次调用漏掉开头的匹配。
  const pattern = new RegExp(
    `${TOOL_CALL_OPEN_RE.source}([\\s\\S]*?)${TOOL_CALL_CLOSE_RE.source}`,
    'gi'
  );
  const cleanedText = fullText.replace(pattern, (_, inner) => {
    const payload = parseToolCallPayload(inner);
    if (!payload) {
      errors.push({ type: 'invalid_json', raw: inner });
    } else if (allowedToolNames && !allowedToolNames.has(payload.name)) {
      errors.push({ type: 'unknown_tool', name: payload.name });
    } else {
      toolCalls.push(createToolCallObject(payload, toolCalls.length));
    }
    return '';
  });

  const unclosed = cleanedText.match(TOOL_CALL_OPEN_RE);
  const unclosedIndex = unclosed ? unclosed.index : -1;
  let finalText = cleanedText;
  if (unclosedIndex !== -1) {
    const raw = cleanedText.slice(unclosedIndex + unclosed[0].length);
    const payload = parseToolCallPayload(raw);
    if (!payload) {
      errors.push({ type: 'truncated_tool_call', raw });
    } else if (allowedToolNames && !allowedToolNames.has(payload.name)) {
      errors.push({ type: 'unknown_tool', name: payload.name });
    } else {
      toolCalls.push(createToolCallObject(payload, toolCalls.length));
    }
    finalText = cleanedText.slice(0, unclosedIndex);
  }

  return { cleanedText: finalText.trim(), toolCalls, errors };
};

/**
 * 创建增量式工具调用流解析器
 * 接收 content delta，识别 `<tool_call>` 块边界，
 * 对外吐出文本增量与已完成的工具调用对象。
 * @returns {{
 *   push: (chunk: string) => { textDelta: string, completedCalls: Array<Object> },
 *   flush: () => { textDelta: string, completedCalls: Array<Object> },
 *   hasPendingCall: () => boolean,
 *   hasEmittedAnyCall: () => boolean
 * }} 解析器实例
 */
const createToolCallStreamParser = (options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  let pendingText = '';
  let inToolCall = false;
  let toolCallBuffer = '';
  let emittedCallCount = 0;
  const errors = [];

  const acceptPayload = (payload, raw, result) => {
    if (!payload) {
      errors.push({ type: 'invalid_json', raw });
      return;
    }
    if (allowedToolNames && !allowedToolNames.has(payload.name)) {
      errors.push({ type: 'unknown_tool', name: payload.name });
      return;
    }
    result.completedCalls.push(createToolCallObject(payload, emittedCallCount));
    emittedCallCount += 1;
  };

  /**
   * 在等待标签出现时，安全地输出已确定不是标签前缀的部分
   * @param {string} text - 当前累积的文本
   * @returns {{ safe: string, remainder: string }} 切分结果
   */
  const splitSafeText = (text) => {
    const openMatch = text.match(TOOL_CALL_OPEN_RE);
    if (openMatch) {
      return { safe: text.slice(0, openMatch.index), remainder: text.slice(openMatch.index) };
    }
    // 标签可能被切在两个 chunk 中间。宽松匹配无法像字面量那样逐前缀试探，改为按上界暂存：
    // 从最后一个 '<' 起若不超过一个标签的长度，就留到下一段再判断。正文里孤立的 '<'
    // 最多延迟 TOOL_CALL_TAG_MAX 个字符，flush() 兜底放出。
    const lastOpen = text.lastIndexOf('<');
    if (lastOpen !== -1 && text.length - lastOpen <= TOOL_CALL_TAG_MAX) {
      return { safe: text.slice(0, lastOpen), remainder: text.slice(lastOpen) };
    }
    return { safe: text, remainder: '' };
  };

  const push = (chunk) => {
    const result = { textDelta: '', completedCalls: [] };
    if (typeof chunk !== 'string' || chunk.length === 0) return result;

    let buffer = chunk;

    while (buffer.length > 0) {
      if (inToolCall) {
        toolCallBuffer += buffer;
        buffer = '';
        const closeMatch = toolCallBuffer.match(TOOL_CALL_CLOSE_RE);
        if (!closeMatch) {
          break;
        }
        const inner = toolCallBuffer.slice(0, closeMatch.index);
        buffer = toolCallBuffer.slice(closeMatch.index + closeMatch[0].length);
        toolCallBuffer = '';
        const payload = parseToolCallPayload(inner);
        acceptPayload(payload, inner, result);
        inToolCall = false;
        continue;
      }

      pendingText += buffer;
      buffer = '';

      const openMatch = pendingText.match(TOOL_CALL_OPEN_RE);
      if (openMatch) {
        const before = pendingText.slice(0, openMatch.index);
        if (before) result.textDelta += before;
        const tail = pendingText.slice(openMatch.index + openMatch[0].length);
        pendingText = '';
        inToolCall = true;
        buffer = tail;
        continue;
      }

      const { safe, remainder } = splitSafeText(pendingText);
      if (safe) result.textDelta += safe;
      pendingText = remainder;
    }

    return result;
  };

  const flush = () => {
    const result = { textDelta: '', completedCalls: [] };
    if (inToolCall && toolCallBuffer) {
      const payload = parseToolCallPayload(toolCallBuffer);
      acceptPayload(payload, toolCallBuffer, result);
      toolCallBuffer = '';
      inToolCall = false;
    }
    if (pendingText) {
      result.textDelta += pendingText;
      pendingText = '';
    }
    return result;
  };

  return {
    push,
    flush,
    hasPendingCall: () => inToolCall,
    hasEmittedAnyCall: () => emittedCallCount > 0,
    hasParseError: () => errors.length > 0,
    getErrors: () => [...errors]
  };
};

/**
 * 累积 OpenAI 原生 delta.tool_calls。网页上游一旦开始原生返回工具调用，桥接层无需再依赖 XML。
 */
const createNativeToolCallAccumulator = (options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  const calls = new Map();
  const errors = [];

  const push = (deltas) => {
    if (!Array.isArray(deltas)) return;
    for (const delta of deltas) {
      if (!delta || typeof delta !== 'object') continue;
      const index = Number.isInteger(delta.index) ? delta.index : calls.size;
      const current = calls.get(index) || {
        index,
        id: delta.id || null,
        type: delta.type || 'function',
        function: { name: '', arguments: '' }
      };
      if (delta.id) current.id = delta.id;
      if (delta.type) current.type = delta.type;
      if (typeof delta.function?.name === 'string' && delta.function.name) {
        const incomingName = delta.function.name;
        if (!current.function.name) {
          current.function.name = incomingName;
        } else if (incomingName === current.function.name || current.function.name.endsWith(incomingName)) {
          // 某些兼容上游会在每个 delta 重复完整 name，不能重复拼接。
        } else if (incomingName.startsWith(current.function.name)) {
          current.function.name = incomingName;
        } else {
          current.function.name += incomingName;
        }
      }
      if (typeof delta.function?.arguments === 'string') current.function.arguments += delta.function.arguments;
      calls.set(index, current);
    }
  };

  const finalize = () => {
    const finalized = [];
    for (const [index, call] of [...calls.entries()].sort((a, b) => a[0] - b[0])) {
      if (!call.function.name) {
        errors.push({ type: 'missing_tool_name', index });
        continue;
      }
      if (allowedToolNames && !allowedToolNames.has(call.function.name)) {
        errors.push({ type: 'unknown_tool', name: call.function.name });
        continue;
      }
      try {
        JSON.parse(call.function.arguments || '{}');
      } catch (_) {
        errors.push({ type: 'invalid_arguments', name: call.function.name });
        continue;
      }
      finalized.push({
        index: finalized.length,
        id: call.id || `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`,
        type: 'function',
        function: {
          name: call.function.name,
          arguments: call.function.arguments || '{}'
        }
      });
    }
    return finalized;
  };

  return {
    push,
    finalize,
    hasAny: () => calls.size > 0,
    hasParseError: () => errors.length > 0,
    getErrors: () => [...errors]
  };
};

module.exports = {
  TOOL_CALL_OPEN,
  TOOL_CALL_CLOSE,
  buildToolSystemPrompt,
  foldToolMessages,
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator,
  looksLikeUnexecutedToolAction,
  serializeToolArguments
};
