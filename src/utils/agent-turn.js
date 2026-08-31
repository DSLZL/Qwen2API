const AGENT_FINAL_OPEN = '<agent_final>'
const AGENT_FINAL_CLOSE = '</agent_final>'
const AGENT_BLOCKED_OPEN = '<agent_blocked>'
const AGENT_BLOCKED_CLOSE = '</agent_blocked>'

// 工具调用的规范标记。定义在这里（依赖图的叶子），tool-prompt.js 和各重试提示共同引用，
// 保证提示词、折叠回写和重试提示永远教同一种形式。
//
// 为什么不是 <tool_call>：那是 Qwen 平台的**原生**格式，而原生意味着平台自己的
// server-side agent loop 也在盯着它 —— 模型一吐出来就被拦截，拿去查平台自己的
// tool registry（里面没有我们的工具），然后把 "Tool <name> does not exists" 塞回
// 模型的生成上下文。模型看到"工具全坏了"，就放弃调用改为口头汇报失败。
// 实测：2026-08-30 19:56 的会话死亡与 5 条 role:function 拦截逐秒对应，名字正是
// "Bash"/"Read"；auto_search:false 也关不掉这个拦截器（18/18 探针通过但拦截照发）。
// 换成平台不认识的标记，拦截器就出局了。旧尖括号形式在读取侧仍然被识别（RL 惯性
// 输出），只是不再教、不再写 —— 见 tool-prompt.js 的 TOOL_CALL_TRIGGER_RE。
const TOOL_CALL_OPEN = '[TOOL CALL]'
const TOOL_CALL_CLOSE = '[END TOOL CALL]'

const escapeRegExp = (value) => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const unwrapExactTag = (value, openTag, closeTag) => {
  const pattern = new RegExp(
    `^\\s*${escapeRegExp(openTag)}([\\s\\S]*?)${escapeRegExp(closeTag)}\\s*$`,
    'i'
  )
  const matched = String(value || '').match(pattern)
  return matched ? matched[1].trim() : null
}

/**
 * Agent 请求的可见输出必须明确声明本回合是“已完成”还是“需要用户输入”。
 * 工具调用由 tool-prompt 解析器先行抽取，因此这里仅处理剩余文本。
 */
const parseAgentControlText = (value) => {
  const raw = String(value || '')
  const trimmed = raw.trim()
  if (!trimmed) return { kind: 'empty', text: '' }

  const finalText = unwrapExactTag(trimmed, AGENT_FINAL_OPEN, AGENT_FINAL_CLOSE)
  if (finalText !== null) return { kind: 'final', text: finalText }

  const blockedText = unwrapExactTag(trimmed, AGENT_BLOCKED_OPEN, AGENT_BLOCKED_CLOSE)
  if (blockedText !== null) return { kind: 'blocked', text: blockedText }

  if (/<\/?agent_(?:final|blocked)>/i.test(trimmed)) {
    return { kind: 'invalid_control', text: trimmed }
  }
  return { kind: 'bare', text: trimmed }
}

/**
 * 增量识别严格 Agent 的 final/blocked 包装。只有开标签位于首个非空白位置时
 * 才开放正文；闭标签及其可能跨 chunk 的前缀始终留在缓冲区。
 *
 * 正文采用与 parseAgentControlText 相同的 trim 语义：丢弃包装内的首尾空白，
 * 中间空白仍按原顺序增量输出。完整合法性最终仍由 parseAgentControlText 判定。
 */
const createAgentControlStreamParser = () => {
  const modes = [
    { kind: 'final', open: AGENT_FINAL_OPEN, close: AGENT_FINAL_CLOSE },
    { kind: 'blocked', open: AGENT_BLOCKED_OPEN, close: AGENT_BLOCKED_CLOSE }
  ]
  let state = 'prefix'
  let mode = null
  let pending = ''
  let bodyStarted = false
  let trailingWhitespace = ''
  let emittedText = false
  let invalid = false

  const createResult = () => ({
    textDelta: '',
    kind: mode?.kind || null,
    opened: false,
    closed: state === 'closed' && !invalid,
    invalid
  })

  const appendBodyText = (value, final, result) => {
    let text = String(value || '')
    if (!bodyStarted) {
      text = text.replace(/^\s+/, '')
      if (!text) {
        if (final) trailingWhitespace = ''
        return
      }
      bodyStarted = true
    }

    const combined = `${trailingWhitespace}${text}`
    const trailing = combined.match(/\s+$/)?.[0] || ''
    const safe = trailing ? combined.slice(0, -trailing.length) : combined
    if (safe) {
      result.textDelta += safe
      emittedText = true
    }
    trailingWhitespace = final ? '' : trailing
  }

  const splitClosePrefix = (value, closeTag) => {
    const lower = value.toLowerCase()
    const close = closeTag.toLowerCase()
    const maxLength = Math.min(lower.length, close.length - 1)
    for (let length = maxLength; length > 0; length--) {
      if (close.startsWith(lower.slice(-length))) {
        return {
          safe: value.slice(0, -length),
          remainder: value.slice(-length)
        }
      }
    }
    return { safe: value, remainder: '' }
  }

  const processBody = (result) => {
    const closeTag = mode.close
    const closeIndex = pending.toLowerCase().indexOf(closeTag.toLowerCase())
    if (closeIndex !== -1) {
      appendBodyText(pending.slice(0, closeIndex), true, result)
      pending = pending.slice(closeIndex + closeTag.length)
      state = 'closed'
      result.closed = true
      if (pending.trim()) {
        invalid = true
        state = 'invalid'
        result.invalid = true
        result.closed = false
      }
      return
    }

    const split = splitClosePrefix(pending, closeTag)
    pending = split.remainder
    appendBodyText(split.safe, false, result)
  }

  const processPrefix = (result) => {
    const leadingLength = pending.match(/^\s*/)?.[0].length || 0
    const candidate = pending.slice(leadingLength)
    if (!candidate) return
    const lowerCandidate = candidate.toLowerCase()
    const matchedMode = modes.find(item => lowerCandidate.startsWith(item.open.toLowerCase()))
    if (matchedMode) {
      mode = matchedMode
      pending = candidate.slice(matchedMode.open.length)
      state = 'body'
      result.kind = mode.kind
      result.opened = true
      processBody(result)
      return
    }

    const isOpenPrefix = modes.some(item => item.open.toLowerCase().startsWith(lowerCandidate))
    if (!isOpenPrefix) {
      invalid = true
      state = 'invalid'
      result.invalid = true
    }
  }

  const push = (chunk) => {
    const result = createResult()
    if (typeof chunk !== 'string' || chunk.length === 0 || state === 'invalid') return result
    pending += chunk
    if (state === 'prefix') processPrefix(result)
    else if (state === 'body') processBody(result)
    else if (state === 'closed' && pending.trim()) {
      invalid = true
      state = 'invalid'
      result.invalid = true
      result.closed = false
    }
    result.kind = mode?.kind || result.kind
    return result
  }

  const flush = () => {
    const result = createResult()
    if (state !== 'closed' || pending.trim()) {
      invalid = true
      state = 'invalid'
      result.invalid = true
      result.closed = false
    }
    result.kind = mode?.kind || null
    return result
  }

  return {
    push,
    flush,
    getState: () => ({
      kind: mode?.kind || null,
      opened: mode !== null,
      closed: state === 'closed' && !invalid,
      invalid,
      hasEmittedText: emittedText
    })
  }
}

const AGENT_CONTROL_TAGS = [
  AGENT_FINAL_OPEN,
  AGENT_FINAL_CLOSE,
  AGENT_BLOCKED_OPEN,
  AGENT_BLOCKED_CLOSE
]

/**
 * 从可见正文中剥离 Agent 回合包装标签。
 *
 * /v1/messages 注入的是与 OpenAI 路径同一份工具提示词，所以模型同样会输出
 * <agent_final>...</agent_final>；但 Anthropic 控制器没有接 Agent 回合门禁，
 * 标签因此原样透传给客户端。这里只做剥离，不做合法性判定：没有门禁就没有
 * 重生回合的地方，把“散文 + 包装”的回合判成 invalid 只会让整个回合失败。
 *
 * 流式必须缓冲：标签可能被切在两个 chunk 中间。push 只返回确定不属于标签的
 * 前缀，结束时必须调用一次 flush 取回缓冲区，否则末尾文本会丢。缓冲区最多
 * 保留一个标签长度的前缀，不会随流增长。
 */
const createAgentTagStripper = () => {
  let pending = ''

  const push = (chunk) => {
    if (typeof chunk !== 'string' || !chunk) return ''
    pending += chunk
    let out = ''
    for (;;) {
      const start = pending.indexOf('<')
      if (start === -1) {
        out += pending
        pending = ''
        return out
      }
      out += pending.slice(0, start)
      pending = pending.slice(start)

      const lower = pending.toLowerCase()
      const matched = AGENT_CONTROL_TAGS.find(tag => lower.startsWith(tag.toLowerCase()))
      if (matched) {
        pending = pending.slice(matched.length)
        continue
      }
      // 可能是被 chunk 边界切断的标签前缀：留在缓冲区等下一段。
      if (AGENT_CONTROL_TAGS.some(tag => tag.toLowerCase().startsWith(lower))) return out
      // 确定不是标签：'<' 属于正文，跳过它继续扫描。
      out += '<'
      pending = pending.slice(1)
    }
  }

  const flush = () => {
    const rest = pending
    pending = ''
    return rest
  }

  return { push, flush }
}

const stripAgentTags = (value) => {
  const stripper = createAgentTagStripper()
  return `${stripper.push(String(value || ''))}${stripper.flush()}`
}

const buildAgentTurnDirective = ({ afterToolResult = false } = {}) => {
  const continuation = afterToolResult
    ? 'The current message is a tool result from the same unfinished task. It is evidence to inspect, not a new task and not a reason to stop after one action.'
    : 'Treat this request as one step in an Agent task. Recover the original acceptance criteria from the conversation before deciding whether the task is complete.'

  return [
    '# Agent loop control (highest-priority output contract)',
    continuation,
    'The client executes tools and automatically sends each tool result back in the next request. Keep that loop alive until the original task is genuinely complete.',
    'Before responding, check the original request, every claimed deliverable, failures in tool results, and whether verification is still missing.',
    'Your entire visible response MUST be exactly one of these modes:',
    `1. If any action, inspection, edit, command, test, retry, or verification remains: emit one or more valid \`${TOOL_CALL_OPEN}...${TOOL_CALL_CLOSE}\` blocks and no prose.`,
    `2. Only when every requested outcome is complete and supported by tool-result evidence: emit ${AGENT_FINAL_OPEN}a concise final report${AGENT_FINAL_CLOSE}.`,
    `3. Only when progress is impossible without new user input or authority: emit ${AGENT_BLOCKED_OPEN}the exact blocker and required input${AGENT_BLOCKED_CLOSE}.`,
    'Bare prose, a plan, a progress update, hidden reasoning without visible output, or a claim such as “done” without the completion wrapper is an invalid Agent turn and will be regenerated.',
    'Never use the completion wrapper merely because one tool call finished. If verification has not run or any requested work remains, call the next tool.'
  ].join('\n')
}

const buildAgentRetryHint = (reason = 'incomplete') => {
  const reasonText = {
    empty: 'The previous attempt ended without a visible answer or executable tool call.',
    bare: 'The previous attempt returned bare prose without declaring a verified final result or emitting the next tool call.',
    invalid_control: 'The previous attempt used a malformed or mixed Agent completion wrapper.',
    invalid_tool_call: 'The previous attempt contained an invalid, truncated, or unknown tool call.',
    required_tool: 'The previous attempt violated tool_choice and did not call the required tool.'
  }[reason] || 'The previous attempt did not produce a valid Agent turn.'

  return [
    '# Agent turn recovery',
    reasonText,
    'Continue the SAME original task. Re-check its acceptance criteria and the latest tool result.',
    `If work remains, output only valid \`${TOOL_CALL_OPEN}...${TOOL_CALL_CLOSE}\` blocks. If and only if all work is verified complete, output ${AGENT_FINAL_OPEN}the final report${AGENT_FINAL_CLOSE}.`,
    `If user input is strictly required, output ${AGENT_BLOCKED_OPEN}the blocker${AGENT_BLOCKED_CLOSE}. Do not output bare planning prose.`
  ].join('\n')
}

module.exports = {
  AGENT_FINAL_OPEN,
  AGENT_FINAL_CLOSE,
  AGENT_BLOCKED_OPEN,
  AGENT_BLOCKED_CLOSE,
  TOOL_CALL_OPEN,
  TOOL_CALL_CLOSE,
  parseAgentControlText,
  createAgentControlStreamParser,
  createAgentTagStripper,
  stripAgentTags,
  buildAgentTurnDirective,
  buildAgentRetryHint
}
