const test = require('node:test')
const assert = require('node:assert/strict')

const {
  buildToolSystemPrompt,
  foldToolMessages,
  looksLikeUnexecutedToolAction,
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator
} = require('../src/utils/tool-prompt.js')

test('Agent tool prompt forbids prose-only actions and premature completion', () => {
  const prompt = buildToolSystemPrompt([{
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write a file',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string', description: 'Target path' } },
        required: ['path']
      }
    }
  }])
  assert.match(prompt, /MUST be a `<tool_call>` block/)
  assert.match(prompt, /Only return a normal-language final answer after the requested task is genuinely complete/)
  assert.match(prompt, /path: string \/\* Target path \*\//)
  assert.equal(looksLikeUnexecutedToolAction('I will inspect the repository now.'), true)
  assert.equal(looksLikeUnexecutedToolAction('我将运行测试并检查结果。'), true)
  assert.equal(looksLikeUnexecutedToolAction('这里是无需调用工具的概念解释。'), false)
})

test('empty tool results remain visible in Agent history', () => {
  const folded = foldToolMessages([
    { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', function: { name: 'read_file', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'call_1', content: '' }
  ])
  assert.match(folded[1].content, />\nnull\n<\/tool_response>/)
})

test('legacy function_call and function result messages remain executable history', () => {
  const folded = foldToolMessages([
    { role: 'assistant', content: null, function_call: { name: 'read_file', arguments: '{"path":"README.md"}' } },
    { role: 'function', name: 'read_file', content: 'file body' }
  ])
  assert.equal(folded[0].role, 'assistant')
  assert.match(folded[0].content, /<tool_call>/)
  assert.match(folded[0].content, /"name":"read_file"/)
  assert.equal(folded[1].role, 'user')
  assert.match(folded[1].content, /<tool_response name="read_file">/)
  assert.match(folded[1].content, /file body/)
})

test('stream parser accepts split valid calls and preserves JSON string arguments', () => {
  const parser = createToolCallStreamParser({ allowedToolNames: ['read_file'] })
  const first = parser.push('before<tool_')
  const second = parser.push('call>{"name":"read_file","arguments":"{\\"path\\":\\"a\\"}"}</tool_call>')
  const tail = parser.flush()

  assert.equal(first.textDelta, 'before')
  assert.equal(second.completedCalls.length, 1)
  assert.equal(second.completedCalls[0].function.arguments, '{"path":"a"}')
  assert.equal(tail.textDelta, '')
  assert.equal(parser.hasParseError(), false)
})

test('truncated and unknown tool calls become explicit parser errors', () => {
  const truncated = createToolCallStreamParser({ allowedToolNames: ['read_file'] })
  truncated.push('<tool_call>{"name":"read_file","arguments":{"path":"a')
  truncated.flush()
  assert.equal(truncated.hasEmittedAnyCall(), false)
  assert.equal(truncated.hasParseError(), true)

  const unknown = parseToolCallsFromText(
    '<tool_call>{"name":"missing","arguments":{}}</tool_call>',
    { allowedToolNames: ['read_file'] }
  )
  assert.equal(unknown.toolCalls.length, 0)
  assert.equal(unknown.errors[0].type, 'unknown_tool')
})

test('native tool accumulator rebuilds fragmented OpenAI tool deltas', () => {
  const accumulator = createNativeToolCallAccumulator({ allowedToolNames: ['read_file'] })
  accumulator.push([{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '' } }])
  accumulator.push([{ index: 0, function: { arguments: '{"path":' } }])
  accumulator.push([{ index: 0, function: { arguments: '"a"}' } }])

  const calls = accumulator.finalize()
  assert.equal(calls.length, 1)
  assert.equal(calls[0].id, 'call_1')
  assert.equal(calls[0].function.arguments, '{"path":"a"}')
  assert.equal(accumulator.hasParseError(), false)
})

// 模型不总是照抄标签。这些变体以前都不匹配字面量，于是整段 XML 作为正文泄漏，
// 而且不记录任何错误 —— 既不 502 也不重试，调用方只看到裸 XML。
const TAG_VARIANTS = [
  ['大写', '<TOOL_CALL>{"name":"read_file","arguments":{}}</TOOL_CALL>'],
  ['标签内空白', '<tool_call >{"name":"read_file","arguments":{}}< /tool_call >'],
  ['复数', '<tool_calls>{"name":"read_file","arguments":{}}</tool_calls>'],
  ['混合大小写', '<Tool_Call>{"name":"read_file","arguments":{}}</Tool_call>']
]

test('tolerant tags: whole-text parser accepts case, spacing and plural variants', () => {
  for (const [label, text] of TAG_VARIANTS) {
    const result = parseToolCallsFromText(text, { allowedToolNames: ['read_file'] })
    assert.equal(result.toolCalls.length, 1, `${label}: 未识别为工具调用`)
    assert.equal(result.toolCalls[0].function.name, 'read_file', label)
    assert.equal(result.errors.length, 0, label)
    assert.equal(result.cleanedText, '', `${label}: XML 泄漏进了正文`)
  }
})

test('tolerant tags: stream parser accepts the same variants, split byte by byte', () => {
  for (const [label, text] of TAG_VARIANTS) {
    const parser = createToolCallStreamParser({ allowedToolNames: ['read_file'] })
    let visible = ''
    const calls = []
    for (const ch of text) {
      const out = parser.push(ch)
      visible += out.textDelta
      calls.push(...out.completedCalls)
    }
    const tail = parser.flush()
    visible += tail.textDelta
    calls.push(...tail.completedCalls)

    assert.equal(calls.length, 1, `${label}: 未识别为工具调用`)
    assert.equal(calls[0].function.name, 'read_file', label)
    assert.equal(visible, '', `${label}: XML 泄漏进了正文`)
    assert.equal(parser.hasParseError(), false, label)
  }
})

test('tolerant tags: ordinary prose with angle brackets still streams through', () => {
  const parser = createToolCallStreamParser({ allowedToolNames: ['read_file'] })
  const text = 'use <div>, and 1 < 2 < 3, and <toolbox> too'
  let visible = ''
  for (const ch of text) visible += parser.push(ch).textDelta
  visible += parser.flush().textDelta
  assert.equal(visible, text)
  assert.equal(parser.hasParseError(), false)
})

test('tolerant tags: a lone "<" is released by flush, not swallowed', () => {
  const parser = createToolCallStreamParser({ allowedToolNames: [] })
  // '<tool_ca' 是一个真标签的前缀，必须留在缓冲区等待后续 chunk……
  assert.equal(parser.push('cost <tool_ca').textDelta, 'cost ')
  // ……但流到此为止，flush 必须把它当正文放出来。
  assert.equal(parser.flush().textDelta, '<tool_ca')
})

test('tolerant tags: the tag-shaped buffer stays bounded on long text', () => {
  const parser = createToolCallStreamParser({ allowedToolNames: [] })
  const text = `${'x'.repeat(4000)}<${'y'.repeat(4000)}`
  const out = parser.push(text)
  // 只有末尾一个标签长度以内的片段可以被扣住。
  assert.ok(text.length - out.textDelta.length <= 24, '缓冲区超过了一个标签的长度')
  assert.equal(out.textDelta + parser.flush().textDelta, text)
})

test('tolerant tags: history is still written in the canonical form', () => {
  const folded = foldToolMessages([
    {
      role: 'assistant',
      tool_calls: [{ id: 'c1', function: { name: 'read_file', arguments: '{"path":"a"}' } }]
    }
  ])
  assert.match(folded[0].content, /<tool_call>/)
  assert.doesNotMatch(folded[0].content, /<tool_calls>/i)
})
