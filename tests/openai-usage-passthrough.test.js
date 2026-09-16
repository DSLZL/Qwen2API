// Reported usage = upstream usage en /v1/chat/completions (stream, non-stream y agent runtime).
// Gemelo de tests/anthropic-usage-passthrough.test.js: Qwen manda `usage` con nombres
// DashScope (input_tokens / output_tokens), acumulado por frame; el proxy leia
// prompt_tokens / completion_tokens y caia SIEMPRE al estimado local.
//
// Harness copiado de tests/openai-residue.test.js (cada archivo corre en su propio proceso).

const test = require('node:test');
const { describe, it } = test;
const assert = require('node:assert/strict');

// El harness requiere account.js, que en modo file haria login real con data/data.json.
process.env.API_KEY = 'usage-test-key';
process.env.DATA_SAVE_MODE = 'none';
process.env.ACCOUNTS = '';
process.env.ENABLE_CLI = 'false';

// Sin red en tests: mismos parches de require-cache que el resto de la suite.
const modelsMap = require('../src/models/models-map.js');
modelsMap.getLatestModels = async () => { throw new Error('offline test: no model fetch'); };
const requestModule = require('../src/utils/request.js');
requestModule.sendChatRequest = async () => ({ status: false });

const { handleStreamResponse, handleNonStreamResponse } = require('../src/controllers/chat.js');

test.after(() => {
  require('../src/utils/account.js').destroy();
});

const createMockResponse = () => ({
  output: '',
  headers: {},
  headersSent: false,
  writableEnded: false,
  statusCode: 200,
  set(headers) { Object.assign(this.headers, headers); return this; },
  setHeader(name, value) { this.headers[name] = value; },
  write(chunk) { this.headersSent = true; this.output += String(chunk); return true; },
  end(chunk = '') { if (chunk) this.write(chunk); this.writableEnded = true; },
  status(code) { this.statusCode = code; return this; },
  json(value) {
    this.headersSent = true;
    this.output += JSON.stringify(value);
    this.writableEnded = true;
    return this;
  }
});

/** Frame como lo manda Qwen: `usage` al nivel del frame, junto a `choices`. */
const frame = (content, usage) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }],
  ...(usage === undefined ? {} : { usage })
})}\n\n`;

const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

const upstreamOf = (frames) => {
  async function* gen() {
    for (const f of frames) yield f;
  }
  return gen();
};

const deltasOf = (output) => output
  .split('\n\n')
  .filter(Boolean)
  .map(chunk => chunk.replace(/^data: /, ''))
  .filter(payload => payload && payload !== '[DONE]')
  .map(payload => JSON.parse(payload));

const QWEN_USAGE = [
  frame('Hel', { input_tokens: 651, output_tokens: 1, total_tokens: 652 }),
  frame('lo', { input_tokens: 651, output_tokens: 2, total_tokens: 653 }),
  STOP
];
const NO_USAGE = [frame('Hello there'), STOP];
const PARTIAL_USAGE = [frame('Hello there', { input_tokens: 651 }), STOP];
const ZERO_USAGE = [frame('Hello there', { input_tokens: 0, output_tokens: 0, total_tokens: 0 }), STOP];
const REQUEST_BODY = { messages: [{ role: 'user', content: 'hi' }] };

const lastStreamUsage = (output) => deltasOf(output).map(e => e.usage).filter(Boolean).pop();

const runStream = async (frames, options = { has_tools: false }) => {
  const res = createMockResponse();
  await handleStreamResponse(res, upstreamOf(frames), false, false, REQUEST_BODY, options);
  return lastStreamUsage(res.output);
};

const runNonStream = async (frames, options = { has_tools: false }) => {
  const res = createMockResponse();
  await handleNonStreamResponse(res, upstreamOf(frames), false, false, 'qwen-test', REQUEST_BODY, options);
  assert.equal(res.statusCode, 200, res.output);
  return JSON.parse(res.output).usage;
};

const assertEstimated = (usage) => {
  assert.ok(usage, 'usage object present');
  assert.ok(usage.prompt_tokens > 0, `prompt_tokens=${usage.prompt_tokens}`);
  assert.ok(usage.completion_tokens > 0, `completion_tokens=${usage.completion_tokens}`);
  assert.equal(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens);
};

describe('reported usage comes from upstream usage (/v1/chat/completions)', () => {
  it('stream: the final chunk carries Qwen counts (cumulative, last typing frame wins)', async () => {
    const usage = await runStream(QWEN_USAGE);
    assert.equal(usage.prompt_tokens, 651);
    assert.equal(usage.completion_tokens, 2);
    assert.equal(usage.total_tokens, 653);
  });

  it('stream: no upstream usage → estimated, never 0', async () => {
    assertEstimated(await runStream(NO_USAGE));
  });

  it('stream: partial upstream usage → only the missing counter is estimated', async () => {
    const usage = await runStream(PARTIAL_USAGE);
    assert.equal(usage.prompt_tokens, 651);
    assert.ok(usage.completion_tokens > 0, `completion_tokens=${usage.completion_tokens}`);
  });

  it('non-stream: body.usage carries Qwen counts', async () => {
    const usage = await runNonStream(QWEN_USAGE);
    assert.equal(usage.prompt_tokens, 651);
    assert.equal(usage.completion_tokens, 2);
    assert.equal(usage.total_tokens, 653);
  });

  it('stream: all-zero upstream usage is treated as absent → estimated', async () => {
    assertEstimated(await runStream(ZERO_USAGE));
  });

  it('non-stream: no upstream usage → estimated, never 0', async () => {
    assertEstimated(await runNonStream(NO_USAGE));
  });

  it('non-stream: all-zero upstream usage is treated as absent → estimated', async () => {
    assertEstimated(await runNonStream(ZERO_USAGE));
  });

  const agentOptions = () => ({
    has_tools: true,
    tool_choice: 'auto',
    allowed_tool_names: ['Read'],
    tool_schemas: { Read: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] } },
    agent_turn_max_attempts: 1,
    upstream_request_body: REQUEST_BODY,
    sendChatRequest: async () => ({ status: false })
  });

  it('agent runtime (has_tools): the accepted attempt reports Qwen counts, not the estimate', async () => {
    const frames = [frame('<agent_final>Hello</agent_final>', { input_tokens: 651, output_tokens: 9, total_tokens: 660 }), STOP];
    const usage = await runNonStream(frames, agentOptions());
    assert.equal(usage.prompt_tokens, 651);
    assert.equal(usage.completion_tokens, 9);
  });

  it('agent runtime (has_tools): no upstream usage → estimated, never 0', async () => {
    const frames = [frame('<agent_final>Hello there</agent_final>'), STOP];
    assertEstimated(await runNonStream(frames, agentOptions()));
  });

  it('agent runtime (has_tools): several attempts → reported usage is the accepted attempt\'s, not the first nor the sum', async () => {
    // Attempt 1: <agent_final> vacío → el gate reintenta ('empty'). Attempt 2: respuesta válida
    // con OTROS números. Lo reportado debe ser lo del attempt aceptado (700/5), no 651/9 ni 1351/14.
    const first = [frame('<agent_final></agent_final>', { input_tokens: 651, output_tokens: 9, total_tokens: 660 }), STOP];
    const second = [frame('<agent_final>Hello</agent_final>', { input_tokens: 700, output_tokens: 5, total_tokens: 705 }), STOP];
    let extraAttempts = 0;
    const sendChatRequest = async () => { extraAttempts += 1; return { status: true, response: upstreamOf(second) }; };
    const usage = await runNonStream(first, { ...agentOptions(), agent_turn_max_attempts: 2, sendChatRequest });
    assert.equal(extraAttempts, 1, 'hubo exactamente un segundo attempt');
    assert.equal(usage.prompt_tokens, 700);
    assert.equal(usage.completion_tokens, 5);
    assert.equal(usage.total_tokens, 705);
  });
});
