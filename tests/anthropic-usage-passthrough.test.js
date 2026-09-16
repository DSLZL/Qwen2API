// Reported usage = upstream usage. Qwen manda `usage` con nombres DashScope
// (input_tokens / output_tokens / total_tokens), acumulado, en cada frame `typing`;
// el frame final no lo trae. El proxy leia prompt_tokens / completion_tokens (OpenAI)
// y por eso TODAS las respuestas caian al estimado local (input_tokens: 8 para "hi",
// cuando Qwen contaba 651).
//
// Seam: los handlers del controller alimentados con frames upstream sinteticos.
// Harness copiado de tests/anthropic-cap-drain.test.js —— cada archivo de test corre
// en su propio proceso.

const test = require('node:test');
const { describe, it } = test;
const assert = require('node:assert/strict');

// El harness requiere account.js, que en modo file haria login real con data/data.json.
process.env.API_KEY = 'usage-test-key';
process.env.DATA_SAVE_MODE = 'none';
process.env.ACCOUNTS = '';
process.env.ENABLE_CLI = 'false';

// Sin red en tests: los parches de require-cache van ANTES de requerir el controller
// (anthropic.js captura sendChatRequest por destructuring en su primer require).
const modelsMap = require('../src/models/models-map.js');
modelsMap.getLatestModels = async () => { throw new Error('offline test: no model fetch'); };
const requestModule = require('../src/utils/request.js');
let upstreamFactory = null;
requestModule.sendChatRequest = async () => (upstreamFactory
  ? { status: true, response: upstreamFactory(), currentAccount: null }
  : { status: false });

const { handleAnthropicStream, handleAnthropicMessages } = require('../src/controllers/anthropic.js');

test.after(() => {
  require('../src/utils/account.js').destroy();
});

const createMockStreamResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  set(headers) { Object.assign(this.headers, headers); return this; },
  status() { return this; },
  write(chunk) { this.output += String(chunk); return true; },
  end(chunk = '') { this.output += String(chunk); this.writableEnded = true; }
});

const createMockJsonResponse = () => ({
  statusCode: 200,
  body: null,
  headers: {},
  set(headers) { Object.assign(this.headers, headers); return this; },
  status(code) { this.statusCode = code; return this; },
  json(payload) { this.body = payload; return this; }
});

/** Frame de respuesta como lo manda Qwen: `usage` al nivel del frame, junto a `choices`. */
const frame = (content, usage) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }],
  ...(usage === undefined ? {} : { usage })
})}\n\n`;

const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

const upstream = (frames) => {
  async function* gen() {
    for (const f of frames) yield f;
  }
  return gen();
};

const eventsOf = (output) => output
  .split('\n\n')
  .filter(Boolean)
  .map(chunk => chunk.split('\n').find(line => line.startsWith('data: ')))
  .filter(Boolean)
  .map(line => JSON.parse(line.slice(6)));

/** `secondAttemptFrames`: lo que devuelve el sendRequest del segundo attempt; null = un solo attempt. */
const runStream = async (frames, secondAttemptFrames = null) => {
  const res = createMockStreamResponse();
  await handleAnthropicStream(res, {
    message_id: 'msg_usage',
    model: 'qwen-test',
    hasTools: false,
    toolChoice: null,
    allowedToolNames: [],
    toolSchemas: {},
    requestBody: { messages: [{ role: 'user', content: 'hi' }] },
    sendRequest: async () => (secondAttemptFrames ? { status: true, response: upstream(secondAttemptFrames) } : { status: false })
  }, upstream(frames));
  return eventsOf(res.output);
};

const deltaUsage = (events) => events.find(e => e.type === 'message_delta').usage;

describe('reported usage comes from upstream usage (Anthropic /v1/messages)', () => {
  it('stream: message_delta carries Qwen input_tokens/output_tokens; cumulative, last typing frame wins', async () => {
    const events = await runStream([
      frame('Hel', { input_tokens: 651, output_tokens: 1, total_tokens: 652 }),
      frame('lo', { input_tokens: 651, output_tokens: 2, total_tokens: 653 }),
      STOP
    ]);
    const usage = deltaUsage(events);
    assert.equal(usage.input_tokens, 651);
    assert.equal(usage.output_tokens, 2);
  });

  it('stream: cache fields are 0 (not null) in message_start and message_delta', async () => {
    const events = await runStream([frame('Hi', { input_tokens: 651, output_tokens: 1 }), STOP]);
    for (const event of [events.find(e => e.type === 'message_start').message, events.find(e => e.type === 'message_delta')]) {
      assert.equal(event.usage.cache_creation_input_tokens, 0);
      assert.equal(event.usage.cache_read_input_tokens, 0);
    }
  });

  it('stream: no upstream usage at all → both counters estimated locally (never 0)', async () => {
    const usage = deltaUsage(await runStream([frame('Hello there'), STOP]));
    assert.ok(usage.input_tokens > 0, `input_tokens=${usage.input_tokens}`);
    assert.ok(usage.output_tokens > 0, `output_tokens=${usage.output_tokens}`);
  });

  it('stream: partial upstream usage → only the missing counter is estimated', async () => {
    const usage = deltaUsage(await runStream([frame('Hello there', { input_tokens: 651 }), STOP]));
    assert.equal(usage.input_tokens, 651);
    assert.ok(usage.output_tokens > 0, `output_tokens=${usage.output_tokens}`);
  });

  it('stream: all-zero upstream usage is treated as absent → estimated', async () => {
    const usage = deltaUsage(await runStream([frame('Hello there', { input_tokens: 0, output_tokens: 0, total_tokens: 0 }), STOP]));
    assert.ok(usage.input_tokens > 0, `input_tokens=${usage.input_tokens}`);
    assert.ok(usage.output_tokens > 0, `output_tokens=${usage.output_tokens}`);
  });

  it('stream: several attempts → the last attempt\'s usage; a later attempt that reports nothing does NOT inherit the first attempt\'s counters', async () => {
    // Attempt 1: sin texto visible → el gate reintenta ('empty'); traia 651/9 del upstream.
    // Attempt 2: texto valido y SOLO output_tokens. Reportado: output 5 (attempt 2) e input
    // estimado —— nunca los 651 del attempt 1.
    const usage = deltaUsage(await runStream(
      [frame('', { input_tokens: 651, output_tokens: 9, total_tokens: 660 }), STOP],
      [frame('Hello there', { output_tokens: 5 }), STOP]
    ));
    assert.equal(usage.output_tokens, 5);
    assert.ok(usage.input_tokens > 0 && usage.input_tokens !== 651, `input_tokens=${usage.input_tokens}`);
  });

  /** Cada argumento es un attempt: el primero es el upstream inicial, los demas, attempts posteriores. */
  const runNonStream = async (...attempts) => {
    const queue = [...attempts];
    upstreamFactory = () => {
      const frames = queue.shift();
      assert.ok(frames, 'upstream pedido mas veces que attempts preparados');
      return upstream(frames);
    };
    try {
      const res = createMockJsonResponse();
      await handleAnthropicMessages({
        body: { model: 'qwen3-max', max_tokens: 64, stream: false, messages: [{ role: 'user', content: 'hi' }] }
      }, res);
      assert.equal(res.statusCode, 200, JSON.stringify(res.body));
      return res.body.usage;
    } finally {
      upstreamFactory = null;
    }
  };

  it('non-stream twin via handleAnthropicMessages: body.usage carries Qwen counts, cache fields 0', async () => {
    const usage = await runNonStream([
      frame('Hel', { input_tokens: 651, output_tokens: 1, total_tokens: 652 }),
      frame('lo', { input_tokens: 651, output_tokens: 2, total_tokens: 653 }),
      STOP
    ]);
    assert.equal(usage.input_tokens, 651);
    assert.equal(usage.output_tokens, 2);
    assert.equal(usage.cache_creation_input_tokens, 0);
    assert.equal(usage.cache_read_input_tokens, 0);
  });

  it('non-stream: no upstream usage → estimated, never 0', async () => {
    const usage = await runNonStream([frame('Hello there'), STOP]);
    assert.ok(usage.input_tokens > 0, `input_tokens=${usage.input_tokens}`);
    assert.ok(usage.output_tokens > 0, `output_tokens=${usage.output_tokens}`);
  });

  it('non-stream: several attempts → the last attempt\'s usage; a later attempt that reports nothing does NOT inherit the first attempt\'s counters', async () => {
    const usage = await runNonStream(
      [frame('', { input_tokens: 651, output_tokens: 9, total_tokens: 660 }), STOP],
      [frame('Hello there', { output_tokens: 5 }), STOP]
    );
    assert.equal(usage.output_tokens, 5);
    assert.ok(usage.input_tokens > 0 && usage.input_tokens !== 651, `input_tokens=${usage.input_tokens}`);
  });
});
