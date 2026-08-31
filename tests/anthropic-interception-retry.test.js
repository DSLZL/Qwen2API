// La defensa contra la interceptacion de plataforma: el modelo regresa por habito RL a la
// forma nativa, la plataforma se la come server-side y le inyecta "Tool <name> does not
// exists"; el turno llega como prosa valida y ningun retry disparaba. La unica evidencia
// que el gateway ve en vivo son los drops role:function del normalizador (Defect A) —
// aqui se pina que esa evidencia dispara exactamente UN retry con el hint canonico.
//
// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
// The config clamps this to [2, 6]; 3 keeps the cap-mutation tests short.
process.env.AGENT_TURN_MAX_ATTEMPTS = '3';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

const { handleAnthropicStream, handleAnthropicNonStream } = require('../src/controllers/anthropic.js');

const createMockStreamResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  destroyed: false,
  set(headers) {
    Object.assign(this.headers, headers);
    return this;
  },
  status() {
    return this;
  },
  write(chunk) {
    this.output += String(chunk);
    return true;
  },
  end(chunk = '') {
    this.output += String(chunk);
    this.writableEnded = true;
  }
});

const createMockJsonResponse = () => ({
  statusCode: 200,
  body: null,
  headers: {},
  set(headers) {
    Object.assign(this.headers, headers);
    return this;
  },
  status(code) {
    this.statusCode = code;
    return this;
  },
  json(payload) {
    this.body = payload;
    return this;
  }
});

const answerFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }]
})}\n\n`;

// Forma exacta del incidente 2026-08-31 08:33: la plataforma consumio el tool call nativo
// del modelo y reinyecto su lookup de registry como frame role:function. Defect A lo
// dropea del stream; interceptedToolNames es la huella que queda.
const interceptionFrame = (name) => `data: ${JSON.stringify({
  choices: [{
    delta: { role: 'function', phase: 'answer', name, content: `Tool ${name} does not exists` },
    finish_reason: null
  }]
})}\n\n`;

const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

/** One upstream turn from raw SSE frames, then a clean stop. */
const turnOf = (...frames) => () => Readable.from([...frames, STOP]);

// La narracion NO debe matchear looksLikeUnexecutedToolAction ("I'll run..."): si lo
// hiciera, missing_tool taparia al branch intercepted y la mutacion que lo borra
// pasaria desapercibida.
const NARRATION = 'The Bash tool seems unavailable in this environment, so the task cannot continue.';
const BRACKET_CALL = '[TOOL CALL]{"name":"read_file","arguments":{"path":"a.txt"}}[END TOOL CALL]';

const scriptedSender = (...turns) => {
  const queue = [...turns];
  const fn = async (body) => {
    fn.calls.push(body);
    const next = queue.shift();
    return next ? { status: true, response: next() } : { status: false };
  };
  fn.calls = [];
  return fn;
};

const baseCtx = (sendRequest, overrides) => ({
  message_id: 'msg_intercept',
  model: 'qwen-test',
  hasTools: true,
  toolChoice: 'auto',
  allowedToolNames: ['read_file'],
  requestBody: { messages: [] },
  sendRequest,
  ...overrides
});

const runStream = (upstream, sendRequest, overrides = {}) => {
  const res = createMockStreamResponse();
  return handleAnthropicStream(res, baseCtx(sendRequest, overrides), upstream()).then(() => res);
};

const runNonStream = (upstream, sendRequest, overrides = {}) => {
  const res = createMockJsonResponse();
  return handleAnthropicNonStream(res, baseCtx(sendRequest, overrides), upstream()).then(() => res);
};

const toolUseNames = (output) =>
  [...output.matchAll(/"type":"tool_use","id":"[^"]*","name":"([^"]*)"/g)].map(m => m[1]);

describe('interception-aware retry (stream)', () => {
  it('recovers the turn: one retry with the canonical hint, tool_use after the narration', async () => {
    const sender = scriptedSender(turnOf(answerFrame(BRACKET_CALL)));
    const res = await runStream(turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)), sender);

    assert.equal(sender.calls.length, 1, 'the interception must trigger exactly one retry');
    assert.deepEqual(toolUseNames(res.output), ['read_file']);

    const retryBody = JSON.stringify(sender.calls[0]);
    assert.match(retryBody, /did not reach the client/, 'hint must say the call was lost, nothing more');
    assert.ok(retryBody.includes('[TOOL CALL]'), 'hint must teach the canonical bracket marker');
    assert.doesNotMatch(retryBody, /<tool_call/i, 'hint must never re-seed the native form');

    // tool_use despues de la narracion ya streameada: mejor que una sesion muerta.
    const narrationAt = res.output.indexOf('unavailable');
    const toolUseAt = res.output.indexOf('"type":"tool_use"');
    assert.ok(narrationAt !== -1 && toolUseAt > narrationAt, 'tool_use must follow the streamed narration');
    assert.match(res.output, /"stop_reason":"tool_use"/);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('two consecutive interceptions: exactly one retry, closes without error events', async () => {
    const sender = scriptedSender(
      turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)),
      turnOf(interceptionFrame('Bash'), answerFrame(NARRATION))
    );
    const res = await runStream(turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)), sender);

    assert.equal(sender.calls.length, 1, 'second interception must deliver as-is, no loop');
    assert.doesNotMatch(res.output, /"type":"error"/);
    assert.match(res.output, /"type":"message_stop"/);
  });

  it('the cap is the interception cap itself, not the after-prose guard', async () => {
    // Sin narracion no hay texto visible, asi que la concesion "una compensacion tras
    // prosa" nunca se activa: solo el tope dedicado puede parar el loop. Con
    // AGENT_TURN_MAX_ATTEMPTS=3, quitar el tope daria 2 retries, no 1.
    const sender = scriptedSender(
      turnOf(interceptionFrame('Bash')),
      turnOf(interceptionFrame('Bash')),
      turnOf(interceptionFrame('Bash'))
    );
    await runStream(turnOf(interceptionFrame('Bash')), sender);

    assert.equal(sender.calls.length, 1, 'exactly ONE interception retry per request');
  });

  it('benign speculative drop: drops alongside an accepted call never retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame(BRACKET_CALL)));
    const res = await runStream(turnOf(interceptionFrame('Bash'), answerFrame(BRACKET_CALL)), sender);

    assert.equal(sender.calls.length, 0, 'an accepted bracket call means the turn is fine');
    assert.deepEqual(toolUseNames(res.output), ['read_file']);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('no tools in play: drops on a prose-only request never retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame('unused')));
    const res = await runStream(
      turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)),
      sender,
      { hasTools: false, allowedToolNames: [] }
    );

    assert.equal(sender.calls.length, 0);
    assert.match(res.output, /"type":"message_stop"/);
  });
});

describe('interception-aware retry (non-stream)', () => {
  it('clean retry: nothing was sent yet, tool_use lands in the response', async () => {
    const sender = scriptedSender(turnOf(answerFrame(BRACKET_CALL)));
    const res = await runNonStream(turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)), sender);

    assert.equal(sender.calls.length, 1);
    assert.match(JSON.stringify(sender.calls[0]), /did not reach the client/);
    assert.equal(res.statusCode, 200);
    const toolBlocks = (res.body?.content || []).filter(block => block.type === 'tool_use');
    assert.deepEqual(toolBlocks.map(block => block.name), ['read_file']);
    assert.equal(res.body.stop_reason, 'tool_use');
  });

  it('same one-shot cap: a second interception delivers the prose, not a 502', async () => {
    // Este loop no tiene guard de texto-ya-enviado (nada salio al cliente), asi que sin
    // el tope dedicado reintentaria hasta maxAttempts: 2 retries en vez de 1.
    const sender = scriptedSender(
      turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)),
      turnOf(interceptionFrame('Bash'), answerFrame(NARRATION))
    );
    const res = await runNonStream(turnOf(interceptionFrame('Bash'), answerFrame(NARRATION)), sender);

    assert.equal(sender.calls.length, 1, 'exactly ONE interception retry per request');
    assert.equal(res.statusCode, 200, 'deliver as-is, not an error');
    const text = (res.body?.content || []).filter(block => block.type === 'text').map(block => block.text).join('');
    assert.match(text, /unavailable/, 'the narration is the answer the client gets');
    assert.equal(res.body.stop_reason, 'end_turn');
  });
});
