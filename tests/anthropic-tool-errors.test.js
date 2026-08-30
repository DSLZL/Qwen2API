// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
// The config clamps this to [2, 6]; 3 keeps the exhaustion test short.
process.env.AGENT_TURN_MAX_ATTEMPTS = '3';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

const { handleAnthropicStream } = require('../src/controllers/anthropic.js');

const createMockResponse = () => ({
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

/** One upstream turn: the given answer chunks, then a clean stop. */
const turn = (...contents) => () => Readable.from([
  ...contents.map(content => `data: ${JSON.stringify({
    choices: [{ delta: { phase: 'answer', content }, finish_reason: null }]
  })}\n\n`),
  'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
]);

/**
 * sendRequest is a ctx destructuring default, so each retry's upstream can be
 * scripted. Records the bodies it was handed, to assert on the retry hints.
 */
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

const runStream = (upstream, sendRequest, overrides = {}) => {
  const res = createMockResponse();
  return handleAnthropicStream(
    res,
    {
      message_id: 'msg_tools',
      model: 'qwen-test',
      hasTools: true,
      toolChoice: 'auto',
      allowedToolNames: ['read_file'],
      requestBody: { messages: [] },
      sendRequest,
      ...overrides
    },
    upstream()
  ).then(() => res);
};

const toolUseNames = (output) =>
  [...output.matchAll(/"type":"tool_use","id":"[^"]*","name":"([^"]*)"/g)].map(m => m[1]);

const BAD_NAME = '<tool_call>{"name":"Bash","arguments":{}}</tool_call>';
const GOOD_CALL = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>';

describe('invented tool names are recoverable, not fatal', () => {
  it('retries and succeeds on the second attempt', async () => {
    const sender = scriptedSender(turn(GOOD_CALL));
    const res = await runStream(turn(BAD_NAME), sender);

    assert.deepEqual(toolUseNames(res.output), ['read_file']);
    assert.doesNotMatch(res.output, /invalid_tool_call_error/);
    assert.equal(sender.calls.length, 1, 'exactly one retry should have been sent');
  });

  it('tells the model the real tool names, which is what makes it recoverable', async () => {
    const sender = scriptedSender(turn(GOOD_CALL));
    await runStream(turn(BAD_NAME), sender);

    const hint = JSON.stringify(sender.calls[0]);
    assert.match(hint, /Bash/, 'the retry hint should name the tool that does not exist');
    assert.match(hint, /read_file/, 'the retry hint should list the allowed tool names');
  });

  it('gives up after the configured attempts and says what went wrong', async () => {
    // Every attempt keeps inventing the same name.
    const sender = scriptedSender(turn(BAD_NAME), turn(BAD_NAME), turn(BAD_NAME));
    const res = await runStream(turn(BAD_NAME), sender);

    assert.match(res.output, /invalid_tool_call_error/);
    // The detail used to be discarded, collapsing three different causes into one
    // opaque sentence with no log line at all for this one.
    assert.match(res.output, /unknown_tool/);
    assert.match(res.output, /Bash/);
    assert.match(res.output, /连续 3 次/, 'should distinguish an exhausted turn from a single failure');
    assert.equal(sender.calls.length, 2, 'AGENT_TURN_MAX_ATTEMPTS=3 means 1 initial + 2 retries');
  });
});

describe('per-attempt parser isolation', () => {
  it('recovers a truncated call on the next attempt', async () => {
    const sender = scriptedSender(turn(GOOD_CALL));
    const res = await runStream(turn('<tool_call>{"name":'), sender);

    assert.deepEqual(toolUseNames(res.output), ['read_file']);
    assert.doesNotMatch(res.output, /invalid_tool_call_error/);
  });

  it('does not let a failed attempt poison a good plain-text answer', async () => {
    // The parser used to be built once for the whole request, so its `errors`
    // array outlived the attempt that filled it. Attempt 2 could then answer
    // perfectly in plain text and still be rejected, because hasParseError()
    // was still reporting attempt 1's failure.
    const sender = scriptedSender(turn('All good, nothing to run.'));
    const res = await runStream(turn('<tool_call>{"name":'), sender);

    assert.doesNotMatch(res.output, /invalid_tool_call_error/);
    assert.match(res.output, /All good, nothing to run\./);
    assert.equal(sender.calls.length, 1, 'one retry was enough; the answer was valid');
  });
});

describe('retrying never duplicates what the client already received', () => {
  it('allows exactly one compensation retry once visible text has been streamed', async () => {
    // This controller streams text as it arrives, so a retry after text is on
    // the wire concatenates two outputs. Prose that only describes an action is
    // the one pre-existing case allowed to retry, and only once.
    const prose = 'I will run the build now.';
    const sender = scriptedSender(turn(prose), turn(prose));
    const res = await runStream(turn(prose), sender);

    assert.equal(sender.calls.length, 1, 'must not keep retrying after streaming text');
    assert.doesNotMatch(res.output, /invalid_tool_call_error/);
  });

  it('still emits no tool_use when the model never calls one', async () => {
    const sender = scriptedSender(turn('I will run the build now.'));
    const res = await runStream(turn('I will run the build now.'), sender);
    assert.deepEqual(toolUseNames(res.output), []);
  });
});

// El fallo reportado: el modelo repite su propio prompt de herramientas. El eco trae
// tags <tool_call> literales, el parser los consume, y antes de este cambio el texto
// tragado desaparecía (frase cortada a la mitad).
const ECHOED_PROMPT = 'Rules: your visible response MUST be a `<tool_call>` block. Call the tool instead.';

/** Reconstruye lo que el cliente ve como respuesta, sin escapes de SSE. */
const visibleTextOf = (output) =>
  [...output.matchAll(/"delta":\{"type":"text_delta","text":("(?:[^"\\]|\\.)*")\}/g)]
    .map(m => JSON.parse(m[1]))
    .join('');

describe('an echoed tool protocol is delivered whole, once', () => {
  it('delivers the sentence character-for-character, backticks included', async () => {
    const res = await runStream(turn(ECHOED_PROMPT), scriptedSender());
    assert.equal(visibleTextOf(res.output), ECHOED_PROMPT);
    assert.deepEqual(toolUseNames(res.output), []);
  });

  it('completes the message instead of aborting it with a 502', async () => {
    const res = await runStream(turn(ECHOED_PROMPT), scriptedSender());
    assert.doesNotMatch(res.output, /invalid_tool_call_error/);
    assert.match(res.output, /"type":"message_stop"/);
  });

  it('does not retry once real prose is on the wire, so nothing is doubled', async () => {
    const sender = scriptedSender(turn(ECHOED_PROMPT), turn(ECHOED_PROMPT));
    const res = await runStream(turn(ECHOED_PROMPT), sender);
    assert.equal(sender.calls.length, 0, 'tool_error must not retry after visible text');
    assert.equal(visibleTextOf(res.output), ECHOED_PROMPT, 'la salida llegó duplicada');
  });

  it('emits the salvaged payload at most once, however many attempts ran', async () => {
    // Tres intentos, todos con un nombre inventado y sin prosa. El rescate de cada
    // intento se descarta al reintentar; el turno cierra con un error explícito en vez
    // de entregar XML crudo como si fuera la respuesta.
    const sender = scriptedSender(turn(BAD_NAME), turn(BAD_NAME), turn(BAD_NAME));
    const res = await runStream(turn(BAD_NAME), sender);
    assert.equal(visibleTextOf(res.output), '', 'el payload rescatado se filtró como respuesta');
    assert.match(res.output, /invalid_tool_call_error/);
    assert.match(res.output, /连续 3 次/);
  });

  it('never writes an error event after a content block', async () => {
    const res = await runStream(turn(ECHOED_PROMPT), scriptedSender());
    const firstContent = res.output.indexOf('content_block_start');
    const errorAt = res.output.indexOf('"type":"error"');
    assert.ok(firstContent !== -1, 'no se emitió contenido');
    assert.equal(errorAt, -1, 'un error event invalida todo el contenido ya enviado');
  });

  it('still retries a bad tool call when no prose reached the client', async () => {
    // La guarda es específica de "ya salió prosa": el payload rescatado no cuenta.
    const sender = scriptedSender(turn(GOOD_CALL));
    const res = await runStream(turn(BAD_NAME), sender);
    assert.equal(sender.calls.length, 1);
    assert.deepEqual(toolUseNames(res.output), ['read_file']);
  });
});
