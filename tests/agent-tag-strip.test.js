const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const { Readable } = require('node:stream');

const { createAgentTagStripper, stripAgentTags } = require('../src/utils/agent-turn.js');
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

const frame = (content) =>
  `data: ${JSON.stringify({ choices: [{ delta: { phase: 'answer', content }, finish_reason: null }] })}\n\n`;

const DONE = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

// Whatever text the client actually receives, reassembled from the SSE frames.
const emittedText = (output) =>
  [...output.matchAll(/"type":"text_delta","text":("(?:[^"\\]|\\.)*")/g)]
    .map(m => JSON.parse(m[1]))
    .join('');

// A retry must never reach the network from a test: the handler takes
// sendRequest as a ctx destructuring default.
const noRetry = async () => ({ status: false });

// Feed a string one character at a time: the worst chunking a real stream can
// produce, so every tag is split across chunk boundaries.
const pushByChar = (text) => {
  const stripper = createAgentTagStripper();
  let out = '';
  for (const ch of text) out += stripper.push(ch);
  return out + stripper.flush();
};

describe('agent control tag stripping', () => {
  it('removes the wrapper and keeps the body', () => {
    assert.equal(stripAgentTags('<agent_final>done</agent_final>'), 'done');
    assert.equal(stripAgentTags('<agent_blocked>need a key</agent_blocked>'), 'need a key');
  });

  it('keeps prose that precedes the wrapper instead of failing the turn', () => {
    // The OpenAI path regenerates this shape as invalid_control. This path has
    // no turn gate, so the only sane outcome is to keep both halves.
    assert.equal(
      stripAgentTags('Server back on 3000.\n<agent_final>Restarted.</agent_final>'),
      'Server back on 3000.\nRestarted.'
    );
  });

  it('strips tags split across chunk boundaries', () => {
    assert.equal(pushByChar('<agent_final>ok</agent_final>'), 'ok');
    assert.equal(pushByChar('a<agent_final>b</agent_final>c'), 'abc');
  });

  it('leaves ordinary angle brackets and markup alone', () => {
    assert.equal(stripAgentTags('use <div> and 1 < 2'), 'use <div> and 1 < 2');
    assert.equal(pushByChar('use <div> and 1 < 2'), 'use <div> and 1 < 2');
    assert.equal(stripAgentTags('<agent_finalized>'), '<agent_finalized>');
  });

  it('flush returns a buffered fragment that never became a tag', () => {
    const stripper = createAgentTagStripper();
    assert.equal(stripper.push('text <age'), 'text ');
    assert.equal(stripper.flush(), '<age');
  });

  it('never buffers more than one tag length', () => {
    const stripper = createAgentTagStripper();
    const out = stripper.push('x'.repeat(5000));
    assert.equal(out.length, 5000);
    assert.equal(stripper.flush(), '');
  });
});

describe('the Anthropic stream strips the wrapper end to end', () => {
  for (const hasTools of [false, true]) {
    it(`removes agent_final from what the client sees (hasTools=${hasTools})`, async () => {
      const res = createMockResponse();
      await handleAnthropicStream(
        res,
        {
          message_id: 'msg_strip',
          model: 'qwen-test',
          hasTools,
          toolChoice: 'auto',
          allowedToolNames: ['read_file'],
          requestBody: { messages: [] },
          sendRequest: noRetry
        },
        // The open tag is split across two frames, which is what actually
        // happens on the wire and what a naive replace() would miss.
        Readable.from([frame('<agent_'), frame('final>All good.</agent_final>'), DONE])
      );

      assert.equal(emittedText(res.output), 'All good.');
      assert.doesNotMatch(res.output, /agent_final/);
    });
  }

  it('does not swallow a trailing fragment that never became a tag', async () => {
    const res = createMockResponse();
    await handleAnthropicStream(
      res,
      {
        message_id: 'msg_tail',
        model: 'qwen-test',
        hasTools: false,
        toolChoice: 'auto',
        allowedToolNames: [],
        requestBody: { messages: [] },
        sendRequest: noRetry
      },
      Readable.from([frame('cost < 5 and <ag'), DONE])
    );
    assert.equal(emittedText(res.output), 'cost < 5 and <ag');
  });
});
