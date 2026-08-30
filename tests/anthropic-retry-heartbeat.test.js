// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
process.env.ANTHROPIC_PING_INTERVAL_MS = '1000';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

const { handleAnthropicStream } = require('../src/controllers/anthropic.js');

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const createMockResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  destroyed: false,
  set(headers) {
    Object.assign(this.headers, headers);
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

// The stream handler takes `sendRequest` as a ctx destructuring default, so the
// retry's upstream call can be replaced without touching the module registry.
//
// The interval is pinned to config's 1s floor at the top of this file, so the
// 2.5s retry below spans it.
const slowSendRequest = (ms) => async () => {
  await delay(ms);
  return { status: false };
};

describe('compensation retry keeps the stream alive', () => {
  it('pings while the retry regenerates, instead of going silent', async () => {
    const res = createMockResponse();

    // tool_choice required + an upstream that answers in prose and calls nothing
    // is exactly `needsRequiredRetry`. Before the fix the retry ran outside any
    // keepalive, so the client saw nothing for its whole duration -- measured at
    // 29.7s against the canary, over the 20s stall threshold.
    await handleAnthropicStream(
      res,
      {
        message_id: 'msg_retry',
        model: 'qwen-test',
        hasTools: true,
        toolChoice: 'required',
        allowedToolNames: ['read_file'],
        requestBody: { messages: [] },
        sendRequest: slowSendRequest(2500)
      },
      Readable.from([
        'data: {"choices":[{"delta":{"phase":"answer","content":"No tool needed."},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
      ])
    );

    const pings = res.output.match(/event: ping\n/g) || [];
    assert.ok(pings.length >= 1, `retry window emitted no ping (output: ${res.output.slice(0, 200)})`);
    assert.doesNotMatch(res.output, /^:/m, 'SSE comments are dropped by SDKs; must be protocol events');
  });

  it('sends no ping when the retry returns promptly', async () => {
    const res = createMockResponse();
    await handleAnthropicStream(
      res,
      {
        message_id: 'msg_fast',
        model: 'qwen-test',
        hasTools: true,
        toolChoice: 'required',
        allowedToolNames: ['read_file'],
        requestBody: { messages: [] },
        sendRequest: slowSendRequest(0)
      },
      Readable.from([
        'data: {"choices":[{"delta":{"phase":"answer","content":"No tool needed."},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
      ])
    );
    assert.equal((res.output.match(/event: ping\n/g) || []).length, 0);
  });
});
