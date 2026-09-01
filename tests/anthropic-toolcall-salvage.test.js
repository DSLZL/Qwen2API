// salvage-3: recuperacion de leaks prose-adjacent (spec-qwen2api-toolcall-salvage-3).
// Tres capas fail-closed: (1) reparacion de comillas + nameHint del tail del trigger,
// con triple compuerta estricta (JSON.parse estricto + allowlist no vacia + schema);
// (2) loop B: tool_error tras prosa consume el cupo retriedAfterVisibleText con un
// retry de texto suprimido; (3) el residuo de protocolo se pela SOLO en la entrega,
// derivado de los spans condenados que registra el parser (jamas un segundo escaneo).
//
// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
process.env.AGENT_TURN_MAX_ATTEMPTS = '3';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

const {
  parseToolCallsFromText,
  createToolCallStreamParser,
  stripToolCallResidue,
  repairLooseToolPayload
} = require('../src/utils/tool-prompt.js');
const { handleAnthropicStream, handleAnthropicNonStream } = require('../src/controllers/anthropic.js');
const { logger } = require('../src/utils/logger.js');

const ALLOWED = ['Bash', 'read_file'];
const SCHEMAS = {
  Bash: {
    type: 'object',
    properties: {
      command: { type: 'string' },
      description: { type: 'string' },
      timeout: { type: 'number' }
    }
  },
  read_file: { type: 'object', properties: { path: { type: 'string' } } }
};

// Reconstruccion exacta del incidente 3 (2026-08-31 16:39, docker logs): el nombre
// queda FUERA del JSON, la clave va sin comillas y el valor perdio su comilla de
// apertura pero conserva la de cierre — la paridad de comillas muere y
// extractBalancedObject jamas cierra → truncated_tool_call.
const INCIDENT3_CMD = 'find /Users/pedro/Documents/git/Prueba/payroll/_bmad-output/planning-artifacts/architecture-Español-2026-09-01 -type f 2>/dev/null';
const INCIDENT3_SPAN = `[TOOL_CALL]Bash{command:${INCIDENT3_CMD}", "description": "List files in architecture-Español directory"}}\n[END TOOL CALL]`;
const INCIDENT3_PROSE = 'Voy a listar los archivos del directorio.';
const INCIDENT3 = `${INCIDENT3_PROSE}\n${INCIDENT3_SPAN}\n`;

const GOOD_CALL = '[TOOL CALL]{"name":"read_file","arguments":{"path":"a.txt"}}[END TOOL CALL]';
const GARBAGE_CALL = '[TOOL CALL]{"name":"garbage","arguments":{}}[END TOOL CALL]';

const streamAll = (text, options, chunk = 7) => {
  const parser = createToolCallStreamParser(options);
  let visible = '';
  let recovered = '';
  const calls = [];
  for (let i = 0; i < text.length; i += chunk) {
    const out = parser.push(text.slice(i, i + chunk));
    visible += out.textDelta;
    recovered += out.recoveredText;
    calls.push(...out.completedCalls);
  }
  const tail = parser.flush();
  visible += tail.textDelta;
  recovered += tail.recoveredText;
  calls.push(...tail.completedCalls);
  return { parser, visible, recovered, calls };
};

describe('incident-3 salvage: name outside the JSON, broken quote parity', () => {
  it('whole-text: exact command recovered, no errors, no residue in cleanedText', () => {
    const result = parseToolCallsFromText(INCIDENT3, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });

    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    const args = JSON.parse(result.toolCalls[0].function.arguments);
    assert.equal(args.command, INCIDENT3_CMD, 'the command must round-trip byte-for-byte');
    assert.equal(args.description, 'List files in architecture-Español directory');
    assert.equal(result.errors.length, 0, 'salvage must run before the truncated condemnation');
    assert.equal(result.cleanedText, INCIDENT3_PROSE);
    assert.doesNotMatch(result.cleanedText, /TOOL.?CALL/i);
    assert.equal(result.residueSpans.length, 0, 'a salvaged span is consumed, not condemned');
  });

  it('streaming (7-char chunks): same call at flush, recoveredText stays empty', () => {
    const { parser, visible, recovered, calls } = streamAll(INCIDENT3, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });

    assert.equal(calls.length, 1);
    assert.equal(calls[0].function.name, 'Bash');
    assert.equal(JSON.parse(calls[0].function.arguments).command, INCIDENT3_CMD);
    assert.equal(parser.getErrors().length, 0);
    assert.equal(recovered, '', 'nothing to recover — the span became a call');
    assert.match(visible, /Voy a listar los archivos/);
    assert.doesNotMatch(visible, /TOOL.?CALL/i, 'zero protocol bytes may reach the visible channel');
  });

  it('whole-text without prose still salvages (parity with the streaming path)', () => {
    const result = parseToolCallsFromText(`${INCIDENT3_SPAN}\n`, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.errors.length, 0);
    assert.equal(result.cleanedText, '');
  });
});

describe('balanced unquoted payload: nameHint + quote repair on the invalid_json path', () => {
  const text = '[TOOL_CALL]Bash{command: "ls"}';

  it('whole-text: salvaged into a Bash call', () => {
    const result = parseToolCallsFromText(text, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.equal(JSON.parse(result.toolCalls[0].function.arguments).command, 'ls');
    assert.equal(result.errors.length, 0);
  });

  it('streaming: same result', () => {
    const { calls, parser } = streamAll(text, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].function.name, 'Bash');
    assert.equal(parser.getErrors().length, 0);
  });
});

describe('salvage gates are fail-closed', () => {
  it('empty allowlist: no salvage, no name-prefix extraction — today\'s truncated error', () => {
    const result = parseToolCallsFromText(`${INCIDENT3_SPAN}\n`, { allowedToolNames: [], toolSchemas: SCHEMAS });
    assert.equal(result.toolCalls.length, 0);
    assert.equal(result.errors[0]?.type, 'truncated_tool_call');
  });

  it('allowlist without schemas: no salvage either (the schema gate is half the boundary)', () => {
    const result = parseToolCallsFromText(`${INCIDENT3_SPAN}\n`, { allowedToolNames: ALLOWED });
    assert.equal(result.toolCalls.length, 0);
    assert.equal(result.errors[0]?.type, 'truncated_tool_call');
  });

  it('schema gate: a repaired key outside input_schema.properties rejects the salvage', () => {
    const result = parseToolCallsFromText('[TOOL_CALL]Bash{command: "ls", banana: "y"}', {
      allowedToolNames: ALLOWED,
      toolSchemas: SCHEMAS
    });
    assert.equal(result.toolCalls.length, 0, 'phantom keys must never execute');
    assert.equal(result.errors[0]?.type, 'invalid_json');
  });

  it('nameHint outside the allowlist rejects the salvage', () => {
    const result = parseToolCallsFromText('[TOOL_CALL]NotATool{command: "ls"}', {
      allowedToolNames: ALLOWED,
      toolSchemas: SCHEMAS
    });
    assert.equal(result.toolCalls.length, 0);
    assert.equal(result.errors[0]?.type, 'invalid_json');
  });

  it('unquoted value with internal quotes fails the strict gate and falls through', () => {
    const result = parseToolCallsFromText('[TOOL_CALL]Bash{command:echo "hi", "description": "x"}', {
      allowedToolNames: ALLOWED,
      toolSchemas: SCHEMAS
    });
    assert.equal(result.toolCalls.length, 0, 'nothing mangled may execute');
    assert.equal(result.errors[0]?.type, 'invalid_json');
  });
});

describe('regression pins around the salvage', () => {
  it('code-fence immunity: a fenced incident-3 span stays documentation, no salvage, no spans', () => {
    const fenced = 'Example of the broken form:\n```\n[TOOL_CALL]Bash{command: "ls"}\n[END TOOL CALL]\n```\nDone.';
    const result = parseToolCallsFromText(fenced, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });
    assert.equal(result.toolCalls.length, 0);
    assert.equal(result.errors.length, 0);
    assert.equal(result.residueSpans.length, 0, 'fence text is never a residue span by construction');
    assert.match(result.cleanedText, /\[TOOL_CALL\]Bash\{command: "ls"\}/, 'the example must survive verbatim');
    assert.equal(stripToolCallResidue(result.cleanedText, result.residueSpans), result.cleanedText);
  });

  it('canonical call after prose behaves exactly as today: suppressed, prose delivered', () => {
    const text = `Some prose first.\n${GOOD_CALL}`;
    const result = parseToolCallsFromText(text, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS });
    assert.equal(result.toolCalls.length, 0, 'not-first-content suppression is untouched');
    assert.equal(result.errors.length, 0);
    assert.equal(result.cleanedText, 'Some prose first.');
    assert.ok(result.warnings.some(w => w.reason === 'not the first content of the answer'));
  });

  it('quote-parity-broken giant span: bounded condemnation, salvage does not hang or throw', () => {
    const giant = `[TOOL_CALL]Bash{command:${'x'.repeat(1024 * 1024 + 64)}`;
    const { parser, calls } = streamAll(giant, { allowedToolNames: ALLOWED, toolSchemas: SCHEMAS }, 64 * 1024);
    assert.equal(calls.length, 0);
    const errors = parser.getErrors();
    assert.equal(errors[0]?.type, 'truncated_tool_call');
  });
});

describe('repairLooseToolPayload invariants', () => {
  it('valid JSON is a fixed point: the repair does not run', () => {
    assert.equal(repairLooseToolPayload('{"command": "ls", "n": 1.5, "ok": true}'), null);
  });

  it('incident-3 region repairs into strict JSON with the exact command', () => {
    const region = `{command:${INCIDENT3_CMD}", "description": "d"}`;
    const repaired = repairLooseToolPayload(region);
    assert.notEqual(repaired, null);
    assert.equal(JSON.parse(repaired).command, INCIDENT3_CMD);
  });

  it('true/false/null stay literals only when followed by a delimiter', () => {
    assert.equal(JSON.parse(repairLooseToolPayload('{a:true, b:null}')).a, true);
    // `falsey` empieza con false pero NO va seguido de delimitador → string.
    assert.equal(JSON.parse(repairLooseToolPayload('{a:falsey"}')).a, 'falsey');
  });

  it('backslashes in a loose value round-trip as bytes, never as escapes', () => {
    const repaired = repairLooseToolPayload('{command:dir C:\\tmp"}');
    assert.equal(JSON.parse(repaired).command, 'dir C:\\tmp');
  });

  it('numbers are consumed as whole tokens, not split at the decimal point', () => {
    assert.equal(JSON.parse(repairLooseToolPayload('{timeout: 1.5, command: "ls"}')).timeout, 1.5);
  });
});

describe('stripToolCallResidue derives only from recorded spans', () => {
  it('removes exactly the condemned span, one occurrence per entry', () => {
    const text = `prose before ${GARBAGE_CALL} prose after`;
    assert.equal(stripToolCallResidue(text, [GARBAGE_CALL]), 'prose before  prose after');
  });

  it('without spans it is the identity — no second independent span search', () => {
    const text = `prose ${GARBAGE_CALL}`;
    assert.equal(stripToolCallResidue(text), text);
    assert.equal(stripToolCallResidue(text, []), text);
  });

  it('falls back to the trimmed span when cleanedText trimming ate edge whitespace', () => {
    assert.equal(stripToolCallResidue('abc', ['  abc  ']), '');
  });
});

// ---------------------------------------------------------------------------
// Loop-level fixtures (B stream / C non-stream) on the canned-upstream harness.
// ---------------------------------------------------------------------------

const captureWarns = async (fn) => {
  const saved = logger.warn;
  const lines = [];
  logger.warn = (message) => { lines.push(String(message)); };
  try {
    await fn();
  } finally {
    logger.warn = saved;
  }
  return lines;
};

const createMockStreamResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  destroyed: false,
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

const answerFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }]
})}\n\n`;

const thinkFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'think', content }, finish_reason: null }]
})}\n\n`;

const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

const turnOf = (...frames) => () => Readable.from([...frames, STOP]);

/** El texto entero en frames de ≤7 chars — la forma del incidente en el wire. */
const chunkedTurn = (text, chunk = 7) => () => {
  const frames = [];
  for (let i = 0; i < text.length; i += chunk) frames.push(answerFrame(text.slice(i, i + chunk)));
  return Readable.from([...frames, STOP]);
};

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
  message_id: 'msg_salvage3',
  model: 'qwen-test',
  hasTools: true,
  toolChoice: 'auto',
  allowedToolNames: ALLOWED,
  toolSchemas: SCHEMAS,
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

const visibleTextOf = (output) =>
  [...output.matchAll(/"delta":\{"type":"text_delta","text":("(?:[^"\\]|\\.)*")\}/g)]
    .map(m => JSON.parse(m[1]))
    .join('');

const thinkingTextOf = (output) =>
  [...output.matchAll(/"delta":\{"type":"thinking_delta","thinking":("(?:[^"\\]|\\.)*")\}/g)]
    .map(m => JSON.parse(m[1]))
    .join('');

const toolArgsOf = (output) =>
  [...output.matchAll(/"delta":\{"type":"input_json_delta","partial_json":("(?:[^"\\]|\\.)*")\}/g)]
    .map(m => JSON.parse(m[1]))
    .join('');

// Turno del incidente 1: la llamada (nombre inventado, JSON valido) abre el turno y
// la narracion viene despues — unknown_tool + prosa visible en el mismo attempt.
const GARBAGE_THEN_PROSE = `${GARBAGE_CALL}\nThe tool seems broken here.`;

describe('loop B: incident-3 wire replay (matrix row 1)', () => {
  it('streams the prose, settles with a Bash tool_use, burns no retry, leaks no marker', async () => {
    const sender = scriptedSender();
    const res = await runStream(chunkedTurn(INCIDENT3), sender);

    assert.equal(sender.calls.length, 0, 'salvage must not burn any retry');
    assert.deepEqual(toolUseNames(res.output), ['Bash']);
    assert.equal(JSON.parse(toolArgsOf(res.output)).command, INCIDENT3_CMD, 'exact find command');
    const visible = visibleTextOf(res.output);
    assert.match(visible, /Voy a listar los archivos/);
    // Criterio de aceptacion 1: cero marcadores en TODO el stream SSE, no solo
    // en los text deltas.
    assert.doesNotMatch(res.output, /TOOL_CALL/i, 'zero trigger bytes anywhere on the wire');
    assert.doesNotMatch(res.output, /END TOOL CALL/i, 'zero closer bytes anywhere on the wire');
    assert.match(res.output, /"stop_reason":"tool_use"/);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });
});

describe('loop B: tool_error after prose → one text-suppressed retry (matrix rows 3-4)', () => {
  it('forwards ONLY tool_use from the retry; its text and thinking never hit the wire', async () => {
    const retryTurn = turnOf(
      thinkFrame('secret retry thinking'),
      answerFrame(`${GOOD_CALL}\nDone reading now.`)
    );
    const sender = scriptedSender(retryTurn);
    let res;
    const warns = await captureWarns(async () => {
      res = await runStream(turnOf(answerFrame(GARBAGE_THEN_PROSE)), sender);
    });

    assert.equal(sender.calls.length, 1, 'exactly the one compensation slot');
    assert.deepEqual(toolUseNames(res.output), ['read_file'], 'the retry\'s valid call is forwarded');
    const visible = visibleTextOf(res.output);
    assert.match(visible, /The tool seems broken here\./, 'attempt-1 prose stays');
    assert.doesNotMatch(visible, /Done reading now/, 'retry text is suppressed');
    assert.doesNotMatch(thinkingTextOf(res.output), /secret retry thinking/, 'retry thinking is suppressed');
    assert.ok(warns.some(l => /被拒绝 \(tool_error\)/.test(l)), `expected the tool_error rejection warn, got:\n${warns.join('\n')}`);
    assert.match(res.output, /"stop_reason":"tool_use"/);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('when the suppressed retry also fails, attempt-1 prose is delivered clean — no 502, no markers', async () => {
    const sender = scriptedSender(turnOf(answerFrame('Nope, still nothing useful.')));
    const res = await runStream(turnOf(answerFrame(GARBAGE_THEN_PROSE)), sender);

    assert.equal(sender.calls.length, 1, 'the slot is single: no second retry');
    const visible = visibleTextOf(res.output);
    assert.match(visible, /The tool seems broken here\./);
    assert.doesNotMatch(visible, /Nope, still nothing useful/, 'failed retry text never reaches the client');
    assert.doesNotMatch(visible, /TOOL.?CALL/i, 'zero protocol bytes on the wire');
    assert.doesNotMatch(res.output, /"type":"error"/, 'prose exists — no 502');
    assert.match(res.output, /"type":"message_stop"/);
  });
});

describe('loop B: delivery strips recorded residue from recoveredBuffer (layer 3)', () => {
  it('slot already burned by missing_tool → tool_error round delivers as-is, but residue-free', async () => {
    // attempt 1: prosa de accion (missing_tool consume el cupo). attempt 2 (retry,
    // sin suprimir): llamada con nombre inventado + prosa → tool_error con cupo
    // agotado → break → entrega. El span condenado esta en recoveredBuffer; la
    // entrega debe pelarlo. Sin la llamada a stripToolCallResidue en :1049 este
    // test falla (mutation check de la capa 3 en B).
    const retryTurn = turnOf(answerFrame(`${GARBAGE_CALL}\nExtra follow-up prose.`));
    const sender = scriptedSender(retryTurn);
    let res;
    const warns = await captureWarns(async () => {
      res = await runStream(turnOf(answerFrame('I will run the build now.')), sender);
    });

    assert.equal(sender.calls.length, 1);
    const visible = visibleTextOf(res.output);
    assert.match(visible, /I will run the build now\./);
    assert.match(visible, /Extra follow-up prose\./);
    assert.doesNotMatch(visible, /TOOL.?CALL/i, 'the condemned span must be stripped at delivery');
    assert.doesNotMatch(visible, /garbage/, 'no payload bytes either');
    assert.ok(warns.some(l => /工具协议出错但已产出内容/.test(l)), 'degraded-delivery warn kept');
    assert.ok(warns.some(l => /剥离协议残渣/.test(l)), 'the strip leaves a log trace');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });
});

describe('loop B: required unfulfilled after streamed prose (matrix row on required)', () => {
  it('closes with end_turn + warn instead of a 502 after streamed prose', async () => {
    const sender = scriptedSender(turnOf(answerFrame('Second prose, still no tool.')));
    let res;
    const warns = await captureWarns(async () => {
      res = await runStream(
        turnOf(answerFrame('Cannot pick a tool, sorry.')),
        sender,
        { toolChoice: 'required' }
      );
    });

    assert.equal(sender.calls.length, 1, 'the single after-prose compensation retry');
    assert.doesNotMatch(res.output, /"type":"error"/, 'a half-delivered message plus error is worse than an unmet required');
    assert.match(res.output, /"stop_reason":"end_turn"/);
    assert.ok(warns.some(l => /required 未兑现/.test(l)), `expected the required-downgrade warn, got:\n${warns.join('\n')}`);
  });

  it('residue-only turn with required still 502s — emptiness is judged on stripped text', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GARBAGE_CALL)), turnOf(answerFrame(GARBAGE_CALL)));
    const res = await runStream(turnOf(answerFrame(GARBAGE_CALL)), sender, { toolChoice: 'required' });

    assert.match(res.output, /invalid_tool_call_error/);
    assert.equal(visibleTextOf(res.output), '', 'no residue may leak as message content');
  });
});

describe('loop B: residue-only turn, retries exhausted (matrix row)', () => {
  it('still 502s exactly as today — never an empty-content message', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GARBAGE_CALL)), turnOf(answerFrame(GARBAGE_CALL)));
    const res = await runStream(turnOf(answerFrame(GARBAGE_CALL)), sender);

    assert.equal(sender.calls.length, 2, 'no prose on the wire → retries run to the cap');
    assert.match(res.output, /invalid_tool_call_error/);
    assert.equal(visibleTextOf(res.output), '');
  });
});

describe('loop C: exhaustion with residue embedded in cleanedText (matrix row)', () => {
  it('delivers with the condemned span removed; detection ran on unstripped text; warn kept', async () => {
    // Una llamada valida + una condenada en el MISMO turno: hay tool_use (no hay
    // 502) y hay error residual → la entrega pela el span condenado. Sin la
    // llamada a stripToolCallResidue en la entrega de C este test falla
    // (mutation check de la capa 3 en C).
    const sender = scriptedSender();
    let res;
    const warns = await captureWarns(async () => {
      res = await runNonStream(turnOf(answerFrame(`${GOOD_CALL}\n${GARBAGE_CALL}`)), sender);
    });

    assert.equal(res.statusCode, 200);
    const blocks = res.body?.content || [];
    assert.deepEqual(blocks.filter(b => b.type === 'tool_use').map(b => b.name), ['read_file']);
    const text = blocks.filter(b => b.type === 'text').map(b => b.text).join('');
    assert.doesNotMatch(text, /TOOL.?CALL/i, 'the condemned span must not be delivered');
    assert.doesNotMatch(text, /garbage/);
    assert.ok(warns.some(l => /剥离协议残渣/.test(l)), `expected the strip warn, got:\n${warns.join('\n')}`);
  });

  it('C incident-3 non-stream: the whole-text path salvages the same call', async () => {
    const sender = scriptedSender();
    const res = await runNonStream(turnOf(answerFrame(INCIDENT3)), sender);

    assert.equal(res.statusCode, 200);
    assert.equal(sender.calls.length, 0, 'no retry burned');
    const blocks = res.body?.content || [];
    const uses = blocks.filter(b => b.type === 'tool_use');
    assert.equal(uses.length, 1);
    assert.equal(uses[0].name, 'Bash');
    assert.equal(uses[0].input.command, INCIDENT3_CMD);
    const text = blocks.filter(b => b.type === 'text').map(b => b.text).join('');
    assert.match(text, /Voy a listar los archivos/);
    assert.doesNotMatch(text, /TOOL.?CALL/i);
    assert.equal(res.body.stop_reason, 'tool_use');
  });
});
