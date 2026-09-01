// Cableado de produccion del salvage-3: una sola prueba de punta a punta por
// handleAnthropicMessages — pina la cadena normalizeAnthropicTools →
// function.parameters → buildInternalRequest.toolSchemas → ctx → parser →
// argumentsMatchToolSchema. Borrar toolSchemas del ctx (o del return de
// buildInternalRequest) hace fallar esta prueba; el resto de la suite inyecta los
// handlers directamente y no ve ese cableado.
//
// Los parches de require-cache van ANTES de requerir el controller: chat-helpers y
// anthropic.js capturan estas funciones por destructuring en su primer require.
process.env.AGENT_TURN_MAX_ATTEMPTS = '2';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

// Sin red en tests: el fetch de modelos revienta y chat-helpers cae a sus
// fallbacks por nombre (try/catch propio en isThinkingEnabled/parserModel).
const modelsMap = require('../src/models/models-map.js');
modelsMap.getLatestModels = async () => { throw new Error('offline test: no model fetch'); };

const requestModule = require('../src/utils/request.js');
let upstreamFactory = null;
requestModule.sendChatRequest = async () => ({
  status: true,
  response: upstreamFactory(),
  currentAccount: null
});

const { handleAnthropicMessages } = require('../src/controllers/anthropic.js');

const INCIDENT3_CMD = 'find /Users/pedro/Documents/git/Prueba/payroll/_bmad-output/planning-artifacts/architecture-Español-2026-09-01 -type f 2>/dev/null';
const INCIDENT3 = `[TOOL_CALL]Bash{command:${INCIDENT3_CMD}", "description": "List files"}}\n[END TOOL CALL]\n`;

const answerFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }]
})}\n\n`;
const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

const createRes = () => ({
  statusCode: 200,
  body: null,
  headers: {},
  set(headers) { Object.assign(this.headers, headers); return this; },
  status(code) { this.statusCode = code; return this; },
  json(payload) { this.body = payload; return this; }
});

const BASH_TOOL = {
  name: 'Bash',
  description: 'run a shell command',
  input_schema: {
    type: 'object',
    properties: {
      command: { type: 'string' },
      description: { type: 'string' }
    },
    required: ['command']
  }
};

describe('production wiring: Anthropic request → toolSchemas → salvage gate (e2e)', () => {
  it('an Anthropic-shaped request with input_schema salvages the incident-3 upstream text', async () => {
    upstreamFactory = () => Readable.from([answerFrame(INCIDENT3), STOP]);
    const req = {
      body: {
        model: 'qwen3-coder-plus',
        max_tokens: 512,
        stream: false,
        messages: [{ role: 'user', content: 'lista los archivos del directorio' }],
        tools: [BASH_TOOL]
      }
    };
    const res = createRes();
    await handleAnthropicMessages(req, res);

    assert.equal(res.statusCode, 200, `expected delivery, got ${JSON.stringify(res.body?.error || null)}`);
    const uses = (res.body?.content || []).filter(b => b.type === 'tool_use');
    assert.equal(uses.length, 1, 'the salvage must fire through the production schema plumbing');
    assert.equal(uses[0].name, 'Bash');
    assert.equal(uses[0].input.command, INCIDENT3_CMD, 'exact command through the full pipeline');
    assert.equal(res.body.stop_reason, 'tool_use');
  });

  it('a duplicated tool name disables its schema — ambiguous ⇒ no salvage (fail closed)', async () => {
    upstreamFactory = () => Readable.from([answerFrame(INCIDENT3), STOP]);
    const req = {
      body: {
        model: 'qwen3-coder-plus',
        max_tokens: 512,
        stream: false,
        messages: [{ role: 'user', content: 'lista los archivos' }],
        tools: [BASH_TOOL, { ...BASH_TOOL, input_schema: { type: 'object', properties: {} } }]
      }
    };
    const res = createRes();
    await handleAnthropicMessages(req, res);

    // Sin schema unico no hay abono: el span cae a truncated_tool_call y el turno
    // (sin prosa, con errores) termina en 502 — jamas un tool_use por schema ambiguo.
    const uses = (res.body?.content || []).filter(b => b.type === 'tool_use');
    assert.equal(uses.length, 0, 'an ambiguous schema must never gate a salvage through');
    assert.equal(res.statusCode, 502);
    assert.equal(res.body?.error?.type, 'invalid_tool_call_error');
  });
});
