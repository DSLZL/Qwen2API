const test = require('node:test');
const assert = require('node:assert/strict');

const cliManager = require('../src/utils/cli.manager.js');
const cliSupport = require('../src/utils/cli-support.js');

test('pollForToken stops after 3 unsuccessful attempts', async () => {
  const originalFetch = global.fetch;
  const originalSetTimeout = global.setTimeout;

  let attempts = 0;
  global.fetch = async () => {
    attempts += 1;
    return {
      ok: false,
      status: 504,
      statusText: 'Gateway Time-out',
      headers: new Map([['content-type', 'text/html']]),
      text: async () => '<html>504 Gateway Time-out</html>'
    };
  };
  global.setTimeout = (fn) => {
    fn();
    return 0;
  };

  try {
    const result = await cliManager.pollForToken('device-code', 'code-verifier');
    assert.equal(attempts, 3);
    assert.deepEqual(result, {
      status: false,
      access_token: null,
      refresh_token: null,
      expiry_date: null
    });
  } finally {
    global.fetch = originalFetch;
    global.setTimeout = originalSetTimeout;
  }
});

test('getAccountCliState reports unsupported CLI without touching account health', () => {
  const state = cliSupport.getAccountCliState({
    cli_info: null,
    cli_unavailable_reason: 'unsupported',
    expires: Math.floor(Date.now() / 1000) + 24 * 60 * 60,
    stats: { chat: { input: 0, output: 0 }, cli: { calls: 0, input: 0, output: 0 } }
  }, {}, Date.now());

  assert.equal(state.status.cli, 'unsupported');
  assert.equal(state.status.kind, 'active');
  assert.equal(state.cliQuotaLimit, 0);
  assert.equal(state.cliRequestNumber, 0);
});

test('getAccountCliState keeps normal accounts on default CLI quota', () => {
  const state = cliSupport.getAccountCliState({
    cli_info: { request_number: 12 },
    expires: Math.floor(Date.now() / 1000) + 24 * 60 * 60,
    stats: { chat: { input: 0, output: 0 }, cli: { calls: 2, input: 10, output: 20 } }
  }, {}, Date.now());

  assert.equal(state.status.kind, 'active');
  assert.equal(state.status.cli, 'available');
  assert.equal(state.cliQuotaLimit, 2000);
  assert.equal(state.cliRequestNumber, 12);
});

test('getAccountCliState marks uninitialized accounts as CLI pending', () => {
  const state = cliSupport.getAccountCliState({
    cli_info: null,
    cli_unavailable_reason: null,
    expires: Math.floor(Date.now() / 1000) + 24 * 60 * 60,
    stats: { chat: { input: 0, output: 0 }, cli: { calls: 0, input: 0, output: 0 } }
  }, {}, Date.now());

  assert.equal(state.status.cli, 'pending');
  assert.equal(state.status.kind, 'active');
  assert.equal(state.cliQuotaLimit, 0);
  assert.equal(state.cliRequestNumber, 0);
});

// The whole point of splitting kind and cli: with a single field the disabled CLI
// shadowed every health state, so a cooling-down account looked perfectly normal.
test('CLI availability never shadows account health', () => {
  const now = Date.now();

  const cooling = cliSupport.getAccountCliState({
    cli_info: null,
    cli_unavailable_reason: 'disabled',
    expires: Math.floor(now / 1000) + 24 * 60 * 60,
    stats: { chat: { input: 0, output: 0 }, cli: { calls: 0, input: 0, output: 0 } }
  }, { cooldownEndsAt: now + 60 * 1000 }, now);

  assert.equal(cooling.status.kind, 'cooldown');
  assert.equal(cooling.status.cli, 'disabled');
  assert.equal(cooling.status.cooldownEndsAt, now + 60 * 1000);

  const expiring = cliSupport.getAccountCliState({
    cli_info: null,
    cli_unavailable_reason: 'disabled',
    expires: Math.floor(now / 1000) + 60 * 60,
    stats: { chat: { input: 0, output: 0 }, cli: { calls: 0, input: 0, output: 0 } }
  }, {}, now);

  assert.equal(expiring.status.kind, 'token_expiring');
  assert.equal(expiring.status.cli, 'disabled');
});

test('getCliAvailability maps every unavailability reason', () => {
  assert.equal(cliSupport.getCliAvailability({ cli_unavailable_reason: 'disabled' }), 'disabled');
  assert.equal(cliSupport.getCliAvailability({ cli_unavailable_reason: 'unsupported' }), 'unsupported');
  assert.equal(cliSupport.getCliAvailability({ cli_info: null }), 'pending');
  assert.equal(cliSupport.getCliAvailability({ cli_info: { request_number: 1 } }), 'available');
});

// Regression: `chatBaseUrl` used to be declared inside the try, so the catch's
// log line threw a ReferenceError that masked the real authorization failure
// and turned the promise into a rejection instead of `false`.
test('authorizeLogin returns false and logs the URL when authorization fails', async () => {
  const { logger } = require('../src/utils/logger');
  const originalFetch = global.fetch;
  const originalError = logger.error;

  const logged = [];
  logger.error = (...args) => { logged.push(args); };

  try {
    global.fetch = async () => ({
      ok: false,
      status: 403,
      statusText: 'Forbidden',
      headers: new Map([['content-type', 'text/plain']]),
      text: async () => 'denied'
    });
    const onNonOk = await cliManager.authorizeLogin('user-code', 'token');
    assert.equal(onNonOk, false);

    global.fetch = async () => { throw new Error('network down'); };
    const onThrow = await cliManager.authorizeLogin('user-code', 'token');
    assert.equal(onThrow, false);

    const urlLogs = logged.filter(args =>
      args.some(a => a && typeof a === 'object' && String(a.url || '').includes('/api/v2/oauth2/authorize')));
    assert.equal(urlLogs.length, 2);
  } finally {
    global.fetch = originalFetch;
    logger.error = originalError;
  }
});
