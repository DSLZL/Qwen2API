const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const dashboard = fs.readFileSync(require.resolve('../public/src/views/dashboard.vue'), 'utf8');

test('dashboard renders inactive CLI states as hover-only gray hint', () => {
  assert.match(dashboard, /v-if="isCliInactive\(token\.email\)"/);
  assert.match(dashboard, /:title="getCliTooltip\(token\.email\)"/);
  assert.match(dashboard, /getCliInactiveLabel\(token\.email\)/);
  assert.match(dashboard, /text-gray-400/);
  assert.match(dashboard, /v-if="cliExpanded && getCliState\(token\.email\) === 'available'"/);
});

test('inactive CLI states cover both unsupported and disabled', () => {
  assert.match(dashboard, /unsupported: 'dash\.acct\.cliUnavailableShort'/);
  assert.match(dashboard, /disabled: 'dash\.acct\.cliDisabledShort'/);
  assert.match(dashboard, /cli === 'disabled'/);
});

test('dashboard renders pending CLI state as blue hint', () => {
  assert.match(dashboard, /v-else-if="getCliState\(token\.email\) === 'pending'"/);
  assert.match(dashboard, /cliPendingShort/);
  assert.match(dashboard, /text-blue-400/);
});

// The badge answers "is this account alive"; CLI availability must not leak into it.
test('status badge carries only health states', () => {
  const map = dashboard.match(/const STATUS_EMOJI = Object\.freeze\(\{[^}]+\}\)/);
  assert.ok(map, 'STATUS_EMOJI map not found');
  assert.doesNotMatch(map[0], /cli_/);
  for (const kind of ['active', 'warn', 'cooldown', 'token_expiring']) {
    assert.match(map[0], new RegExp(`${kind}:`));
  }
});

test('every locale defines the new cli_disabled strings', () => {
  for (const lang of ['ru', 'zh', 'en']) {
    const messages = JSON.parse(fs.readFileSync(require.resolve(`../public/src/locales/${lang}.json`), 'utf8'));
    assert.ok(messages.dash.acct.cliDisabledShort, `${lang}: dash.acct.cliDisabledShort is missing`);
    assert.ok(messages.dash.acct.status.cliDisabled, `${lang}: dash.acct.status.cliDisabled is missing`);
  }
});
