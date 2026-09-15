// Unidad: normalizacion del `usage` upstream. Qwen (DashScope) manda
// input_tokens / output_tokens; OpenAI manda prompt_tokens / completion_tokens.
// El proxy reporta en formato OpenAI y estima localmente SOLO lo que upstream no dio.

const test = require('node:test');
const { describe, it } = test;
const assert = require('node:assert/strict');

const {
  normalizeUpstreamUsage,
  mergeUpstreamUsage,
  resolveUsage,
  describeUsageSource
} = require('../src/utils/precise-tokenizer.js');

describe('normalizeUpstreamUsage', () => {
  it('DashScope naming (what Qwen sends) → OpenAI naming', () => {
    assert.deepEqual(
      normalizeUpstreamUsage({ input_tokens: 651, output_tokens: 9, total_tokens: 660 }),
      { prompt_tokens: 651, completion_tokens: 9 }
    );
  });

  it('OpenAI naming passes through', () => {
    assert.deepEqual(
      normalizeUpstreamUsage({ prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 }),
      { prompt_tokens: 12, completion_tokens: 3 }
    );
  });

  it('partial usage: the present field is kept, the missing one is null', () => {
    assert.deepEqual(normalizeUpstreamUsage({ input_tokens: 651 }), { prompt_tokens: 651, completion_tokens: null });
    assert.deepEqual(normalizeUpstreamUsage({ output_tokens: 9 }), { prompt_tokens: null, completion_tokens: 9 });
  });

  it('numeric strings are coerced', () => {
    assert.deepEqual(normalizeUpstreamUsage({ input_tokens: '651', output_tokens: '9' }), { prompt_tokens: 651, completion_tokens: 9 });
  });

  it('negative, NaN and non-numeric fields count as absent', () => {
    assert.deepEqual(normalizeUpstreamUsage({ input_tokens: -1, output_tokens: 'nine' }), null);
    assert.deepEqual(normalizeUpstreamUsage({ input_tokens: NaN, output_tokens: 4 }), { prompt_tokens: null, completion_tokens: 4 });
  });

  it('non-object or no usable field → null', () => {
    for (const raw of [null, undefined, 'usage', 42, [], {}, { foo: 1 }]) {
      assert.equal(normalizeUpstreamUsage(raw), null, `raw=${JSON.stringify(raw)}`);
    }
  });

  it('zero is "not reported": all-zero → null, a single zero field → null for that field', () => {
    assert.equal(normalizeUpstreamUsage({ input_tokens: 0, output_tokens: 0, total_tokens: 0 }), null);
    assert.deepEqual(normalizeUpstreamUsage({ input_tokens: 651, output_tokens: 0 }), { prompt_tokens: 651, completion_tokens: null });
  });
});

describe('describeUsageSource', () => {
  it('"upstream" only when both counters came from upstream; otherwise "estimated"', () => {
    assert.equal(describeUsageSource({ prompt_tokens: 651, completion_tokens: 9 }), 'upstream');
    assert.equal(describeUsageSource({ prompt_tokens: 651, completion_tokens: null }), 'estimated');
    assert.equal(describeUsageSource(null), 'estimated');
  });
});

describe('mergeUpstreamUsage (per-frame accumulation)', () => {
  it('cumulative counts: the last frame that reports a field wins', () => {
    let acc = null;
    acc = mergeUpstreamUsage(acc, { input_tokens: 651, output_tokens: 1 });
    acc = mergeUpstreamUsage(acc, { input_tokens: 651, output_tokens: 2 });
    assert.deepEqual(acc, { prompt_tokens: 651, completion_tokens: 2 });
  });

  it('a frame without usage (or with a partial one) keeps what was already accumulated', () => {
    let acc = mergeUpstreamUsage(null, { input_tokens: 651, output_tokens: 5 });
    acc = mergeUpstreamUsage(acc, undefined);
    acc = mergeUpstreamUsage(acc, { output_tokens: 7 });
    assert.deepEqual(acc, { prompt_tokens: 651, completion_tokens: 7 });
  });
});

describe('resolveUsage (fill only what upstream never reported)', () => {
  const estimate = () => ({ prompt_tokens: 8, completion_tokens: 3, total_tokens: 11 });

  it('upstream reported both → estimator is not even called', () => {
    let calls = 0;
    const usage = resolveUsage({ prompt_tokens: 651, completion_tokens: 9 }, () => { calls++; return estimate(); });
    assert.deepEqual(usage, { prompt_tokens: 651, completion_tokens: 9, total_tokens: 660 });
    assert.equal(calls, 0);
  });

  it('only the missing field is estimated; total is recomputed', () => {
    assert.deepEqual(
      resolveUsage({ prompt_tokens: 651, completion_tokens: null }, estimate),
      { prompt_tokens: 651, completion_tokens: 3, total_tokens: 654 }
    );
  });

  it('nothing from upstream → full estimate', () => {
    assert.deepEqual(resolveUsage(null, estimate), { prompt_tokens: 8, completion_tokens: 3, total_tokens: 11 });
  });
});
