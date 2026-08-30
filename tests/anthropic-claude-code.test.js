const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

// Test parserModel pass-through (no hardcoded claude mapping)
const { parserModel } = require('../src/utils/chat-helpers.js');

// Test anthropic compatibility - max_tokens no longer ignored
const { analyzeAnthropicCompatibility } = require('../src/controllers/anthropic.compatibility.js');

// Test authorization error format
const { validateApiKey } = require('../src/middlewares/authorization.js');

// Test buildInternalRequest envelope structure
const anthropicController = require('../src/controllers/anthropic.js');

describe('Claude Code Compatibility', () => {
  describe('Model Pass-Through', () => {
    it('passes qwen model names through to upstream lookup', async () => {
      const result = await parserModel('qwen3-coder-plus');
      assert.ok(typeof result === 'string');
      assert.ok(result.length > 0);
    });

    it('returns default for empty model', async () => {
      const result = await parserModel('');
      assert.equal(result, 'qwen3-coder-plus');
    });

    it('returns default for null model', async () => {
      const result = await parserModel(null);
      assert.equal(result, 'qwen3-coder-plus');
    });

    it('passes non-claude custom model names through', async () => {
      const result = await parserModel('my-custom-model');
      assert.ok(typeof result === 'string');
    });
  });

  describe('max_tokens Handling', () => {
    it('does not list max_tokens in ignored fields', () => {
      const result = analyzeAnthropicCompatibility({ max_tokens: 8192, model: 'test', messages: [] });
      assert.ok(!result.ignoredFields.includes('max_tokens'), 'max_tokens should not be ignored');
    });

    it('lists max_tokens as neither partial nor future_risk', () => {
      const result = analyzeAnthropicCompatibility({ max_tokens: 8192, model: 'test', messages: [] });
      assert.ok(!result.partialFields.includes('max_tokens'));
      assert.ok(!result.futureRiskFields.includes('max_tokens'));
    });
  });

  describe('Thinking Budget Cap', () => {
    it('accepts thinking budgets up to 131072', async () => {
      const { isThinkingEnabled } = require('../src/utils/chat-helpers.js');
      const config = await isThinkingEnabled('qwen3-coder-plus-thinking', true, 100000);
      assert.equal(config.thinking_enabled, true);
      assert.equal(config.budget, 100000);
    });

    it('rejects thinking budgets at or above 131072', async () => {
      const { isThinkingEnabled } = require('../src/utils/chat-helpers.js');
      const config = await isThinkingEnabled('qwen3-coder-plus-thinking', true, 131072);
      assert.equal(config.thinking_enabled, true);
      assert.equal(config.budget, undefined);
    });
  });

  describe('Authorization Error Format', () => {
    it('validateApiKey returns isValid false for invalid keys', () => {
      const result = validateApiKey('invalid-key');
      assert.equal(result.isValid, false);
    });

    it('validateApiKey returns isValid false for empty key', () => {
      const result = validateApiKey('');
      assert.equal(result.isValid, false);
    });
  });

  describe('Response Field Completeness', () => {
    it('flattenAnthropicMessages handles basic messages', () => {
      const messages = [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'hi there' }
      ];
      const result = anthropicController.flattenAnthropicMessages(messages);
      assert.equal(result.length, 2);
      assert.equal(result[0].role, 'user');
      assert.equal(result[1].role, 'assistant');
    });

    it('normalizeAnthropicTools converts Anthropic tools to OpenAI format', () => {
      const tools = [{
        name: 'get_weather',
        description: 'Get weather',
        input_schema: { type: 'object', properties: { city: { type: 'string' } } }
      }];
      const result = anthropicController.normalizeAnthropicTools(tools);
      assert.equal(result.length, 1);
      assert.equal(result[0].type, 'function');
      assert.equal(result[0].function.name, 'get_weather');
    });

    it('mapAnthropicStopReason maps end_turn correctly', () => {
      const result = anthropicController.mapAnthropicStopReason('stop', false, true);
      assert.equal(result, 'end_turn');
    });

    it('mapAnthropicStopReason maps tool_use correctly', () => {
      const result = anthropicController.mapAnthropicStopReason(null, true, true);
      assert.equal(result, 'tool_use');
    });

    it('mapAnthropicStopReason maps max_tokens correctly', () => {
      const result = anthropicController.mapAnthropicStopReason('length', false, true);
      assert.equal(result, 'max_tokens');
    });
  });
});
