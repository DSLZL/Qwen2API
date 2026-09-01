---
title: 'Anthropic Endpoint Agent Loop Parity'
type: 'bugfix'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 1
baseline_commit: 'cd38d91055a9a142d9e269004ce7d30bde0447d2'
context:
- CLAUDE.md
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Claude Code hallucinates "Tool Bash does not exists" / "Tool Read does not exists" when using the Anthropic-compatible `/v1/messages` endpoint, but works correctly through the OpenAI-compatible `/v1/chat/completions` endpoint. Root cause verified by code inspection: `buildInternalRequest` in `src/controllers/anthropic.js` (lines 216-333) is missing two critical agent-loop injections that `chat-middleware.js` applies to every tool-enabled OpenAI request:

1. **Missing `ensureAgentCurrentEnvelope`** — wraps user content with `# Current message` marker + JSON structure so Qwen upstream distinguishes current turn from history.
2. **Missing `buildAgentTurnDirective`** — appends explicit agent-loop contract instructing model that client executes tools and sends results back, preventing premature completion or tool-name hallucination.
3. **Missing `afterToolResult` detection** — without detecting when last message is a `tool_result`, the directive always uses initial-task framing instead of continuation framing during multi-turn tool loops.

Without these, Qwen model receives raw tool definitions but no behavioral contract, causing it to treat tools as informational rather than actionable and hallucinate validation errors.

**Approach:** Add the three missing pieces to `buildInternalRequest` in exact same order as OpenAI path (`chat-middleware.js` lines 144-172): envelope wrap → prefix system+tool prompt → append turn directive. Detect `afterToolResult` from original messages array before flattening.

## Boundaries & Constraints

**Always:**
- Touch only `src/controllers/anthropic.js` — specifically imports (line 14) and `buildInternalRequest` function (lines 216-333).
- Match existing code style: CommonJS requires, Chinese comments where adjacent code uses them, same variable naming patterns.
- `ensureAgentCurrentEnvelope` imported from `chat-middleware.js` (line 205 export).
- `buildAgentTurnDirective` added to existing import from `agent-turn.js` (line 14).
- Verify no require cycle between `anthropic.js` and `chat-middleware.js` after adding cross-import.

**Ask First:** None.

**Never:**
- Modify `chat-middleware.js`, `agent-turn.js`, or `tool-prompt.js` — they already export needed functions.
- Change response handling, SSE streaming, tag stripping, or error formatting — separate concerns.
- Inject agent-loop primitives when `hasTools` is false.

</frozen-after-approval>

## Code Map

- `src/controllers/anthropic.js:14` -- Import line for `agent-turn.js`; add `buildAgentTurnDirective` to existing destructured require.
- `src/controllers/anthropic.js:216-333` -- `buildInternalRequest` function; sole modification target.
- `src/controllers/anthropic.js:217-223` -- Before `flattenAnthropicMessages(messages)` call at line 223: detect `afterToolResult` by checking if last entry in original `messages` array has role `user` with any `tool_result` content block. Pattern: `const originalLast = Array.isArray(messages) ? messages[messages.length - 1] : null; const afterToolResult = originalLast?.role === 'user' && Array.isArray(originalLast?.content) && originalLast.content.some(b => b?.type === 'tool_result');`
- `src/controllers/anthropic.js:228-229` -- `hasTools` flag and `toolPrompt` construction; gate all new injections on `hasTools`.
- `src/controllers/anthropic.js:242-262` -- Prefix concatenation block. After prefix+content assembly completes (after line 262), apply `ensureAgentCurrentEnvelope(last.content, last.role || 'user')` to the content, then append `buildAgentTurnDirective({ afterToolResult })` after the wrapped content. This matches OpenAI ordering: envelope wrap first, then prefix prepend, then directive append.
- `src/middlewares/chat-middleware.js:16-38` -- `ensureAgentCurrentEnvelope` definition; idempotent guard at line 19 prevents double-wrapping when `parserMessages` already added JSONL markers. Reference for behavior, do NOT modify.
- `src/middlewares/chat-middleware.js:115` -- OpenAI `afterToolResult` detection pattern (checks `role === 'tool'`). Anthropic equivalent checks for `tool_result` content block in user message instead.
- `src/middlewares/chat-middleware.js:144-172` -- OpenAI injection ordering reference: envelope → prefix → directive.
- `src/utils/agent-turn.js:253-270` -- `buildAgentTurnDirective` definition; accepts `{ afterToolResult }` boolean.
- `src/utils/agent-turn.js:299` -- Export of `buildAgentTurnDirective`.
- `src/middlewares/chat-middleware.js:205` -- Export of `ensureAgentCurrentEnvelope`.

## Tasks & Acceptance

**Execution:**
- [ ] `src/controllers/anthropic.js` -- Add `buildAgentTurnDirective` to line 14 import from `agent-turn.js` -- Required function currently missing from Anthropic path.
- [ ] `src/controllers/anthropic.js` -- Add new require for `ensureAgentCurrentEnvelope` from `../middlewares/chat-middleware.js` -- Function lives in middleware, not utils. Verify no circular dependency after adding.
- [ ] `src/controllers/anthropic.js` -- Detect `afterToolResult` from original `messages` array before `flattenAnthropicMessages` call (before line 223) -- Check last message for `tool_result` content block presence to match OpenAI path semantics adapted for Anthropic format.
- [ ] `src/controllers/anthropic.js` -- Inside `hasTools` guard after prefix concatenation (after line 262): wrap `last.content` with `ensureAgentCurrentEnvelope(content, role)`, then append `buildAgentTurnDirective({ afterToolResult })` after full content assembly -- Matches OpenAI injection ordering exactly.
- [ ] `src/controllers/anthropic.js` -- Handle edge case where `last.content` is undefined or empty string -- `ensureAgentCurrentEnvelope` handles this via `String(text || '')` coercion, but verify no literal "undefined" string leaks into output.

**Acceptance Criteria:**
- Given an Anthropic `/v1/messages` request with non-empty `tools` array, when `buildInternalRequest` runs, then the last parsed message content contains `# Agent loop control (highest-priority output contract)` appended after user content.
- Given an Anthropic `/v1/messages` request with non-empty `tools` array, when `buildInternalRequest` runs, then the last parsed message content is wrapped with `# Current message` marker via `ensureAgentCurrentEnvelope` before tool/system prefix is prepended.
- Given an Anthropic `/v1/messages` request where the last original message contains a `tool_result` content block, when `buildInternalRequest` detects `afterToolResult`, then `buildAgentTurnDirective` receives `{ afterToolResult: true }` and produces continuation framing.
- Given an Anthropic `/v1/messages` request with no `tools` or empty array, when `buildInternalRequest` runs, then neither `ensureAgentCurrentEnvelope` nor `buildAgentTurnDirective` is invoked.
- Given an Anthropic `/v1/messages` request where the last message has no text content (only `tool_use` blocks), when `buildInternalRequest` runs, then no literal "undefined" string appears in the output content.
- Given the test suite at `tests/`, when `npm test` runs, then all existing tests pass without modification.
- Given the new cross-module require, when Node loads `anthropic.js`, then no circular dependency warning or runtime error occurs.

## Spec Change Log

- Party mode review identified 3 risks: (1) potential require cycle from anthropic→chat-middleware, (2) undefined content edge case, (3) afterToolResult detection timing. Investigation confirmed `ensureAgentCurrentEnvelope` is idempotent (guard at chat-middleware.js:19) and safe to add — no double-wrapping risk. Amendments applied: added no-cycle verification task, added undefined-content AC, specified exact afterToolResult detection code for Anthropic format.

## Verification

**Commands:**
- `npm test` -- expected: all tests pass, zero failures.
- `node -e "require('./src/controllers/anthropic.js')"` -- expected: no circular dependency error or warning.

**Manual checks (if no CLI):**
- Send Anthropic `/v1/messages` request with tools via curl; verify upstream Qwen request body contains `# Agent loop control` and `# Current message` markers in last message content.
- Send follow-up request with `tool_result` block; verify directive text includes "The current message is a tool result from the same unfinished task".

## Suggested Review Order

**Agent-loop injection entry point**

- New imports enabling agent-loop parity with OpenAI path
  [`anthropic.js:14`](../../src/controllers/anthropic.js#L14)

**afterToolResult detection**

- Detects tool_result in original Anthropic messages before flattening
  [`anthropic.js:223`](../../src/controllers/anthropic.js#L223)

**tool_choice=none guard**

- Prevents agent injection when tools disabled, matching OpenAI semantics
  [`anthropic.js:234`](../../src/controllers/anthropic.js#L234)

**Envelope + directive injection block**

- Wraps content and appends turn directive after prefix assembly
  [`anthropic.js:269`](../../src/controllers/anthropic.js#L269)
