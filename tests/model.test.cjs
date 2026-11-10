const assert = require("node:assert/strict");
const fs = require("node:fs");
const { afterEach, beforeEach, test } = require("node:test");
const ts = require("typescript");

// Exercise the public Planner entry point with the real model request code.
require.extensions[".ts"] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    fileName: filename
  });
  module._compile(outputText, filename);
};
const { testPlanner } = require("../lib/engine.ts");
const { getPlanner } = require("../lib/store.ts");

const input = "预算 80 元，想买点办公室分享的咸口零食";
const validPlan = { goal: "推荐办公室分享零食", reasoning: "按预算筛选并核算价格", steps: [] };
let originalFetch;
let originalEnv;

beforeEach(() => {
  originalFetch = global.fetch;
  originalEnv = { ...process.env };
  process.env.OPENAI_API_KEY = "test-model-key";
  process.env.DEEPSEEK_API_KEY = "";
  process.env.AI_BASE_URL = "https://api.deepseek.com/v1";
  process.env.AI_MODEL = "deepseek-flash";
});

afterEach(() => {
  global.fetch = originalFetch;
  for (const key of Object.keys(process.env)) if (!(key in originalEnv)) delete process.env[key];
  Object.assign(process.env, originalEnv);
});

function mockCompletion(content, finishReason = "stop", inspectRequest = () => {}) {
  global.fetch = async (url, options) => {
    inspectRequest(url, JSON.parse(options.body));
    return Response.json({ choices: [{ finish_reason: finishReason, message: { content } }] });
  };
}

test("DeepSeek structured calls disable thinking without changing the output limit", async () => {
  const planner = await getPlanner();
  mockCompletion(JSON.stringify(validPlan), "stop", (url, body) => {
    assert.equal(url, "https://api.deepseek.com/v1/chat/completions");
    assert.deepEqual(body.thinking, { type: "disabled" });
    assert.equal(body.max_tokens, planner.maxTokens);
    assert.deepEqual(body.response_format, { type: "json_object" });
  });
  assert.equal((await testPlanner(planner, input)).goal, validPlan.goal);
});

test("Volcengine Ark structured calls also disable thinking", async () => {
  process.env.AI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3";
  process.env.AI_MODEL = "ep-test-model";
  mockCompletion(JSON.stringify(validPlan), "stop", (url, body) => {
    assert.equal(url, "https://ark.cn-beijing.volces.com/api/v3/chat/completions");
    assert.deepEqual(body.thinking, { type: "disabled" });
    assert.equal(body.model, "ep-test-model");
  });
  await testPlanner(await getPlanner(), input);
});

test("other compatible services receive no provider-specific thinking parameter", async () => {
  process.env.AI_BASE_URL = "https://api.openai.com/v1";
  process.env.AI_MODEL = "gpt-4o-mini";
  mockCompletion(JSON.stringify(validPlan), "stop", (_url, body) => {
    assert.equal("thinking" in body, false);
  });
  await testPlanner(await getPlanner(), input);
});

test("truncated arrays report the Planner and token limit before JSON parsing", async () => {
  const planner = await getPlanner();
  const truncated = '{"goal":"推荐办公室零食","steps":[{"capability":"need-extraction","type":"skill"}';
  mockCompletion(truncated, "length");
  await assert.rejects(testPlanner(planner, input), error => {
    assert.match(error.message, /customer-service-planner/);
    assert.match(error.message, /截断/);
    assert.match(error.message, /finish_reason=length/);
    assert.ok(error.message.includes(`max_tokens=${planner.maxTokens}`));
    assert.equal(error instanceof SyntaxError, false);
    return true;
  });
});

test("a length-limited response is rejected even if its prefix is valid JSON", async () => {
  mockCompletion(JSON.stringify(validPlan), "length");
  await assert.rejects(testPlanner(await getPlanner(), input), /输出被截断/);
});

test("an empty answer exhausted by reasoning reports truncation", async () => {
  mockCompletion("", "length");
  await assert.rejects(testPlanner(await getPlanner(), input), /输出被截断/);
});

test("malformed model JSON includes the failing capability and finish reason", async () => {
  mockCompletion('{"steps":[{"capability":"need-extraction"} {"capability":"risk-check"}]}');
  await assert.rejects(testPlanner(await getPlanner(), input), error => {
    assert.match(error.message, /customer-service-planner/);
    assert.match(error.message, /无效 JSON/);
    assert.match(error.message, /finish_reason=stop/);
    return true;
  });
});

test("complete JSON wrapped in a Markdown code block remains supported", async () => {
  mockCompletion(`\`\`\`json\n${JSON.stringify(validPlan)}\n\`\`\``);
  assert.equal((await testPlanner(await getPlanner(), input)).goal, validPlan.goal);
});
