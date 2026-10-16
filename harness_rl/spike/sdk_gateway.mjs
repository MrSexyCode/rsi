// 0.2 spike: one long-lived Node process running many Pi sessions through the SDK.
// Reads JSONL requests on stdin: {"id": "...", "prompt": "..."}; writes JSONL results on stdout:
// {"id", "t_start", "t_end", "stopReason", "text", "error"}. Each request gets a fresh in-memory session,
// no tools, no discovery, an empty system prompt, and an X-Rollout-Id header equal to its id.
// Usage: PI_AGENT_DIR=<dir with models.json> node sdk_gateway.mjs
import { createInterface } from "node:readline";

const PKG = "/home/ubuntu/rsi/harness_rl/pi_runtime/node_modules/@earendil-works/pi-coding-agent/dist/index.js";
const { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } = await import(PKG);

const agentDir = process.env.PI_AGENT_DIR;
const cwd = process.env.PI_CWD ?? process.cwd();
const modelRuntime = await ModelRuntime.create({ authPath: `${agentDir}/auth.json`, modelsPath: `${agentDir}/models.json` });
await modelRuntime.setRuntimeApiKey("rlproxy", "sdk");
const model = modelRuntime.getModel("rlproxy", "qwen3-8b");
if (!model) throw new Error("model rlproxy/qwen3-8b not found");

async function runOne(req) {
	const t_start = Date.now() / 1000;
	const extension = (pi) => {
		pi.on("before_agent_start", () => ({ systemPrompt: "" }));
		pi.on("before_provider_headers", (event) => { event.headers["X-Rollout-Id"] = req.id; });
	};
	const resourceLoader = new DefaultResourceLoader({
		cwd, agentDir, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
		extensionFactories: [extension], appendSystemPrompt: [],
	});
	await resourceLoader.reload();
	const { session } = await createAgentSession({
		cwd, agentDir, modelRuntime, model, thinkingLevel: "off", noTools: "all", resourceLoader,
		sessionManager: SessionManager.inMemory(cwd),
		settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }),
	});
	try {
		await session.bindExtensions?.({});
		await session.prompt(req.prompt);
		const last = [...session.messages].reverse().find((m) => m.role === "assistant");
		return { id: req.id, t_start, t_end: Date.now() / 1000, stopReason: last?.stopReason, errorMessage: last?.errorMessage, text: session.getLastAssistantText() };
	} catch (e) {
		return { id: req.id, t_start, t_end: Date.now() / 1000, error: String(e) };
	} finally {
		session.dispose();
	}
}

const rl = createInterface({ input: process.stdin });
const pending = [];
rl.on("line", (line) => {
	if (!line.trim()) return;
	const req = JSON.parse(line);
	pending.push(runOne(req).then((r) => process.stdout.write(JSON.stringify(r) + "\n")));
});
rl.on("close", async () => {
	await Promise.all(pending);
	process.exit(0);
});
process.stdout.write(JSON.stringify({ ready: true }) + "\n");
