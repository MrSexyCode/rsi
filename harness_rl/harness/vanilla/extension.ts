// Harness extension: turns off every tool and controls the system prompt as harness.json says.
//   system_prompt.mode = "replace": send system_prompt.md verbatim; an empty file sends no system message.
//   system_prompt.mode = "pi-default": keep Pi's own prompt, with the <cwd> section pinned to a constant path.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(here, "harness.json"), "utf8"));

export default function (pi: any) {
	pi.on("session_start", () => {
		pi.setActiveTools([]);
	});
	const sp = manifest.system_prompt;
	if (sp.mode === "replace") {
		const text = readFileSync(join(here, sp.file), "utf8");
		pi.on("before_agent_start", () => ({ systemPrompt: text }));
	} else if (sp.mode === "pi-default") {
		pi.on("before_agent_start", (event: any) => {
			event.systemPromptOptions.cwd = sp.cwd;
		});
	} else {
		throw new Error(`unknown system_prompt.mode: ${sp.mode}`);
	}
}
