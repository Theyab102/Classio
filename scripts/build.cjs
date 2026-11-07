const { spawnSync } = require("node:child_process");
const env = { ...process.env, CI: "false" };
const blocked = Object.keys(env).filter(name => /^REACT_APP_.*(?:GROQ|GEMINI|OPENAI|ANTHROPIC|API_KEY|SERVICE_ACCOUNT|RATE_LIMIT_SECRET)/i.test(name));
for (const name of blocked) delete env[name];
if (blocked.length) console.warn("Excluded secret-related public build variable names:", blocked.join(", "));
const result = spawnSync(process.execPath, [require.resolve("react-scripts/scripts/build")], { env, stdio: "inherit" });
process.exit(result.status ?? 1);
