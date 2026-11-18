const { spawnSync } = require("node:child_process");
require("./check-vendor.cjs");
const env = { ...process.env, CI: "false", GENERATE_SOURCEMAP: "false" };
const blocked = Object.keys(env).filter((name) =>
  /^(?:REACT_APP_|NEXT_PUBLIC_).*(?:GROQ|GEMINI|OPENROUTER|CLOUDFLARE|OLLAMA|OPENAI|ANTHROPIC|KEY|TOKEN|CREDENTIAL|SERVICE_ACCOUNT|SECRET|PASSWORD|PRIVATE)/i.test(
    name,
  ),
);
for (const name of blocked) delete env[name];
if (blocked.length)
  console.warn(
    "Excluded secret-related public build variable names:",
    blocked.join(", "),
  );
const result = spawnSync(
  process.execPath,
  [require.resolve("react-scripts/scripts/build")],
  { env, stdio: "inherit" },
);
if (result.status === 0) {
  const fs = require("node:fs"),
    path = require("node:path"),
    target = path.join(__dirname, "../build/local-ai");
  fs.mkdirSync(target, { recursive: true });
  const source = path.dirname(require.resolve("onnxruntime-web"));
  for (const name of fs
    .readdirSync(source)
    .filter((name) =>
      /^ort-wasm-simd-threaded(?:\.(?:jsep|asyncify|jspi))?\.(?:mjs|wasm)$/.test(
        name,
      ),
    ))
    fs.copyFileSync(path.join(source, name), path.join(target, name));
  fs.copyFileSync(
    path.join(__dirname, "../public/local-ai-runtime-license.txt"),
    path.join(target, "LICENSE.txt"),
  );
}
process.exit(result.status ?? 1);
