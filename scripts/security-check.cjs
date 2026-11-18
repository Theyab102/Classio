const fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  cp = require("node:child_process"),
  crypto = require("node:crypto");
const root = path.resolve(__dirname, ".."),
  mode = process.argv[2] || "all";
const run = (args, options = {}) =>
  cp.spawnSync("git", args, { cwd: root, encoding: "utf8", ...options });
const scanner = process.env.GITLEAKS_BIN || "gitleaks",
  temp = fs.mkdtempSync(path.join(os.tmpdir(), "classio-security-"));
const publicFirebaseHash =
  "f44974192bbe24af103ac60856d08e20f1dcc9de8b0aae2a205644596555c596";
const sha = (value) => crypto.createHash("sha256").update(value).digest("hex");
let errors = [];
function classify(rows, history, bundle = false, staged = false) {
  return rows.filter((row) => {
    if (bundle && row.RuleID === "gcp-api-key") {
      const content = fs.readFileSync(
          path.join(root, "build", row.File),
          "utf8",
        ),
        line = content.split("\n")[row.StartLine - 1] || "",
        keys = line.match(/AIza[\w-]{30,50}/g) || [];
      return (
        !keys.length ||
        keys.some((value) => sha(value) !== publicFirebaseHash) ||
        !content.includes("firebaseapp.com")
      );
    }
    if (
      row.RuleID !== "gcp-api-key" ||
      row.File.replaceAll("\\", "/") !== "src/App.js"
    )
      return true;
    const content = history
      ? run(["show", row.Commit + ":" + row.File]).stdout
      : staged
        ? run(["show", ":" + row.File]).stdout
        : fs.readFileSync(path.join(root, row.File), "utf8");
    const line = content.split("\n")[row.StartLine - 1] || "",
      key = /apiKey:\s*["'](AIza[\w-]+)["']/.exec(line)?.[1];
    // Exact reviewed Firebase public web-config value only. Other Google/provider
    // keys, different values, locations or files remain blocking findings.
    return !(
      key &&
      sha(key) === publicFirebaseHash &&
      content.includes("firebaseapp.com") &&
      content.includes("initializeApp")
    );
  });
}
function scan(args, name, history = false) {
  const report = path.join(temp, name + ".json"),
    result = cp.spawnSync(
      scanner,
      [
        ...args,
        "--config",
        path.join(root, ".gitleaks.toml"),
        "--redact=100",
        "--no-banner",
        "--report-format=json",
        "--report-path=" + report,
        "--exit-code=10",
        "--log-level=error",
      ],
      { cwd: root, encoding: "utf8" },
    );
  if (result.error || ![0, 10].includes(result.status))
    throw Error(
      "Gitleaks did not complete. Install the checksum-verified scanner; scanning cannot be skipped.",
    );
  const rows = fs.existsSync(report)
    ? JSON.parse(fs.readFileSync(report, "utf8"))
    : [];
  for (const row of rows)
    if (path.isAbsolute(row.File))
      row.File = path.relative(args[0] === "dir" ? args[1] : root, row.File);
  const blocked = classify(rows, history, name === "bundle", name === "staged");
  for (const row of blocked)
    errors.push({
      check: name,
      rule: row.RuleID,
      file: row.File,
      line: row.StartLine,
      commit: row.Commit?.slice(0, 12),
    });
  console.log(
    name +
      ": " +
      blocked.length +
      " private-credential findings; " +
      (rows.length - blocked.length) +
      " reviewed public Firebase configuration matches.",
  );
}
function runtimeSecrets() {
  return Object.entries(process.env).filter(
    ([name, value]) =>
      /(?:KEY|TOKEN|SECRET|PASSWORD|SERVICE_ACCOUNT)/i.test(name) &&
      !/^REACT_APP_(?:CLASSIO_CONTACT|CLASSIO_COPYRIGHT)/.test(name) &&
      value &&
      value.length >= 16 &&
      value !== "true" &&
      value !== "false",
  );
}
function checkBytes(file, bytes) {
  for (const [name, value] of runtimeSecrets())
    if (bytes.includes(Buffer.from(value)))
      errors.push({ check: "runtime-secret", file, variable: name });
}
try {
  const tracked = run([
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
    "-z",
  ])
    .stdout.split("\0")
    .filter(Boolean);
  for (const file of tracked) {
    if (
      /(?:^|\/)\.env(?:\.|$)/.test(file) &&
      !/^\.env(?:\..+)?\.example$/.test(file)
    )
      errors.push({ check: "tracked-environment", file });
    if (/(?:^|\/)\.vercel\/|\.(?:pem|key)$|service-account.*\.json$/.test(file))
      errors.push({ check: "tracked-credential-export", file });
  }
  if (mode === "staged") {
    scan(["git", "--staged"], "staged");
    const staged = run([
      "diff",
      "--cached",
      "--name-only",
      "--diff-filter=ACMR",
      "-z",
    ])
      .stdout.split("\0")
      .filter(Boolean);
    for (const file of staged)
      checkBytes(file, run(["show", ":" + file], { encoding: null }).stdout);
  } else {
    // Export exactly tracked + newly staged source, excluding generated/dependency files.
    const snapshot = path.join(temp, "tracked");
    fs.mkdirSync(snapshot);
    for (const file of tracked) {
      const source = path.join(root, file);
      if (!fs.existsSync(source) || !fs.statSync(source).isFile()) continue;
      const target = path.join(snapshot, file);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      const bytes = fs.readFileSync(source);
      checkBytes(file, bytes);
      fs.writeFileSync(target, bytes);
    }
    scan(["dir", snapshot], "tracked");
  }
  if (mode !== "staged")
    scan(["git", "--log-opts=--all", root], "history", true);
  if (mode === "all" || mode === "bundle") {
    const build = path.join(root, "build");
    if (!fs.existsSync(build))
      throw Error(
        "Run the production build before the complete security check.",
      );
    scan(["dir", build], "bundle");
    const visit = (directory) => {
      for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const file = path.join(directory, entry.name);
        if (entry.isDirectory()) visit(file);
        else if (/\.(?:js|json|html|css|map)$/.test(file)) {
          const bytes = fs.readFileSync(file);
          checkBytes(path.relative(root, file), bytes);
          const text = bytes.toString("utf8");
          if (
            /(?:gsk_[a-zA-Z0-9]{48,}|sk-or-v1-[a-f\d]{48,}|-----BEGIN (?:RSA )?PRIVATE KEY-----)/.test(
              text,
            )
          )
            errors.push({
              check: "bundle-private-key-pattern",
              file: path.relative(root, file),
            });
        }
      }
    };
    visit(build);
  }
  if (errors.length) {
    console.error(
      JSON.stringify({ security: "BLOCKED", findings: errors }, null, 2),
    );
    process.exitCode = 1;
  } else
    console.log(
      "PASS: no private credentials detected. No secret values printed.",
    );
} catch (error) {
  console.error("SECURITY CHECK FAILED: " + error.message);
  process.exitCode = 1;
} finally {
  fs.rmSync(temp, { recursive: true, force: true });
}
