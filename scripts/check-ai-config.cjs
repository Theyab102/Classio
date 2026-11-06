const { configurationStatus } = require("../server/ai-configuration");

const report = configurationStatus();
console.log("AI provider settings:", report.providers.join(", ") || "none");
if (!report.configured) {
  console.log("Missing or invalid server settings:");
  for (const name of report.missing) console.log(`- ${name}`);
  console.log("Set these securely in the Vercel project's Production environment, then redeploy.");
  process.exitCode = 1;
} else console.log("Required settings are present. A live request is still needed to verify permissions and model availability.");
