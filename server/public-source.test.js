const { test } = require("node:test");
const assert = require("node:assert/strict");
const { publicIp, publicUrl, publicAddresses, extractSource } = require("./public-source");
test("public source reading rejects local addresses, mixed DNS answers and credential URLs", async () => {
  for (const ip of ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "::1", "::ffff:127.0.0.1", "fc00::1"]) assert.equal(publicIp(ip), false);
  assert.equal(publicIp("93.184.216.34"), true);
  for (const url of ["file:///etc/passwd", "http://localhost/page", "https://user:pass@example.com", "http://127.0.0.1", "http://[::1]", "https://example.com:3000"]) assert.throws(() => publicUrl(url));
  await assert.rejects(publicAddresses(new URL("https://example.com"), async () => [{ address: "93.184.216.34", family: 4 }, { address: "10.0.0.1", family: 4 }]));
});
test("article extraction keeps real tables and source figures without page scripts or navigation", () => {
  const html = '<html><title>Motion</title><nav>Account menus</nav><article><h1>Projectile motion</h1><p>A projectile moves under gravity. Horizontal velocity stays constant when drag can be neglected. This is an educational explanation with a worked example.</p><figure><img src="/figures/motion.png" alt="Horizontal and vertical velocity" width="700" height="400"></figure><img src="http://127.0.0.1/private.png"><table><tr><th>Quantity</th><th>Unit</th></tr><tr><td>Speed</td><td>m/s</td></tr></table><script>privateTracking()</script></article></html>';
  const result = extractSource(html, "https://example.com/physics");
  assert.equal(result.images.length, 1); assert.equal(result.images[0].src, "https://example.com/figures/motion.png");
  assert.deepEqual(result.tables[0][1], ["Speed", "m/s"]); assert(!/privateTracking|Account menus/.test(result.text));
});
