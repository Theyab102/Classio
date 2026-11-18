const { test } = require("node:test"),
  assert = require("node:assert/strict"),
  { spawnSync } = require("node:child_process");
const pm = require("picomatch");
test("the reported single-branch extglob attack patterns cannot block the event loop", () => {
  const runner = `const p=require(${JSON.stringify(require.resolve("picomatch"))});for(const pattern of ['+(*)','+(*a)','+(a*)','*(*?)','+([ab]*)','+(@(a|aa))'])for(const options of [{},{maxExtglobRecursion:false}])if(p.isMatch('a'.repeat(80)+'!',pattern,options))throw Error('Unexpected match');`;
  const child = spawnSync(process.execPath, ["-e", runner], {
    timeout: 1500,
    encoding: "utf8",
  });
  assert.equal(
    child.error,
    undefined,
    "Attack regression exceeded the isolated time limit",
  );
  assert.equal(child.status, 0, child.stderr);
});
test("the mitigation preserves standard globs, literal repeats, exclusions and the actual test-discovery pattern", () => {
  assert(pm.isMatch("src/App.test.js", "**/?(*.)+(spec|test).[jt]s?(x)"));
  assert(pm.isMatch("foo/bar.js", "**/*.js"));
  assert(pm.isMatch("foo.js", "*.js"));
  assert(pm.isMatch("aaaa", "+(a)"));
  assert(pm.isMatch("foofoo", "+(foo)"));
  assert(!pm.isMatch("src/private.js", "**/*.test.js"));
  assert(pm.isMatch("a/b/node_modules/example/index.js", "**/node_modules/**"));
});
