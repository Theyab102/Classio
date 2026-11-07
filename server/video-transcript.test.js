const { test } = require("node:test");
const assert = require("node:assert/strict");
const { playerResponse, captionText, publicTranscript } = require("./video-transcript");

test("parses a player JSON value without evaluating webpage scripts", () => {
  const value = { videoDetails: { title: 'A title with } and "quoted" text' }, captions: { x: true } };
  assert.deepEqual(playerResponse(`prefix; var ytInitialPlayerResponse = ${JSON.stringify(value)}; window.doNotExecute();`), value);
  assert.equal(playerResponse("unavailable"), null);
  assert.equal(captionText(JSON.stringify({ events: [{ segs: [{ utf8: "A force" }, { utf8: "changes motion." }] }] })), "A force changes motion.");
  assert.equal(captionText('<transcript><text start="0">Force &amp; motion</text><text>Newton</text></transcript>'), "Force & motion Newton");
});
test("retrieves actual caption text and never follows an arbitrary caption host", async () => {
  let calls = 0;
  const fixture = host => JSON.stringify({ videoDetails: { title: "Forces" }, captions: { playerCaptionsTracklistRenderer: { captionTracks: [{ languageCode: "en", baseUrl: `https://${host}/api/timedtext?v=abcdefghijk` }] } } });
  const result = await publicTranscript("abcdefghijk", async url => {
    calls++;
    if (url.includes("/watch")) return { ok: true, text: async () => `var ytInitialPlayerResponse = ${fixture("www.youtube.com")};` };
    assert.match(url, /www\.youtube\.com\/api\/timedtext/);
    return { ok: true, text: async () => JSON.stringify({ events: [{ segs: [{ utf8: "Force changes motion." }] }] }) };
  });
  assert.equal(result.transcript, "Force changes motion."); assert.equal(calls, 2);
  calls = 0;
  await assert.rejects(publicTranscript("abcdefghijk", async () => { calls++; return { ok: true, text: async () => `var ytInitialPlayerResponse = ${fixture("example.invalid")};` }; }));
  assert.equal(calls, 1);
});
test("missing captions are reported as unavailable rather than converted into title-based notes", async () => {
  await assert.rejects(publicTranscript("abcdefghijk", async () => ({ ok: true, text: async () => 'var ytInitialPlayerResponse = {"videoDetails":{"title":"Forces"}};' })), /TRANSCRIPT_UNAVAILABLE/);
});
