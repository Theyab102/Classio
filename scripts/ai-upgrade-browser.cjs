const {frontendAsset}=require('./browser-assets.cjs');
// Real browser UI, storage, refresh and worker loading; hosted responses are fixtures.
const http = require("node:http"),
  fs = require("node:fs"),
  path = require("node:path"),
  assert = require("node:assert/strict"),
  os = require("node:os");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, "../build"),
  output = fs.mkdtempSync(path.join(os.tmpdir(), "classio-ai-upgrade-"));
const csp = require("../vercel.json").headers[0].headers.find(
  (row) => row.key === "Content-Security-Policy",
).value;
const server = http.createServer((req, res) => {
  try {
    const name = new URL(req.url, "http://local").pathname,
      file = frontendAsset(root,name);
    if (!file.startsWith(root + path.sep)) throw Error();
    res.setHeader("Content-Security-Policy", csp);
    res.setHeader(
      "Content-Type",
      {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".wasm": "application/wasm",
        ".html": "text/html",
        ".css": "text/css",
        ".json": "application/json",
        ".woff2": "font/woff2",
      }[path.extname(file)] || "application/octet-stream",
    );
    res.end(fs.readFileSync(file));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = "http://127.0.0.1:" + server.address().port,
    browser = await chromium.launch({
      executablePath: "/usr/bin/chromium",
      headless: true,
      args: [
        "--no-sandbox",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
      ],
    });
  let active;
  try {
    for (const size of [
      { name: "desktop", width: 1440, height: 1000 },
      { name: "tablet-landscape", width: 1024, height: 768 },
      { name: "tablet-portrait", width: 768, height: 1024 },
      { name: "phone", width: 390, height: 844 },
    ]) {
      if (
        process.env.CLASSIO_AI_TARGET &&
        process.env.CLASSIO_AI_TARGET !== size.name
      )
        continue;
      const context = await browser.newContext({
          viewport: size,
          hasTouch: size.name !== "desktop",
        }),
        page = await context.newPage();
      active = page;
      page.setDefaultTimeout(20000);
      const errors = [],
        slides = [],
        models = [],
        calls = [];
      let broken = true,waitingSetup=false;
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("request", (req) => {
        if (/huggingface|hf\.co|mlc\.ai/.test(req.url()))
          models.push(req.url());
      });
      await context.addInitScript(() =>
        localStorage.setItem(
          "classio_access_v2",
          JSON.stringify({ decision: "accepted", at: Date.now() }),
        ),
      );
      await context.route("**/api/status", (route) =>
        route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            aiConfigured: true,
            providers: [
              { id: "groq", status: "available" },
              { id: "cloudflare", status: "available" },
              { id: "openrouter", status: "temporarily-limited" },
            ],
            freeOnly: true,
            browserLocalEnabled: true,
          }),
        }),
      );
      await context.route("**/api/session", (route) =>
        route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            csrf: "fixture-csrf",
            expiresAt: Date.now() + 3600000,
          }),
        }),
      );
      await context.route("**/api/images", (route) =>
        route.fulfill({
          contentType: "application/json",
          body: '{"backgroundRemoval":false,"imageEditing":false,"searchProviders":[]}',
        }),
      );
      await context.route("**/api/study", (route) => {
        const data = route.request().postDataJSON();
        if(waitingSetup&&data.task==="presentation")return route.fulfill({status:503,contentType:"application/json",body:JSON.stringify({code:"AI_NOT_CONFIGURED",error:"AI tools are waiting for the site’s server setup. Your work is kept."})});
        calls.push(data);
        assert.equal(
          route.request().headers()["x-classio-csrf"],
          "fixture-csrf",
        );
        let text = "Physics";
        const prompt = data.messages?.[0]?.content || "";
        if (prompt.startsWith("Plan exactly 20"))
          text = JSON.stringify({
            title: "Forces",
            designs: [],
            outline: Array.from({ length: 20 }, (_, i) => ({
              title: "Concept " + (i + 1),
              focus: "Focus " + (i + 1),
              type: "content",
            })),
          });
        else if (prompt.startsWith("Write ONLY slide")) {
          const n = Number(/Write ONLY slide (\d+)/.exec(prompt)[1]);
          slides.push(n);
          if (n === 15 && broken)
            return route.fulfill({
              status: 503,
              contentType: "application/json",
              body: '{"code":"PROVIDER_UNAVAILABLE","error":"The free providers are temporarily unavailable. Saved progress is kept."}',
            });
          text = JSON.stringify({
            title: "Forces",
            slides: [
              {
                type: "content",
                title: "Concept " + n,
                bullets: ["Distinct source-based explanation " + n],
              },
            ],
          });
        } else if (prompt.includes("short-answer study questions"))
          text = JSON.stringify(
            [1, 2].map((i) => ({
              type: "short-answer",
              question: "Question " + i + ": SI unit of force?",
              answer: "Newton",
              acceptedAnswers: ["N", "newtons"],
              hint: "Use the SI name",
              explanation: "Force is measured in newtons.",
            })),
          );
        return route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ text }),
        });
      });
      const menu = async () => {
        const button = page.getByRole("button", {
          name: "Open menu",
          exact: true,
        });
        if (await button.isVisible()) await button.click();
      };
      const main = async (name) => {
        await menu();
        await page.getByRole("link", { name, exact: true }).click();
      };
      const nav = async (name) => {
        await menu();
        await page
          .getByRole("navigation", { name: "Study tools" })
          .getByRole("link", { name, exact: true })
          .click();
      };
      const choose = async (label, value) => {
        await page.getByRole("combobox", { name: label, exact: true }).click();
        await page.getByRole("option", { name: value, exact: true }).click();
      };
      const shot = async (name) => {
        assert(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
          "Page overflow at " + size.name + " " + name,
        );
        await page.screenshot({
          path: path.join(output, size.name + "-" + name + ".png"),
          fullPage: true,
          animations: "disabled",
        });
      };
      await page.goto(origin);
      await page
        .getByRole("heading", { name: "Library", exact: true })
        .waitFor();
      await main("Settings");
      await page
        .getByRole("combobox", { name: "AI mode", exact: true })
        .waitFor();
      await page
        .getByRole("combobox", { name: "AI mode", exact: true })
        .click();
      const options = page.getByRole("listbox");
      assert(
        !(await options
          .getByRole("option", { name: /Gemini|Ollama|Private server/ })
          .count()),
      );
      await page.getByRole("option", { name: "Groq", exact: true }).click();
      assert.equal(
        await page.evaluate(() => localStorage.getItem("classio_ai_mode")),
        "groq",
      );
      const settings = page.locator(".classio-ai-settings");
      assert(
        !/Groq_Key|Gemini_Key|packup|API_KEY|CLASSIO_SESSION_SECRET|primary credential/i.test(
          await settings.innerText(),
        ),
      );
      await settings
        .getByText("On-device AI and semantic search", { exact: true })
        .click();
      await page.waitForTimeout(100);
      assert.equal(models.length, 0, "A model downloaded without consent");
      await shot("settings-light");
      await page.getByRole("switch", { name: "Dark mode" }).click();
      await page.waitForFunction(
        () => document.documentElement.dataset.theme === "dark",
      );
      await shot("settings-dark");
      await page.getByRole("switch", { name: "Dark mode" }).click();
      await page.waitForFunction(
        () => document.documentElement.dataset.theme === "light",
      );
      if (size.name === "desktop") {
        await page.evaluate(async()=>{for(const name of ['webllm/model','transformers-cache']){const cache=await caches.open(name);await cache.put('/synthetic-model-cache-check',new Response('synthetic public model bytes'));}});
        const removeModels=settings.getByRole('button',{name:'Remove downloaded models',exact:true});
        await removeModels.click();
        await page.waitForFunction(async()=>{const names=await caches.keys();return !names.includes('webllm/model')&&!names.includes('transformers-cache');});
        await page.waitForFunction(()=>!Array.from(document.querySelectorAll('.classio-ai-settings button')).find(button=>button.textContent==='Remove downloaded models')?.disabled);
        assert.equal(await settings.getByRole('alert').count(),0,'Fresh-worker model cache cleanup failed');
        assert.equal(models.length,0,'Cache removal downloaded model weights');
        await context.route("https://huggingface.co/**", (route) =>
          route.abort(),
        );
        await context.route("https://cdn-lfs.huggingface.co/**", (route) =>
          route.abort(),
        );
        await settings
          .getByRole("checkbox", {
            name: "Allow model downloads on this device",
          })
          .check();
        await settings
          .getByRole("button", { name: "Enable semantic search", exact: true })
          .click();
        await settings.getByRole("alert").waitFor({ timeout: 30000 });
        assert(
          (await settings.getByRole("alert").innerText()).includes(
            "could not load",
          ),
          "Download failure must be handled visibly",
        );
        await settings
          .getByRole("button", { name: "Disable local AI", exact: true })
          .click();
      }
      await page
        .getByRole("button", { name: "Back to library", exact: true })
        .click();
      await page
        .getByRole("button", { name: "Blank document", exact: true })
        .click();
      const close = page.getByRole("button", {
        name: "Close AI Assistant",
        exact: true,
      });
      if (await close.count()) await close.click();
      await page
        .getByRole("textbox", { name: "Study notes", exact: true })
        .fill("Forces change motion. The SI unit of force is the newton.");
      await nav("Quizzes");
      await page
        .getByRole("spinbutton", { name: "Number of questions" })
        .fill("2");
      await choose("Question type", "Short answer");
      await page
        .getByRole("button", { name: "Generate quiz", exact: true })
        .click();
      await page
        .getByRole("textbox", { name: "Your answer", exact: true })
        .fill("Newton");
      await page
        .getByRole("button", { name: "Check answer", exact: true })
        .click();
      await page.getByText("Correct.", { exact: true }).waitFor();
      assert.equal(calls.at(-1).provider, "groq");
      await shot("short-answer");
      await nav("Presentations");waitingSetup=true;
      await page.getByRole("button",{name:"Generate",exact:true}).click();
      await page.getByText("AI tools are waiting for the site’s server setup. Your work is kept.",{exact:true}).waitFor();
      await page.getByText("Generation hasn’t started yet.",{exact:true}).waitFor();
      assert.equal(await page.getByText("Generation progress saved",{exact:true}).count(),0,"Empty attempt claimed progress was generated");
      assert.equal(await page.getByText(/0 of 1 sections saved/).count(),0);
      await shot("empty-attempt-fixed");waitingSetup=false;

      if (size.name === "desktop") {
        await nav("Presentations");
        await page.getByLabel("Number of slides").fill("20");
        await page
          .getByRole("button", { name: "Generate", exact: true })
          .click();
        await page
          .getByText(
            "The free providers are temporarily unavailable. Saved progress is kept.",
            { exact: true },
          )
          .waitFor();
        await page.getByText(/14 of 20 sections saved/).waitFor();
        const encrypted = await page.evaluate(
          () =>
            new Promise((resolve, reject) => {
              const req = indexedDB.open("classio_guest_files", 1);
              req.onerror = () => reject(req.error);
              req.onsuccess = () => {
                const tx = req.result.transaction("files", "readonly"),
                  store = tx.objectStore("files"),
                  keys = store.getAllKeys(),
                  rows = store.getAll();
                tx.oncomplete = () => {
                  const i = keys.result.findIndex(
                    (key) =>
                      String(key).includes("generation-job:") &&
                      String(key).endsWith(":presentation"),
                  );
                  resolve(
                    i >= 0 &&
                      rows.result[i].data instanceof ArrayBuffer &&
                      !JSON.stringify(rows.result[i]).includes(
                        "Distinct source-based",
                      ),
                  );
                  req.result.close();
                };
              };
            }),
        );
        assert(
          encrypted,
          "Guest recovery must be persisted as encrypted bytes",
        );
        broken = false;
        slides.length = 0;
        const recoveryUrl = page.url();
        await page.reload();
        assert.equal(page.url(), recoveryUrl, "Refresh changed the presentation URL");
        await nav("Presentations");
        await page
          .getByRole("button", { name: "Resume generation", exact: true })
          .click();
        await page.locator(".deck-slide-card").nth(19).waitFor();
        assert.deepEqual(slides, [15, 16, 17, 18, 19, 20]);
        assert.equal(await page.locator(".deck-slide-card").count(), 20);
        await page
          .getByRole("button", { name: "Resume generation", exact: true })
          .waitFor({ state: "detached" });
        await shot("recovered-20-slides");
        await page.reload();
        assert.equal(page.url(), recoveryUrl, "Completed deck URL was lost on refresh");
        await nav("Presentations");
        assert.equal(
          await page.locator(".deck-slide-card").count(),
          20,
          "Completed deck did not survive refresh",
        );
        assert.equal(
          await page
            .getByRole("button", { name: "Resume generation", exact: true })
            .count(),
          0,
        );
      }
      assert.deepEqual(errors, [], "Uncaught runtime errors");
      console.log(
        "PASS " +
          size.name +
          ": safe provider choices, optional local-model loading, light/dark settings, short-answer quiz" +
          (size.name === "desktop"
            ? ", encrypted guest checkpoints and a real refresh/resume of slides 15–20."
            : "."),
      );
      await context.close();
    }
    console.log("Artifacts: " + output);
  } catch (error) {
    if (active)
      await active
        .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
        .catch(() => {});
    console.error("Artifacts: " + output);
    throw error;
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
