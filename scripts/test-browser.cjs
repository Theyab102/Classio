// End-to-end UI regression checks. AI and speech transports use explicit fixtures;
// these checks do not establish live Gemini, Groq or speech-service availability.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const PptxGenJS = require('pptxgenjs');
const { parseStudyRequest } = require('../server/study-request.js');

const root = path.resolve(__dirname, '../build');
const output = process.env.CLASSIO_BROWSER_OUTPUT || fs.mkdtempSync(path.join(os.tmpdir(), 'classio-browser-'));
fs.mkdirSync(output, { recursive: true });
const notes = '# Forces and motion\n\nA force changes motion.\n\n## Key relationship\n\n$F = ma$\n\n| Quantity | Unit |\n| --- | --- |\n| Force | Newton |\n\n- Balanced forces sum to zero.\n- Acceleration depends on mass.';
const units = Array.from({ length: 3 }, (_, i) => ({ title: `Motion lesson ${i + 1}`, lesson: '## Understanding force\n\nForces change motion.\n\n### Worked example\n\nA two newton force on a one kilogram mass produces $a = 2$ metres per second squared.\n\n**Takeaway:** use $F=ma$.', question: `What is the unit of force in lesson ${i + 1}?`, options: ['Newton', 'Joule', 'Watt'], correct: 0, explanation: 'Force is measured in newtons.' }));
const questions = Array.from({ length: 5 }, (_, i) => ({ question: `Question ${i + 1}: which unit measures force?`, options: ['Newton', 'Joule', 'Watt', 'Metre'], correct: 0, hint: 'Think of the scientist associated with laws of motion.', explanation: 'The newton is the SI unit of force, $1 N = 1 kg m/s^2$.' }));
const cards = Array.from({ length: 8 }, (_, i) => ({ question: `Force concept ${i + 1}`, answer: `Newton: explanation ${i + 1}.` }));
const episode = { title: 'Forces in everyday life', turns: Array.from({ length: 6 }, (_, i) => ({ speaker: i % 2 ? 'B' : 'A', section: `Force section ${Math.floor(i / 2) + 1}`, text: `Force explanation ${i + 1}. A force changes motion, and balanced forces add to zero.` })) };
function wave() {
  const n = 22050 * 4, bytes = Buffer.alloc(44 + n * 2);
  bytes.write('RIFF', 0); bytes.writeUInt32LE(36 + n * 2, 4); bytes.write('WAVEfmt ', 8); bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20); bytes.writeUInt16LE(1, 22); bytes.writeUInt32LE(22050, 24); bytes.writeUInt32LE(44100, 28); bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34); bytes.write('data', 36); bytes.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) bytes.writeInt16LE(Math.round(Math.sin(i / 22050 * 2 * Math.PI * 220) * 1000), 44 + i * 2);
  return bytes;
}
function pdf() {
  const objects = ['<< /Type /Catalog /Pages 2 0 R >>', '<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>'];
  for (const [index, label] of ['Forces and motion', 'Balanced forces are zero'].entries()) {
    const stream = `BT /F1 24 Tf 60 700 Td (${label}) Tj ET`;
    objects.push('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents ' + (4 + index * 2) + ' 0 R >>', `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  }
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  let result = '%PDF-1.4\n', offsets = [0];
  objects.forEach((object, i) => { offsets.push(Buffer.byteLength(result)); result += `${i + 1} 0 obj\n${object}\nendobj\n`; });
  const xref = Buffer.byteLength(result);
  result += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n` + offsets.slice(1).map(value => `${String(value).padStart(10, '0')} 00000 n \n`).join('') + `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
  return Buffer.from(result);
}
const mime = { '.js': 'application/javascript', '.mjs': 'application/javascript', '.css': 'text/css', '.html': 'text/html', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const target = path.join(root, ['/', '/privacy', '/terms', '/copyright'].includes(pathname) ? 'index.html' : decodeURIComponent(pathname));
  if (!target.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  try { res.setHeader('Content-Type', mime[path.extname(target)] || 'application/octet-stream'); res.end(fs.readFileSync(target)); }
  catch { res.writeHead(404); res.end('Not found'); }
});

(async () => {
  assert(fs.existsSync(path.join(root, 'index.html')), 'Run npm run build first.');
  const deck = new PptxGenJS(); deck.layout = 'LAYOUT_WIDE';
  ['Forces in motion', 'Balanced forces'].forEach(text => { const slide = deck.addSlide(); slide.addText(text, { x: 1, y: 1, w: 10, h: 1, fontSize: 28 }); });
  const pptx = Buffer.from(await deck.write({ outputType: 'nodebuffer' }));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const executablePath = process.env.CLASSIO_CHROMIUM_PATH || (fs.existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
  const browser = await chromium.launch({ executablePath, headless: true, args: ['--no-sandbox'] });
  let active;
  try {
    for (const size of [{ name: 'desktop', width: 1440, height: 1000 }, { name: 'phone', width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport: size, hasTouch: size.width < 600, colorScheme: 'light' });
      const page = await context.newPage(); active = page; page.setDefaultTimeout(15000);
      const errors = [], calls = []; let failAI = false, failSpeech = false;
      page.on('pageerror', error => errors.push(error.message));
      page.on('dialog', dialog => dialog.accept());
      await page.route('**/api/study', async route => {
        const data = route.request().postDataJSON();
        assert(parseStudyRequest(data), 'Browser sent an invalid AI request');
        calls.push(data); const prompt = data.messages[0].content;
        let text = 'Force is measured in newtons.';
        if (prompt.includes('self-paced course')) text = JSON.stringify(units);
        else if (prompt.includes('multiple-choice')) text = JSON.stringify(questions);
        else if (prompt.includes('flashcards') && prompt.includes('JSON array')) text = JSON.stringify(cards);
        else if (prompt.includes('two-host educational podcast')) text = JSON.stringify(episode);
        else if (prompt.includes('slide') && prompt.includes('JSON')) text = JSON.stringify({ title: 'Forces', slides: [{ type: 'title', title: 'Forces' }, { type: 'table', title: 'Units', table: [['Quantity', 'Unit'], ['Force', 'Newton']] }] });
        else if (prompt.toLowerCase().includes('study notes')) text = notes;
        await route.fulfill({ status: failAI ? 503 : 200, contentType: 'application/json', body: JSON.stringify(failAI ? { error: 'The AI service is temporarily unavailable.', code: 'PROVIDER_UNAVAILABLE' } : { text }) });
      });
      await page.route('**/api/transcript?*', route => route.fulfill({ status: 422, contentType: 'application/json', body: JSON.stringify({ error: 'A video transcript is unavailable.', manualAvailable: true }) }));
      await page.route('**/*.hf.space/voices', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify([{ id: 'aria', label: 'Aria', gender: 'female' }, { id: 'echo', label: 'Echo', gender: 'male' }]) }));
      await page.route('**/*.hf.space/generate-podcast', route => route.fulfill(failSpeech ? { status: 503, contentType: 'application/json', body: '{"error":"Voice service unavailable"}' } : { contentType: 'audio/wav', body: wave() }));
      await page.route('**/transcript/*', route => route.fulfill({ contentType: 'application/json', body: '{"success":false}' }));
      await page.route('**/youtubetranscript.com/**', route => route.fulfill({ contentType: 'text/xml', body: '<transcript/>' }));
      await page.route('**://*.youtube.com/oembed*', route => route.fulfill({ contentType: 'application/json', body: '{"title":"Forces"}' }));
      async function overflow(label) { assert((await page.evaluate(() => document.documentElement.scrollWidth)) <= size.width + 1, `${size.name}: ${label} overflows`); }
      async function shot(label) { await overflow(label); await page.screenshot({ path: path.join(output, `${size.name}-${label}.png`), fullPage: true }); }
      async function menu() { if (size.width < 600) await page.getByRole('button', { name: 'Open menu', exact: true }).click(); }
      async function nav(label, main = false) { await menu(); await page.getByRole('navigation', { name: main ? 'Main menu' : 'Study tools', exact: true }).getByRole('button', { name: label, exact: true }).click(); }
      async function library() { await page.getByRole('button', { name: 'Back to library', exact: true }).click(); await page.getByRole('button', { name: 'Create new', exact: true }).waitFor(); }
      async function appearance(value) {
        await menu(); const select = page.getByLabel('Appearance', { exact: true });
        if (!(await select.count())) await page.getByRole('button', { name: 'Expand sidebar', exact: true }).click();
        await page.getByLabel('Appearance', { exact: true }).selectOption(value);
        if (size.width < 600) await page.getByRole('button', { name: 'Close menu', exact: true }).click();
      }
      await page.goto(origin);
      await page.getByRole('button', { name: 'Get started' }).click(); await page.getByLabel('Date of birth').fill('2000-01-01'); await page.getByRole('button', { name: 'Continue', exact: true }).click(); await page.getByRole('checkbox').check(); await page.getByRole('button', { name: 'Open workspace', exact: true }).click();
      await page.getByRole('button', { name: 'Create new', exact: true }).waitFor();
      for (const label of ['Blank document', 'Record audio', 'Upload document', 'Website link']) assert.equal(await page.getByRole('button', { name: label, exact: true }).count(), 1);
      assert(!(await page.locator('body').innerText()).includes('Opening your workspace'));
      await shot('library-light'); await appearance('dark'); await shot('library-dark'); await appearance('light');

      await page.getByRole('button', { name: 'Blank document', exact: true }).click();
      const editor = page.getByRole('textbox', { name: 'Study notes', exact: true }); await editor.fill('Forces change the motion of objects.'); await editor.press('Control+a'); await page.getByRole('button', { name: 'Bold', exact: true }).click(); await page.getByLabel('Font size', { exact: true }).selectOption('20px'); assert.match(await editor.innerHTML(), /<strong>/);
      await editor.press('End'); await page.getByRole('button', { name: 'Insert equation', exact: true }).click(); await page.getByLabel('Equation LaTeX', { exact: true }).fill('F=ma'); await page.locator('.notes-inspector').getByRole('button', { name: 'Insert equation', exact: true }).click(); assert.equal(await editor.locator('.note-equation').count(), 1);
      await page.getByRole('button', { name: 'Insert table', exact: true }).click(); assert.equal(await editor.locator('table tr').count(), 3); await page.getByRole('button', { name: 'Delete table', exact: true }).click();
      await page.getByRole('button', { name: 'Generate', exact: true }).click(); const gen = page.locator('.notes-inspector'); await gen.getByLabel('Topic', { exact: true }).fill('Forces'); failAI = true; await gen.getByRole('button', { name: 'Generate notes', exact: true }).click(); await gen.getByRole('alert').waitFor(); failAI = false; await gen.getByRole('button', { name: 'Generate notes', exact: true }).click(); await gen.getByRole('region', { name: 'AI draft' }).waitFor(); assert.match(await editor.innerText(), /Forces change/); await gen.getByRole('button', { name: 'Replace notes', exact: true }).click(); assert.equal(await editor.locator('table').count(), 1); assert.equal(await editor.locator('.note-equation').count(), 1);
      await page.getByRole('button', { name: 'Note options', exact: true }).click(); await page.getByRole('menuitem', { name: 'Saved versions', exact: true }).click(); await gen.getByLabel('Version name', { exact: true }).fill('Before review'); await gen.getByRole('button', { name: 'Save version', exact: true }).click(); await page.getByRole('button', { name: 'Close note tools', exact: true }).click(); await shot('notes-light');

      console.log(size.name + ': notes and drafts passed');
      // The visible composer opens the shared assistant with the actual material.
      await page.getByPlaceholder('Ask Classio anything...', { exact: true }).fill('Explain force'); await page.locator('.inline-assistant').getByRole('button', { name: /^Ask about / }).click(); const chat = page.locator('.assistant-panel'); await chat.getByLabel('Message', { exact: true }).fill('What unit measures force?'); await chat.getByRole('button', { name: 'Send', exact: true }).click(); await chat.getByText('Force is measured in newtons.', { exact: true }).waitFor(); assert.match(calls.at(-1).messages[0].content, /Forces and motion/);
      const imageBytes = await page.screenshot({ clip: { x: 0, y: 0, width: 80, height: 80 } }); await chat.locator('input[type=file]').setInputFiles({ name: 'study-diagram.png', mimeType: 'image/png', buffer: imageBytes }); await chat.locator('.attachment-preview').waitFor(); await chat.getByLabel('Message', { exact: true }).fill('Explain the attached study diagram'); await chat.getByRole('button', { name: 'Send', exact: true }).click(); await chat.getByText('Force is measured in newtons.', { exact: true }).nth(1).waitFor(); assert.equal(parseStudyRequest(calls.at(-1)).vision, true);
      await shot('assistant-light'); if (size.width >= 1200) { const box = await chat.boundingBox(), main = await page.locator('.notes-workspace').boundingBox(); assert(main.x + main.width <= box.x + 1, 'Assistant covers the notes'); }
      await page.getByRole('button', { name: 'Close AI Assistant', exact: true }).click();

      await nav('Lessons'); await page.getByRole('button', { name: 'Generate lessons', exact: true }).click(); await page.getByRole('button', { name: 'Add lessons to notes', exact: true }).waitFor(); await page.locator('.lesson-check .answer-choices button').first().click(); await page.getByText('1 of 3 lessons completed', { exact: true }).waitFor(); await page.getByRole('button', { name: 'Continue', exact: true }).click(); await shot('lessons-light');
      console.log(size.name + ': lessons passed');
      await nav('Quizzes'); await page.getByRole('button', { name: 'Generate quiz', exact: true }).click(); await page.getByRole('button', { name: 'Show hint', exact: true }).click(); await page.getByText('Think of the scientist associated with laws of motion.', { exact: true }).waitFor();
      for (let i = 0; i < 5; i++) { await page.locator('.answer-choices button').first().click(); await page.getByRole('button', { name: 'Check answer', exact: true }).click(); if (i < 4) await page.getByRole('button', { name: 'Next', exact: true }).click(); }
      await page.getByRole('button', { name: 'See results', exact: true }).click(); await page.getByRole('heading', { name: '5 of 5 correct', exact: true }).waitFor(); await page.getByRole('button', { name: 'Review answers', exact: true }).click(); await shot('quiz-light');
      console.log(size.name + ': quiz passed');
      await nav('Flashcards'); await page.getByRole('button', { name: 'Generate cards', exact: true }).click(); await page.getByRole('dialog', { name: 'Generate flashcards', exact: true }).getByRole('button', { name: 'Generate cards', exact: true }).click(); await page.getByRole('button', { name: 'Start studying', exact: true }).waitFor(); await page.getByRole('button', { name: 'Edit current flashcard', exact: true }).click(); await page.getByRole('dialog', { name: 'Edit flashcard', exact: true }).locator('textarea').nth(1).fill('Newton, the unit of force.'); await page.getByRole('button', { name: 'Save card', exact: true }).click(); await page.getByRole('button', { name: 'Start studying', exact: true }).click(); await page.getByRole('button', { name: 'Flashcard question. Flip to answer', exact: true }).click(); await page.locator('.flashcard-grades button').nth(1).click(); await page.getByRole('button', { name: '1 Learning', exact: true }).waitFor(); await shot('flashcards-light');
      console.log(size.name + ': flashcards passed');
      await nav('Podcasts'); failSpeech = true; await page.getByRole('button', { name: 'Create podcast', exact: true }).click(); await page.getByRole('alert').waitFor(); assert.equal(await page.getByRole('button', { name: 'Play podcast', exact: true }).count(), 0, 'A transcript must not produce a fake player'); const scriptRequests = calls.length; failSpeech = false; await page.getByRole('button', { name: 'Prepare audio', exact: true }).click(); await page.getByRole('button', { name: 'Play podcast', exact: true }).waitFor({ timeout: 30000 }); assert.equal(calls.length, scriptRequests, 'Audio retry must preserve the generated transcript'); await page.waitForFunction(() => document.querySelector('.podcast-player audio')?.duration > 20); await page.getByRole('button', { name: 'Play podcast', exact: true }).click(); await page.getByRole('button', { name: 'Pause podcast', exact: true }).click(); await page.getByRole('slider', { name: 'Podcast playback position', exact: true }).fill('8'); await page.getByRole('combobox', { name: 'Playback speed', exact: true }).selectOption('1.5'); await page.getByRole('slider', { name: 'Podcast volume', exact: true }).fill('0.5'); await page.getByRole('button', { name: 'Full transcript', exact: true }).click(); assert.equal(await page.locator('.podcast-turn').count(), 6); await shot('podcast-light');
      await nav('Notes'); await nav('Podcasts'); await page.waitForFunction(() => document.querySelector('.podcast-player audio')?.currentTime >= 7.9); await appearance('dark'); await shot('podcast-dark');
      await nav('Presentations'); await page.locator('details.deck-options summary').click(); await page.getByLabel('Presentation topic').fill('Forces'); await page.getByLabel('Number of slides').fill('2'); await page.getByRole('button', { name: 'Generate', exact: true }).click(); await page.getByRole('region', { name: 'Presentation design options' }).waitFor(); await page.getByRole('button', { name: 'Use Lecture', exact: true }).click(); assert.equal(await page.locator('.deck-canvas table').count(), 1); await overflow('presentations'); await nav('Study games'); await overflow('study games');
      await nav('Notes'); await appearance('dark'); await shot('notes-dark'); await nav('Lessons'); await page.getByText('1 of 3 lessons completed', { exact: true }).waitFor(); await shot('lessons-dark'); await nav('Quizzes'); await shot('quiz-dark'); await nav('Flashcards'); await shot('flashcards-dark');
      await nav('AI Assistant', true); await chat.getByText('What unit measures force?', { exact: true }).waitFor(); await shot('assistant-dark'); await page.getByRole('button', { name: 'Close AI Assistant', exact: true }).click();

      await menu(); await page.getByRole('button', { name: 'Edit character', exact: true }).click(); const avatar = page.getByRole('dialog', { name: 'Your Classio character', exact: true }); await avatar.getByLabel('Display name', { exact: true }).fill('Sam'); await avatar.getByRole('button', { name: 'Hair', exact: true }).click(); await avatar.getByRole('button', { name: 'Hairstyle: Two buns', exact: true }).click(); assert.equal(await avatar.locator('.avatar-stage [data-hair-style="13"]').count(), 1); await shot('avatar-dark'); await avatar.getByRole('button', { name: 'Save changes', exact: true }).click();
      await menu(); await page.getByRole('button', { name: 'Edit character', exact: true }).click(); await avatar.getByLabel('Display name', { exact: true }).fill('Discarded name'); await avatar.getByRole('button', { name: 'Cancel', exact: true }).click(); await menu(); await page.getByRole('button', { name: 'Edit character', exact: true }).click(); assert.equal(await avatar.getByLabel('Display name', { exact: true }).inputValue(), 'Sam'); await avatar.getByRole('button', { name: 'Outfit', exact: true }).click(); assert((await avatar.locator('.avatar-option-grid button svg').count()) >= 16); await avatar.getByRole('button', { name: 'Cancel', exact: true }).click();
      await library(); await page.getByRole('button', { name: 'New folder', exact: true }).click(); await page.getByLabel('Folder name', { exact: true }).fill('Physics'); await page.getByRole('button', { name: 'Create folder', exact: true }).click(); await page.getByRole('button', { name: 'Actions for Untitled note', exact: true }).click(); await page.getByRole('menuitem', { name: 'Move to', exact: true }).click(); await page.getByRole('menuitem', { name: 'Physics', exact: true }).click(); await page.getByRole('button', { name: 'Open folder Physics', exact: true }).waitFor();
      await page.getByRole('searchbox', { name: 'Search library', exact: true }).fill('Force'); assert.equal(await page.getByRole('button', { name: 'Open Untitled note', exact: true }).count(), 1); await page.getByRole('searchbox', { name: 'Search library', exact: true }).fill('');

      for (const [name, type, buffer] of [['Forces.pdf', 'application/pdf', pdf()], ['Forces.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', pptx]]) {
        await page.getByRole('button', { name: 'Upload document', exact: true }).click(); await page.locator('input[type=file]').last().setInputFiles({ name, mimeType: type, buffer }); await page.getByRole('button', { name: `Open ${name}`, exact: true }).click(); await nav('Source');
        if (name.endsWith('.pdf')) { await page.locator('canvas.pdf-original-canvas').waitFor(); await page.getByRole('button', { name: 'Next PDF page', exact: true }).click(); await page.getByLabel('Search PDF', { exact: true }).fill('Balanced'); await page.getByText('1 matching pages', { exact: true }).waitFor(); await page.getByLabel('PDF zoom', { exact: true }).selectOption('125'); await page.getByRole('button', { name: 'Explain page', exact: true }).waitFor(); await page.waitForFunction(() => !document.querySelector('.pdf-render-status')); await page.getByRole('button', { name: 'Pen', exact: true }).click(); const scroll = page.locator('.pdf-scroll'); await scroll.evaluate(element => { element.scrollTop = 0; }); await scroll.scrollIntoViewIfNeeded(); const box = await scroll.boundingBox(), inkBox = await page.locator('.studio-ink').boundingBox(); const drawX = Math.max(box.x, inkBox.x) + 40, drawY = Math.max(box.y, inkBox.y) + 50; await page.mouse.move(drawX, drawY); await page.mouse.down(); await page.mouse.move(drawX + 60, drawY + 40, { steps: 10 }); await page.mouse.up(); assert.equal(await page.getByRole('button', { name: 'Undo ink', exact: true }).isEnabled(), true); await nav('Notes'); await nav('Source'); assert.equal(await page.getByLabel('PDF page number', { exact: true }).inputValue(), '2'); assert.equal(await page.getByLabel('PDF zoom', { exact: true }).inputValue(), '125'); assert.equal(await page.getByRole('button', { name: 'Undo ink', exact: true }).isEnabled(), true); }
        else { await page.getByLabel('Search presentation', { exact: true }).fill('Balanced'); await page.getByText('1 matching slide', { exact: true }).waitFor(); await page.getByRole('button', { name: 'Next slide', exact: true }).click(); await page.getByLabel('PowerPoint zoom', { exact: true }).selectOption('125'); }
        await shot(name.endsWith('.pdf') ? 'source-pdf-dark' : 'source-pptx-dark'); await appearance('light'); await shot(name.endsWith('.pdf') ? 'source-pdf-light' : 'source-pptx-light'); await appearance('dark'); await library();
      }
      await page.getByRole('button', { name: 'Website link', exact: true }).click(); await page.getByLabel('Link', { exact: true }).fill('https://www.youtube.com/watch?v=abcdefghijk'); await page.getByRole('button', { name: 'Add link', exact: true }).click(); const before = calls.length; await page.getByRole('button', { name: 'Get Notes', exact: true }).click(); await page.getByRole('alert').waitFor(); assert.equal(calls.length, before, 'Missing transcript must not trigger invented notes'); await library();
      await page.getByRole('button', { name: 'Actions for Forces.pdf', exact: true }).click(); await page.getByRole('menuitem', { name: 'Delete', exact: true }).click(); await page.getByRole('button', { name: 'Cancel', exact: true }).click(); assert.equal(await page.getByRole('button', { name: 'Open Forces.pdf', exact: true }).count(), 1); await page.getByRole('button', { name: 'Actions for Forces.pdf', exact: true }).click(); await page.getByRole('menuitem', { name: 'Delete', exact: true }).click(); await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click(); assert.equal(await page.getByRole('button', { name: 'Open Forces.pdf', exact: true }).count(), 0);
      assert.deepEqual(errors, [], 'Browser runtime errors');
      await page.reload(); await page.getByRole('button', { name: 'Create new', exact: true }).waitFor(); assert.equal(await page.getByRole('button', { name: 'Open Untitled note', exact: true }).count(), 0, 'Guest work survived a new session'); assert.equal(await page.getByLabel('Date of birth').count(), 0, 'Valid age confirmation was lost');
      console.log(`PASS ${size.name}: notes, equations, AI errors/context/history, lessons, quizzes, cards, real WAV playback/position, avatars, folders, uploads, source search/zoom and guest lifecycle. Transports are fixtures.`);
      await context.close();
    }
    for (const width of [320, 768, 1024]) {
      for (const colorScheme of ['light', 'dark']) {
        const context = await browser.newContext({ viewport: { width, height: 1000 }, hasTouch: true, colorScheme, reducedMotion: 'reduce' }); const page = await context.newPage(); active = page; const errors = []; page.on('pageerror', error => errors.push(error.message));
        await page.goto(origin); await page.getByRole('button', { name: 'Get started' }).click(); await page.getByLabel('Date of birth').fill('2000-01-01'); await page.getByRole('button', { name: 'Continue', exact: true }).click(); await page.getByRole('checkbox').check(); await page.getByRole('button', { name: 'Open workspace', exact: true }).click(); await page.getByRole('button', { name: 'Blank document', exact: true }).click(); await page.getByRole('textbox', { name: 'Study notes', exact: true }).fill('Testing tablet and phone layouts.');
        if (width < 600) await page.getByRole('button', { name: 'Open menu', exact: true }).click(); await page.getByRole('navigation', { name: 'Main menu', exact: true }).getByRole('button', { name: 'AI Assistant', exact: true }).click(); await page.locator('.assistant-composer textarea').waitFor();
        assert((await page.evaluate(() => document.documentElement.scrollWidth)) <= width + 1, `${width} ${colorScheme} overflow`); const input = await page.locator('.assistant-composer textarea').boundingBox(); assert(input.width >= 200 && input.x >= 0 && input.x + input.width <= width + 1, 'Chat composer clipped'); await page.screenshot({ path: path.join(output, `responsive-${width}-${colorScheme}.png`), fullPage: true }); assert.deepEqual(errors, []); await context.close();
      }
      console.log(`PASS responsive ${width}px: both themes, reduced motion and visible assistant composer.`);
    }
    console.log(`Screenshots: ${output}`);
  } catch (error) {
    if (active && !active.isClosed()) { fs.writeFileSync(path.join(output, 'failure.html'), await active.content()); await active.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }); }
    console.error(`Browser artifacts: ${output}`); throw error;
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
