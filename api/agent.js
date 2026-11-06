const { jsonRequest, reject, publicError } = require('../server/http');
const { requireAccess } = require('../server/access');
const { configured, reserveRequest } = require('../server/ai-quota');
const admin = require('../server/admin');
const { createHash } = require('node:crypto');
const { generate } = require('../server/ai-providers');
const { courseInput, courseMessages, parseCourse } = require('../src/shared/course-schema.js');

// This allowlisted tool creates course content only. It cannot run commands,
// choose database paths, delete content, or read another account's documents.
module.exports = async (req, res) => {
  if (!jsonRequest(req, res)) return;
  const args = courseInput(req.body);
  if (!args) return reject(res, 400, 'INVALID_TOOL_ARGUMENTS', 'Check the course topic, difficulty and lesson count.');
  let job;
  let claimed = false;
  try {
    if (!configured()) return reject(res, 503, 'AI_CONFIG_MISSING', 'The operator must configure Firebase Admin, a rate-limit secret, and an AI provider before courses can be built.');
    const access = await requireAccess(req, { account: true });
    const db = admin.database();
    // Paths derive from the verified token, never from an email or request UID.
    const target = db.doc(`users/${access.user.uid}/files/${args.fileId}`);
    const source = (await target.get()).data();
    if (!source || source._deleted) return reject(res, 404, 'FILE_NOT_FOUND', 'This study file has not synced to your account yet. Wait for saving to finish, then retry.');
    job = db.doc(`users/${access.user.uid}/aiJobs/${req.body.requestId}`);
    const signature = createHash('sha256').update(JSON.stringify({ args, provider: req.body.provider || 'auto', allowFallback: req.body.allowFallback === true })).digest('hex');
    const state = await db.runTransaction(async transaction => {
      const existing = (await transaction.get(job)).data();
      if (existing?.signature && existing.signature !== signature) return 'conflict';
      if (existing?.status === 'complete') {
        if (existing.fileId !== args.fileId) return 'conflict';
        return existing;
      }
      if (existing?.status === 'pending' && Date.now() - existing.startedAt < 90000) return 'busy';
      transaction.set(job, { status: 'pending', fileId: args.fileId, signature, startedAt: Date.now() });
      return 'claimed';
    });
    if (state === 'busy' || state === 'conflict') return reject(res, 409, 'GENERATION_IN_PROGRESS', 'This request is already being processed. Wait before retrying.');
    if (state?.status === 'complete') return res.status(200).json({ course: state.course, provider: state.provider, fallback: state.fallback, saved: true, fileId: args.fileId });
    claimed = true;
    const quota = await reserveRequest(req, 'study', access.user.uid);
    if (quota.unavailable) throw Object.assign(new Error('Usage limits could not be checked. Please retry.'), { status: 503, code: 'QUOTA_UNAVAILABLE' });
    if (quota.retryAfter) {
      res.setHeader('Retry-After', String(quota.retryAfter));
      throw Object.assign(new Error('The free AI request limit was reached. Please retry later.'), { status: 429, code: 'USAGE_LIMIT' });
    }
    const output = await generate({ messages: courseMessages(args, typeof source.notes === 'string' && source.notes ? source.notes : (args.sourceText || '')), maxTokens: 7000 }, req.body, {adult:access.adult===true});
    let units;
    try { units = parseCourse(output.text, args.lessons); }
    catch { throw Object.assign(new Error('The AI returned an incomplete course. Your previous course is unchanged. Retry generation.'), { status: 502, code: 'INVALID_COURSE' }); }
    const course = { title: args.topic, difficulty: args.difficulty, objectives: args.objectives || '', units, completed: [], generatedAt: new Date().toISOString() };
    await db.runTransaction(async transaction => {
      const current = (await transaction.get(target)).data();
      if (!current || current._deleted) throw Object.assign(new Error('The file was removed before the course could be saved.'), { status: 409, code: 'FILE_CHANGED' });
      transaction.update(target, { course, name: current._isBlank ? args.topic : current.name, _updatedAt: new Date() });
      transaction.set(job, { status: 'complete', fileId: args.fileId, signature, course, provider: output.provider, fallback: output.fallback, completedAt: Date.now() });
    });
    return res.status(200).json({ course, saved: true, fileId: args.fileId, provider: output.provider, fallback: output.fallback });
  } catch (error) {
    if (job && claimed) await job.set({ status: 'failed', failedAt: Date.now() }, { merge: true }).catch(() => {});
    return reject(res, publicError(error) ? error.status : 503, publicError(error) ? error.code : 'COURSE_UNAVAILABLE', publicError(error) ? error.message : 'Course generation could not finish. Your previous content is unchanged. Please retry.');
  }
};
