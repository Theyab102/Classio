const dns = require("node:dns/promises");
const http = require("node:http"), https = require("node:https");
const zlib = require("node:zlib");
const ipaddr = require("ipaddr.js");
const { load } = require("cheerio");
const LIMIT = 2_000_000;

function publicIp(value) {
  try { let address = ipaddr.parse(value); if (address.kind() === "ipv6" && address.isIPv4MappedAddress()) address = address.toIPv4Address(); return address.range() === "unicast"; } catch { return false; }
}
function publicUrl(value, base) {
  const url = new URL(value, base);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || (url.port && !["80", "443"].includes(url.port)) || /(?:^|\.)(?:localhost|local|internal|test|invalid)$/.test(url.hostname)) throw new Error("PUBLIC_URL_REQUIRED");
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (ipaddr.isValid(host) && !publicIp(host)) throw new Error("PUBLIC_URL_REQUIRED");
  return url;
}
async function publicAddresses(url, lookup = dns.lookup) {
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const values = ipaddr.isValid(host) ? [{ address: host, family: ipaddr.parse(host).kind() === "ipv6" ? 6 : 4 }] : await lookup(host, { all: true, verbatim: true });
  if (!values.length || values.some(value => !publicIp(value.address))) throw new Error("PUBLIC_URL_REQUIRED");
  return values;
}
async function readHtml(value, depth = 0) {
  if (depth > 3) throw new Error("TOO_MANY_REDIRECTS");
  const url = publicUrl(value), addresses = await publicAddresses(url);
  const response = await new Promise((resolve, reject) => {
    const transport = url.protocol === "https:" ? https : http;
    const request = transport.get(url, { headers: { "User-Agent": "ClassioStudyReader/1.0", Accept: "text/html,application/xhtml+xml", "Accept-Encoding": "identity" },
      // Pin the validated DNS answer. The original hostname remains in TLS/SNI.
      lookup: (_host, options, callback) => { const selected = addresses[0]; callback(null, options?.all ? addresses : selected.address, options?.all ? undefined : selected.family); }, signal: AbortSignal.timeout(10000) }, result => {
      const chunks = []; let bytes = 0;
      result.on("data", chunk => { bytes += chunk.length; if (bytes > LIMIT) { result.destroy(new Error("SOURCE_TOO_LARGE")); return; } chunks.push(chunk); });
      result.on("error", reject); result.on("end", () => resolve({ status: result.statusCode, headers: result.headers, body: Buffer.concat(chunks) }));
    }); request.on("error", reject);
  });
  if ([301, 302, 303, 307, 308].includes(response.status)) return readHtml(publicUrl(response.headers.location, url).href, depth + 1);
  if (response.status !== 200 || !/^text\/html|^application\/xhtml\+xml/i.test(response.headers["content-type"] || "")) throw new Error("SOURCE_UNAVAILABLE");
  let body = response.body;
  if (response.headers["content-encoding"] === "gzip") body = zlib.gunzipSync(body, { maxOutputLength: LIMIT });
  else if (response.headers["content-encoding"] === "br") body = zlib.brotliDecompressSync(body, { maxOutputLength: LIMIT });
  return { html: body.toString("utf8"), url: url.href };
}
function extractSource(html, sourceUrl) {
  const $ = load(html);
  const title = ($('meta[property="og:title"]').attr("content") || $("h1").first().text() || $("title").text() || new URL(sourceUrl).hostname).trim().slice(0, 200);
  $("script,style,noscript,nav,header,footer,aside,iframe,form,svg,button,[aria-hidden=true]").remove();
  const article = $("article").first(), main = $("main,[role=main],.mw-parser-output").first(), root = article.length ? article : main.length ? main : $("body");
  const images = [], seen = new Set();
  root.find("img").each((_index, element) => {
    if (images.length >= 6) return;
    const image = $(element), src = image.attr("data-src") || image.attr("src"), alt = (image.attr("alt") || image.closest("figure").find("figcaption").text() || "Source figure").trim().slice(0, 250);
    if (!src || /(?:pixel|tracking|spacer|avatar|logo|icon)[-_.\/]/i.test(src) || (Number(image.attr("width")) > 0 && Number(image.attr("width")) < 100) || (Number(image.attr("height")) > 0 && Number(image.attr("height")) < 70)) return;
    try { const url = publicUrl(src, sourceUrl).href; if (!seen.has(url)) { seen.add(url); images.push({ src: url, title: alt, source: sourceUrl }); } } catch {}
  });
  const tables = [];
  root.find("table").slice(0, 4).each((_i, table) => { const rows = []; $(table).find("tr").slice(0, 12).each((_j, tr) => { const cells = $(tr).find("th,td").slice(0, 6).map((_k, cell) => $(cell).text().replace(/\s+/g, " ").trim().slice(0, 250)).get(); if (cells.length > 1) rows.push(cells); }); if (rows.length > 1) tables.push(rows); });
  root.find("br").replaceWith("\n"); root.find("p,h1,h2,h3,h4,li,tr,section").append("\n");
  const text = root.text().replace(/[ \t]+/g, " ").replace(/\n\s*\n/g, "\n\n").trim().slice(0, 24000);
  if (text.length < 100) throw new Error("SOURCE_UNAVAILABLE");
  return { title, text, images, tables, url: sourceUrl };
}
async function publicSource(url) { const page = await readHtml(publicUrl(url).href); return extractSource(page.html, page.url); }
module.exports = { publicIp, publicUrl, publicAddresses, extractSource, publicSource };
