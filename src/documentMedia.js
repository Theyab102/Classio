import { loadPdfLibrary, openPdf } from "./pdf";

export const isPdfFile = file => Boolean(file && (file.type === "application/pdf" || /\.pdf$/i.test(file.name || "")));

export function safeImageSource(source) {
  return typeof source === "string" && (/^https:\/\//i.test(source) ||
    /^data:image\/(?:png|jpeg|webp|gif);base64,[a-z0-9+/=]+$/i.test(source));
}

function compactImage(source, width, height) {
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1000 / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d");
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  for (const quality of [.82, .65, .48]) {
    const data = canvas.toDataURL("image/jpeg", quality);
    if (data.length < 135000) return data;
  }
  const smaller = document.createElement("canvas");
  smaller.width = Math.max(1, Math.round(canvas.width * .65));
  smaller.height = Math.max(1, Math.round(canvas.height * .65));
  smaller.getContext("2d").drawImage(canvas, 0, 0, smaller.width, smaller.height);
  return smaller.toDataURL("image/jpeg", .6);
}

export async function prepareNoteImage(file) {
  if (!file || !/^image\/(png|jpeg|webp|gif)$/.test(file.type) || file.size > 20 * 1024 * 1024) {
    throw new Error("Choose a PNG, JPEG, WebP or GIF image under 20 MB.");
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return compactImage(image, image.naturalWidth, image.naturalHeight);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function prepareSourceFigure(figure) {
  if (!figure?.src?.startsWith("data:") || figure.src.length < 180000) return figure;
  const image = new Image(); image.src = figure.src; await image.decode();
  return { ...figure, src: compactImage(image, image.naturalWidth, image.naturalHeight) };
}

export async function getPdfContext(file, limit = 15000) {
  const pdf = await openPdf({ data: new Uint8Array(await file.arrayBuffer()) });
  try {
    let text = "";
    const pages = Math.min(pdf.numPages, 35);
    for (let index = 1; index <= pages && text.length < limit; index++) {
      const page = await pdf.getPage(index);
      const content = await page.getTextContent();
      text += `\n[PDF page ${index}]\n${content.items.map(item => (item.str || "") + (item.hasEOL ? "\n" : " ")).join("")}\n`;
      page.cleanup();
    }
    return text.slice(0, limit);
  } finally { await pdf.destroy(); }
}

function waitForPdfImage(page, name) {
  const cache = name.startsWith("g_") ? page.commonObjs : page.objs;
  return new Promise(resolve => {
    const timeout = setTimeout(() => resolve(null), 1800);
    cache.get(name, data => { clearTimeout(timeout); resolve(data); });
  });
}

function imageFromPdfData(image, ImageKind) {
  if (!image?.width || !image?.height || image.width * image.height > 8_000_000) return null;
  if (image.bitmap) return compactImage(image.bitmap, image.width, image.height);
  if (!image.data || ![ImageKind.RGBA_32BPP, ImageKind.RGB_24BPP, ImageKind.GRAYSCALE_1BPP].includes(image.kind)) return null;
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  const pixels = context.createImageData(image.width, image.height);
  const rgba = pixels.data;
  const stride = Math.ceil(image.width / 8);
  for (let index = 0; index < rgba.length; index += 4) {
    const pixel = index / 4;
    if (image.kind === ImageKind.GRAYSCALE_1BPP) {
      const x = pixel % image.width;
      const y = Math.floor(pixel / image.width);
      const value = image.data[y * stride + (x >> 3)] & (128 >> (x % 8)) ? 255 : 0;
      rgba.set([value, value, value, 255], index);
    } else {
      const channels = image.kind === ImageKind.RGBA_32BPP ? 4 : 3;
      const start = pixel * channels;
      rgba[index] = image.data[start];
      rgba[index + 1] = image.data[start + 1];
      rgba[index + 2] = image.data[start + 2];
      rgba[index + 3] = channels === 4 ? image.data[start + 3] : 255;
    }
  }
  context.putImageData(pixels, 0, 0);
  return compactImage(canvas, canvas.width, canvas.height);
}

export async function getPdfFigures(file, pageNumber, { pagePreview = false } = {}) {
  const { OPS, ImageKind } = await loadPdfLibrary();
  const pdf = await openPdf({ data: new Uint8Array(await file.arrayBuffer()) });
  try {
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > pdf.numPages) {
      throw new Error(`Choose a PDF page between 1 and ${pdf.numPages}.`);
    }
    const page = await pdf.getPage(pageNumber);
    const results = [];
    if (!pagePreview) {
      const operators = await page.getOperatorList();
      const seen = new Set();
      for (let index = 0; index < operators.fnArray.length && results.length < 4; index++) {
        const operator = operators.fnArray[index];
        const [value, width, height] = operators.argsArray[index] || [];
        if (operator !== OPS.paintImageXObject && operator !== OPS.paintInlineImageXObject) continue;
        if (typeof value === "string") {
          if (seen.has(value) || width < 100 || height < 75) continue;
          seen.add(value);
        }
        const image = typeof value === "string" ? await waitForPdfImage(page, value) : value;
        if (!image || image.width < 100 || image.height < 75) continue;
        const src = imageFromPdfData(image, ImageKind);
        if (src) results.push({ src, title: `PDF page ${pageNumber} · image ${results.length + 1}` });
      }
    }
    if (!results.length) {
      const base = page.getViewport({ scale: 1 });
      const viewport = page.getViewport({ scale: 1000 / Math.max(base.width, base.height) });
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(viewport.width);
      canvas.height = Math.ceil(viewport.height);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
      results.push({ src: compactImage(canvas, canvas.width, canvas.height), title: `PDF page ${pageNumber} · page snapshot` });
    }
    return results;
  } finally { await pdf.destroy(); }
}

async function commonsImages(query, signal) {
  const term = query.trim().slice(0, 100);
  if (!term) return [];
  const params = new URLSearchParams({
    action: "query", generator: "search", gsrsearch: term, gsrnamespace: "6",
    gsrlimit: "12", prop: "imageinfo", iiprop: "url|mime|extmetadata|size", iiurlwidth: "720",
    iiextmetadatafilter: "LicenseShortName|Artist|ImageDescription", format: "json", origin: "*",
  });
  const response = await fetch(`https://commons.wikimedia.org/w/api.php?${params}`, { signal, referrerPolicy: "no-referrer" });
  if (!response.ok) throw new Error("Wikimedia Commons image search is temporarily unavailable.");
  const data = await response.json();
  if (data.error) throw new Error("Wikimedia Commons could not complete this search.");
  return Object.values(data.query?.pages || {}).sort((a, b) => a.index - b.index).flatMap(page => {
    const info = page.imageinfo?.[0];
    if (!info || !/^image\/(jpeg|png|webp|gif|svg\+xml)$/.test(info.mime) || !info.thumburl?.startsWith("https://upload.wikimedia.org/") ||
        !info.descriptionurl?.startsWith("https://commons.wikimedia.org/")) return [];
    const license = info.extmetadata?.LicenseShortName?.value;
    if (!license) return [];
    const title = page.title.replace(/^File:|\.[a-z]+$/gi, "").replace(/_/g, " ");
    const artist = new DOMParser().parseFromString(info.extmetadata?.Artist?.value || "", "text/html").body.textContent.trim();
    const context = new DOMParser().parseFromString(info.extmetadata?.ImageDescription?.value || "", "text/html").body.textContent.trim().slice(0, 800);
    const credit = [title, artist, license].filter(Boolean).join(" · ");
    return [{ src: info.thumburl, source: info.descriptionurl, title, credit, context, ...(info.thumbwidth&&info.thumbheight?{width:info.thumbwidth,height:info.thumbheight}:{}) }];
  }).slice(0, 8);
}

let capabilities;
async function extraImageSearch(query,signal){
  if(!capabilities)capabilities=fetch('/api/images').then(r=>r.ok?r.json():{}).catch(()=>{capabilities=null;return {};});
  if(!(await capabilities).searchProviders?.length)return[];
  const response=await fetch('/api/images',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'search',query}),signal});
  if(!response.ok)throw new Error('Additional image sources are unavailable.');
  const data=await response.json();return Array.isArray(data.images)?data.images:[];
}
export async function searchStudyImages(query,signal){
  const results=await Promise.allSettled([commonsImages(query,signal),extraImageSearch(query,signal)]);
  if(signal?.aborted)throw new DOMException('Image search cancelled','AbortError');
  const images=results.flatMap(result=>result.status==='fulfilled'?result.value:[]);
  if(!images.length&&results.every(result=>result.status==='rejected'))throw new Error('Image search is temporarily unavailable.');
  return [...new Map(images.map(image=>[image.src,image])).values()];
}
export async function acknowledgeStudyImage(image,signal){
  if(image.provider!=='unsplash')return;
  const response=await fetch('/api/images',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'track-download',id:image.providerId}),signal});
  if(!response.ok||!(await response.json()).tracked)throw new Error('This image source could not confirm the selection. Choose another image or try again.');
}

export function likelyStudyImage(query, results) {
  const tokens = value => new Set((value.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || []).map(word => word.replace(/s$/, "")));
  const generic = new Set(["the", "and", "for", "with", "from", "diagram", "image", "picture", "labeled", "illustration", "photo", "model"]);
  const subject = [...tokens(query)].filter(word => !generic.has(word));
  if (!subject.length) return null;
  const ranked = results.map(result => {
    const title = tokens((result.context || "") + " " + result.title);
    return { result, matches: subject.filter(word => title.has(word)).length };
  }).sort((a, b) => b.matches - a.matches);
  return ranked[0]?.matches >= Math.max(1, Math.ceil(subject.length * .6)) ? ranked[0].result : null;
}

export async function getPresentationFigures(file) {
  const { readSlides } = await import("./PowerPointViewer");
  const { slides,width,height } = await readSlides(file);
  return slides.flatMap((slide, index) => slide.items.filter(item => item.type === "image" && (!item.position || (parseFloat(item.position.width)>=6&&parseFloat(item.position.height)>=6&&parseFloat(item.position.width)*parseFloat(item.position.height)>=160))).slice(0, 3).map((item, imageIndex) => ({
    src: item.src, title: slide.texts[0] || "Source slide figure", context: slide.texts.join(" ").slice(0, 1000), credit: "Source slide " + (index + 1) + " · figure " + (imageIndex + 1),...(item.position?{width:parseFloat(item.position.width)*width/100,height:parseFloat(item.position.height)*height/100}:{}),
  })));
}
