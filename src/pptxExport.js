const pptxBundle = new URL("../node_modules/pptxgenjs/dist/pptxgen.bundle.js", import.meta.url);
let library;

export function loadPptxExporter() {
  if (window.PptxGenJS) return Promise.resolve(window.PptxGenJS);
  library ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = pptxBundle.href;
    script.onload = () => {
      if (window.PptxGenJS) resolve(window.PptxGenJS);
      else reject(new Error("Could not initialize PowerPoint export."));
    };
    script.onerror = () => reject(new Error("Could not load PowerPoint export."));
    document.head.append(script);
  }).catch(error => { library = null; throw error; });
  return library;
}
