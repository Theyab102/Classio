let pdfLibrary;

export function loadPdfLibrary() {
  pdfLibrary ??= import("pdfjs-dist").then(library => {
    library.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).href;
    return library;
  });
  return pdfLibrary;
}

export function openPdf(source) {
  return loadPdfLibrary().then(library => library.getDocument({ ...source, isEvalSupported: false }).promise);
}
