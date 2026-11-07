// Several drawers can overlap. Restoring a captured "hidden" value from one
// drawer used to leave the page unable to scroll after both drawers closed.
const locks = new Set();
let previous = "";
let rootWasLocked = false;
export function lockBodyScroll() {
  const token = Symbol("dialog");
  if (!locks.size) { previous = document.body.style.overflow; rootWasLocked = document.documentElement.classList.contains("classio-scroll-locked"); document.body.style.overflow = "hidden"; document.documentElement.classList.add("classio-scroll-locked"); }
  locks.add(token);
  return () => {
    if (!locks.delete(token)) return;
    if (!locks.size) { document.body.style.overflow = previous; if (!rootWasLocked) document.documentElement.classList.remove("classio-scroll-locked"); }
  };
}
