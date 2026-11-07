// Firestore forbids a bare array nested directly inside another array. Wrap
// only those arrays so editable presentation tables can sync without flattening.
export function encodeWorkspace(value, insideArray = false) {
  if (Array.isArray(value)) {
    const items = value.map(item => encodeWorkspace(item, true));
    return insideArray ? { __classio_nested_array_v1: items } : items;
  }
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, encodeWorkspace(item)]));
  return value;
}
export function decodeWorkspace(value) {
  if (Array.isArray(value)) return value.map(decodeWorkspace);
  if (value && typeof value === "object") {
    if (Object.keys(value).length === 1 && Array.isArray(value.__classio_nested_array_v1)) return value.__classio_nested_array_v1.map(decodeWorkspace);
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decodeWorkspace(item)]));
  }
  return value;
}
