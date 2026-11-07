import { decodeWorkspace, encodeWorkspace } from "./workspaceCodec";

test("presentation tables survive cloud encoding without losing existing workspace fields", () => {
  const folders = [{ id: "science", files: [{ id: "a", notes: "My notes", presentation: [{ type: "table", table: [["Quantity", "Unit"], ["Force", "Newton"]] }], course: { completed: [0], units: [{ options: ["A", "B", "C"] }] } }], color: "#528ca6" }];
  const encoded = encodeWorkspace(folders);
  const inspect = (value, parentArray = false) => { if (Array.isArray(value)) { expect(parentArray).toBe(false); value.forEach(item => inspect(item, true)); } else if (value && typeof value === "object") Object.values(value).forEach(item => inspect(item)); };
  inspect(encoded);
  expect(decodeWorkspace(encoded)).toEqual(folders);
  expect(decodeWorkspace(folders)).toEqual(folders);
});
