import { readAccess, rememberAccess } from "./access";

beforeEach(() => { localStorage.clear(); sessionStorage.clear(); });

test("an expired ineligible decision can be checked again even with the legacy tab record", () => {
  localStorage.setItem("classio_access_v2", JSON.stringify({ decision: "blocked", at: Date.now() - 86400001 }));
  sessionStorage.setItem("classio_access_v1", "blocked");
  expect(readAccess()).toBeNull();
});

test("stores only the decision and its time, and rejects unknown decisions", () => {
  rememberAccess("accepted");
  expect(Object.keys(JSON.parse(localStorage.getItem("classio_access_v2"))).sort()).toEqual(["at", "decision"]);
  rememberAccess("bypass");
  expect(readAccess()).toBe("accepted");
});

test("blocked browser storage does not turn eligibility into a service outage", () => {
  const storage = jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Blocked"); });
  try { expect(() => rememberAccess("eligible")).not.toThrow(); }
  finally { storage.mockRestore(); }
});
