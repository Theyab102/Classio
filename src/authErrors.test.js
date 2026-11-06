import { signInErrorMessage, withRestoreTimeout } from "./authErrors";

test.each([
  ["auth/popup-blocked", /Allow popups/],
  ["auth/unauthorized-domain", /authorized domains/],
  ["auth/operation-not-allowed", /enable Google/],
  ["auth/network-request-failed", /Check your connection/],
  ["auth/popup-closed-by-user", /finish choosing your account/],
  ["auth/web-storage-unsupported", /Allow site storage/],
])("explains how to recover from %s without losing guest work", (code, message) => {
  expect(signInErrorMessage({ code })).toMatch(message);
  expect(signInErrorMessage({ code })).toMatch(/guest session is still open/);
});

test("does not expose raw Firebase error details", () => {
  expect(signInErrorMessage({ message: "internal credential detail" })).not.toMatch(/credential/);
});

test("bounds a stalled restore and leaves settled requests unchanged", async () => {
  jest.useFakeTimers();
  try {
    const pending = withRestoreTimeout(new Promise(() => {}));
    const rejected = expect(pending).rejects.toThrow("Workspace restore timed out");
    jest.advanceTimersByTime(8000);
    await rejected;
    await expect(withRestoreTimeout(Promise.resolve("saved workspace"))).resolves.toBe("saved workspace");
    await expect(withRestoreTimeout(Promise.reject(new Error("offline")))).rejects.toThrow("offline");
    expect(jest.getTimerCount()).toBe(0);
  } finally { jest.useRealTimers(); }
});
