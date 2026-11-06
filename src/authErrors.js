export function signInErrorMessage(error) {
  const messages = {
    "auth/popup-blocked": "Your browser blocked Google sign-in. Allow popups for this website, then try again.",
    "auth/popup-closed-by-user": "The Google sign-in window was closed. Try again and finish choosing your account.",
    "auth/cancelled-popup-request": "Another sign-in window is already open. Finish signing in there or close it and try again.",
    "auth/unauthorized-domain": "Google sign-in is not configured for this website address. The site owner needs to add this hostname to Firebase Authentication's authorized domains.",
    "auth/operation-not-allowed": "Google sign-in is not enabled for this website. The site owner needs to enable Google in Firebase Authentication.",
    "auth/network-request-failed": "Google sign-in could not connect. Check your connection and try again.",
    "auth/web-storage-unsupported": "Google sign-in needs browser storage. Allow site storage or try a regular browser window.",
    "auth/user-disabled": "This account is disabled. Contact the site owner for help.",
    "auth/invalid-api-key": "Sign-in is unavailable because this website's Firebase configuration is invalid. Contact the site owner.",
    "auth/api-key-not-valid.-please-pass-a-valid-api-key.": "Sign-in is unavailable because this website's Firebase configuration is invalid. Contact the site owner.",
  };
  return `${messages[error?.code] || "Google sign-in could not finish. Please try again."} Your guest session is still open.`;
}

// Firebase and IndexedDB requests can stay pending when a connection or database
// upgrade is blocked. Such a request must not prevent opening the workspace.
export function withRestoreTimeout(promise, timeoutMs = 8000) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("Workspace restore timed out.")), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timer));
}
