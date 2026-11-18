class ProviderError extends Error {
  constructor(status, code, retryable = false, retryAfter = 0) {
    super(code);
    this.status = status;
    this.code = code;
    this.retryable = retryable;
    this.retryAfter = retryAfter;
  }
}
module.exports = { ProviderError };
