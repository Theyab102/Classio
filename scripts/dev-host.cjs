// CRA 5 passes an undefined allowedHost for an explicit loopback HOST when a
// proxy is configured. Supply an explicit local allowlist, keeping host checks.
const configurationPath = require.resolve('react-scripts/config/webpackDevServer.config');
const configuration = require(configurationPath);
require.cache[configurationPath].exports = (...args) => ({
  ...configuration(...args),
  allowedHosts: ['localhost', '127.0.0.1'],
});
