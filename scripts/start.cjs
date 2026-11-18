// Keep CRA's application configuration while using the patched WDS 5 API.
process.env.NODE_ENV = "development";
process.env.BABEL_ENV = "development";
require("react-scripts/config/env");
const unsafe =
  /^REACT_APP_.*(?:KEY|TOKEN|SECRET|CREDENTIAL|PASSWORD|PRIVATE|SERVICE_ACCOUNT)/i;
for (const name of Object.keys(process.env))
  if (unsafe.test(name)) delete process.env[name];
const configPath = require.resolve(
    "react-scripts/config/webpackDevServer.config",
  ),
  legacy = require(configPath);
require.cache[configPath].exports = (...args) => {
  const {
    https,
    onBeforeSetupMiddleware: before,
    onAfterSetupMiddleware: after,
    ...config
  } = legacy(...args);
  config.server = https
    ? { type: "https", options: typeof https === "object" ? https : {} }
    : "http";
  config.allowedHosts = process.env.CLASSIO_DEV_ALLOWED_HOSTS
    ? process.env.CLASSIO_DEV_ALLOWED_HOSTS.split(",")
        .map((host) => host.trim())
        .filter(Boolean)
    : "auto";
  delete config.headers;
  config.devMiddleware.publicPath = config.devMiddleware.publicPath.replace(
    /\/?$/,
    "/",
  );
  config.setupMiddlewares = (middlewares, server) => {
    before?.(server);
    const trailing = [];
    after?.({
      app: {
        use: (middleware) =>
          trailing.push({ name: "classio-after", middleware }),
      },
    });
    return [...middlewares, ...trailing];
  };
  return config;
};
require("react-scripts/scripts/start");
