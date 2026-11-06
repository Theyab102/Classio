// Keep CRA's React build configuration while using patched webpack-dev-server 5.
// The adapter translates removed v4 hooks and HTTPS options; host checks stay on.
const configurationPath=require.resolve('react-scripts/config/webpackDevServer.config');
const configuration=require(configurationPath);
require.cache[configurationPath].exports=(...args)=>{
  const {onBeforeSetupMiddleware,onAfterSetupMiddleware,https,...options}=configuration(...args);
  const publicPath=options.devMiddleware.publicPath||'/';
  return {...options,allowedHosts:['localhost','127.0.0.1'],
    ...(https?{server:{type:'https',options:https}}:{}),
    devMiddleware:{...options.devMiddleware,publicPath:publicPath.endsWith('/')?publicPath:publicPath+'/'},
    setupMiddlewares(middlewares,server){
      onBeforeSetupMiddleware?.(server);
      const after=[];
      onAfterSetupMiddleware?.({...server,app:{use:middleware=>after.push(middleware)}});
      return [...middlewares,...after.map((middleware,index)=>({name:'classio-after-'+index,middleware}))];
    },
  };
};
const DevServer=require('webpack-dev-server');
// CRA's shutdown handler calls the v4 name; v5 supplies stopCallback instead.
if(!DevServer.prototype.close)DevServer.prototype.close=function(){this.stopCallback(()=>{});};
