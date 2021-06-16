(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5320],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return c},kt:function(){return f}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function p(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var l=r.createContext({}),s=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},c=function(e){var n=s(e.components);return r.createElement(l.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},d=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,c=p(e,["components","mdxType","originalType","parentName"]),d=s(t),f=o,m=d["".concat(l,".").concat(f)]||d[f]||u[f]||a;return t?r.createElement(m,i(i({ref:n},c),{},{components:t})):r.createElement(m,i({ref:n},c))}));function f(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=t.length,i=new Array(a);i[0]=d;var p={};for(var l in n)hasOwnProperty.call(n,l)&&(p[l]=n[l]);p.originalType=e,p.mdxType="string"==typeof e?e:o,i[1]=p;for(var s=2;s<a;s++)i[s]=t[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}d.displayName="MDXCreateElement"},3899:function(e,n,t){"use strict";t.d(n,{Z:function(){return l},T:function(){return s}});var r=t(2122),o=t(7294),a=t(6742),i=t(2263),p=t(907);function l(e){return o.createElement(a.Z,(0,r.Z)({},e,{to:(n=e.to,l=(0,p.zu)(),(0,i.default)().siteConfig.customFields.githubLinkVersionToBaseUrl[null!=(t=null==l?void 0:l.name)?t:"current"]+n),target:"_blank"}));var n,t,l}function s(e){var n,t=null!=(n=e.text)?n:"Example";return o.createElement(l,e,o.createElement("span",null,"\xa0"),o.createElement("img",{src:"https://img.shields.io/badge/-"+t+"-informational",alt:"Example"}))}},4459:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return l},metadata:function(){return s},toc:function(){return c},default:function(){return d}});var r=t(2122),o=t(9756),a=(t(7294),t(3905)),i=t(3899),p=["components"],l={id:"app_help",title:"Customizing Application's help",sidebar_label:"Customizing Application's help"},s={unversionedId:"configure_hydra/app_help",id:"version-1.1/configure_hydra/app_help",isDocsHomePage:!1,title:"Customizing Application's help",description:"Hydra provides two different help options:",source:"@site/versioned_docs/version-1.1/configure_hydra/app_help.md",sourceDirName:"configure_hydra",slug:"/configure_hydra/app_help",permalink:"/docs/configure_hydra/app_help",editUrl:"https://github.com/facebookresearch/kats/edit/master/website/versioned_docs/version-1.1/configure_hydra/app_help.md",version:"1.1",lastUpdatedBy:"Omry Yadan",lastUpdatedAt:1623349300,formattedLastUpdatedAt:"6/10/2021",sidebar_label:"Customizing Application's help",frontMatter:{id:"app_help",title:"Customizing Application's help",sidebar_label:"Customizing Application's help"},sidebar:"version-1.1/docs",previous:{title:"Customizing working directory pattern",permalink:"/docs/configure_hydra/workdir"},next:{title:"Colorlog plugin",permalink:"/docs/plugins/colorlog"}},c=[],u={toc:c};function d(e){var n=e.components,t=(0,o.Z)(e,p);return(0,a.kt)("wrapper",(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)(i.T,{text:"Example application",to:"examples/configure_hydra/custom_help",mdxType:"ExampleGithubLink"}),(0,a.kt)("p",null,"Hydra provides two different help options:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"--help")," : Application specific help"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"--hydra-help")," Hydra specific help. ")),(0,a.kt)("p",null,"Example output of ",(0,a.kt)("inlineCode",{parentName:"p"},"--help"),":"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text"},"$ python my_app.py --help\n== AwesomeApp ==\n\nThis is AwesomeApp!\nYou can choose a db driver by appending\n== Configuration groups ==\nCompose your configuration from those groups (db=mysql)\n\ndb: mysql, postgresql\n\n\n== Config ==\nThis is the config generated for this run.\nYou can override everything, for example:\npython my_app.py db.user=foo db.pass=bar\n-------\ndb:\n  driver: mysql\n  user: omry\n  pass: secret\n\n-------\n\nPowered by Hydra (https://hydra.cc)\nUse --hydra-help to view Hydra specific help\n")),(0,a.kt)("p",null,"This output is generated from the following config group option (selected in ",(0,a.kt)("inlineCode",{parentName:"p"},"config.yaml")," to be used by default): "),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="hydra/help/my_app_help.yaml"',title:'"hydra/help/my_app_help.yaml"'},"# App name, override to match the name your app is known by\napp_name: AwesomeApp\n\n# Help header, customize to describe your app to your users\nheader: == ${hydra.help.app_name} ==\n\nfooter: |-\n  Powered by Hydra (https://hydra.cc)\n  Use --hydra-help to view Hydra specific help\n\n# Basic Hydra flags:\n#   $FLAGS_HELP\n#\n# Config groups, choose one of:\n#   $APP_CONFIG_GROUPS: All config groups that does not start with hydra/.\n#   $HYDRA_CONFIG_GROUPS: All the Hydra config groups (starts with hydra/)\n#\n# Configuration generated with overrides:\n#   $CONFIG : Generated config\n#\ntemplate: |-\n  ${hydra.help.header}\n\n  This is ${hydra.help.app_name}!\n  You can choose a db driver by appending\n  == Configuration groups ==\n  Compose your configuration from those groups (db=mysql)\n\n  $APP_CONFIG_GROUPS\n\n  == Config ==\n  This is the config generated for this run.\n  You can override everything, for example:\n  python my_app.py db.user=foo db.pass=bar\n  -------\n  $CONFIG\n  -------\n  \n  ${hydra.help.footer}\n")))}d.isMDXComponent=!0}}]);