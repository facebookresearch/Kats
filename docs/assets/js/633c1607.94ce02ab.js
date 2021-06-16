(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9146],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return l},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=r.createContext({}),u=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},l=function(e){var t=u(e.components);return r.createElement(s.Provider,{value:t},e.children)},f={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},p=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,s=e.parentName,l=o(e,["components","mdxType","originalType","parentName"]),p=u(n),d=i,m=p["".concat(s,".").concat(d)]||p[d]||f[d]||a;return n?r.createElement(m,c(c({ref:t},l),{},{components:n})):r.createElement(m,c({ref:t},l))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,c=new Array(a);c[0]=p;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o.mdxType="string"==typeof e?e:i,c[1]=o;for(var u=2;u<a;u++)c[u]=n[u];return r.createElement.apply(null,c)}return r.createElement.apply(null,n)}p.displayName="MDXCreateElement"},3899:function(e,t,n){"use strict";n.d(t,{Z:function(){return s},T:function(){return u}});var r=n(2122),i=n(7294),a=n(6742),c=n(2263),o=n(907);function s(e){return i.createElement(a.Z,(0,r.Z)({},e,{to:(t=e.to,s=(0,o.zu)(),(0,c.default)().siteConfig.customFields.githubLinkVersionToBaseUrl[null!=(n=null==s?void 0:s.name)?n:"current"]+t),target:"_blank"}));var t,n,s}function u(e){var t,n=null!=(t=e.text)?t:"Example";return i.createElement(s,e,i.createElement("span",null,"\xa0"),i.createElement("img",{src:"https://img.shields.io/badge/-"+n+"-informational",alt:"Example"}))}},2503:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return s},metadata:function(){return u},toc:function(){return l},default:function(){return p}});var r=n(2122),i=n(9756),a=(n(7294),n(3905)),c=n(3899),o=["components"],s={id:"hierarchical_static_config",title:"A hierarchical static configuration"},u={unversionedId:"tutorials/structured_config/hierarchical_static_config",id:"version-1.0/tutorials/structured_config/hierarchical_static_config",isDocsHomePage:!1,title:"A hierarchical static configuration",description:"Dataclasses can be nested and then accessed via a common root.  The entire tree is type checked.",source:"@site/versioned_docs/version-1.0/tutorials/structured_config/2_hierarchical_static_config.md",sourceDirName:"tutorials/structured_config",slug:"/tutorials/structured_config/hierarchical_static_config",permalink:"/docs/1.0/tutorials/structured_config/hierarchical_static_config",editUrl:"https://github.com/facebookresearch/kats/edit/master/website/versioned_docs/version-1.0/tutorials/structured_config/2_hierarchical_static_config.md",version:"1.0",lastUpdatedBy:"Omry Yadan",lastUpdatedAt:1611167889,formattedLastUpdatedAt:"1/20/2021",sidebarPosition:2,frontMatter:{id:"hierarchical_static_config",title:"A hierarchical static configuration"},sidebar:"version-1.0/docs",previous:{title:"Minimal example",permalink:"/docs/1.0/tutorials/structured_config/minimal_example"},next:{title:"Config Groups",permalink:"/docs/1.0/tutorials/structured_config/config_groups"}},l=[],f={toc:l};function p(e){var t=e.components,n=(0,i.Z)(e,o);return(0,a.kt)("wrapper",(0,r.Z)({},f,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)(c.T,{to:"examples/tutorials/structured_configs/2_static_complex",mdxType:"ExampleGithubLink"}),(0,a.kt)("p",null,"Dataclasses can be nested and then accessed via a common root.  The entire tree is type checked."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'@dataclass\nclass MySQLConfig:\n    host: str = "localhost"\n    port: int = 3306\n\n@dataclass\nclass UserInterface:\n    title: str = "My app"\n    width: int = 1024\n    height: int = 768\n\n@dataclass\nclass MyConfig:\n    db: MySQLConfig = MySQLConfig()\n    ui: UserInterface = UserInterface()\n\ncs = ConfigStore.instance()\ncs.store(name="config", node=MyConfig)\n\n@hydra.main(config_name="config")\ndef my_app(cfg: MyConfig) -> None:\n    print(f"Title={cfg.ui.title}, size={cfg.ui.width}x{cfg.ui.height} pixels")\n\nif __name__ == "__main__":\n    my_app()\n')))}p.isMDXComponent=!0}}]);