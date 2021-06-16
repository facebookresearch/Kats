(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4988],{3905:function(e,t,r){"use strict";r.d(t,{Zo:function(){return p},kt:function(){return m}});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),c=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},p=function(e){var t=c(e.components);return n.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},d=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),d=c(r),m=o,f=d["".concat(s,".").concat(m)]||d[m]||u[m]||a;return r?n.createElement(f,i(i({ref:t},p),{},{components:r})):n.createElement(f,i({ref:t},p))}));function m(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:o,i[1]=l;for(var c=2;c<a;c++)i[c]=r[c];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}d.displayName="MDXCreateElement"},7883:function(e,t,r){"use strict";r.r(t),r.d(t,{frontMatter:function(){return l},metadata:function(){return s},toc:function(){return c},default:function(){return u}});var n=r(2122),o=r(9756),a=(r(7294),r(3905)),i=["components"],l={id:"release",title:"Release process",sidebar_label:"Release process"},s={unversionedId:"development/release",id:"version-1.1/development/release",isDocsHomePage:!1,title:"Release process",description:"The release process may be automated in the future.",source:"@site/versioned_docs/version-1.1/development/release.md",sourceDirName:"development",slug:"/development/release",permalink:"/docs/development/release",editUrl:"https://github.com/facebookresearch/kats/edit/master/website/versioned_docs/version-1.1/development/release.md",version:"1.1",lastUpdatedBy:"Omry Yadan",lastUpdatedAt:1623349300,formattedLastUpdatedAt:"6/10/2021",sidebar_label:"Release process",frontMatter:{id:"release",title:"Release process",sidebar_label:"Release process"},sidebar:"version-1.1/docs",previous:{title:"Documentation",permalink:"/docs/development/documentation"},next:{title:"Introduction",permalink:"/docs/upgrades/intro"}},c=[],p={toc:c};function u(e){var t=e.components,r=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"The release process may be automated in the future."),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Checkout master"),(0,a.kt)("li",{parentName:"ul"},"Update the Hydra version in ",(0,a.kt)("inlineCode",{parentName:"li"},"hydra/__init__.py")),(0,a.kt)("li",{parentName:"ul"},"Update NEWS.md with towncrier"),(0,a.kt)("li",{parentName:"ul"},"Create a wheel and source dist for hydra-core: ",(0,a.kt)("inlineCode",{parentName:"li"},"python -m build")),(0,a.kt)("li",{parentName:"ul"},"Upload pip package: ",(0,a.kt)("inlineCode",{parentName:"li"},"python -m twine upload dist/*"))))}u.isMDXComponent=!0}}]);