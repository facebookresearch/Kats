(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{102:function(e,t,a){"use strict";var n=a(0),l=a.n(n);var r=function(e,t,a){const[l,r]=Object(n.useState)(void 0);Object(n.useEffect)(()=>{let n,i;function c(){const c=function(){let e=0,t=null;for(n=document.getElementsByClassName("anchor");e<n.length&&!t;){const l=n[e],{top:r}=l.getBoundingClientRect();r>=0&&r<=a&&(t=l),e+=1}return t}();if(c){let a=0,n=!1;for(i=document.getElementsByClassName(e);a<i.length&&!n;){const e=i[a],{href:s}=e,o=decodeURIComponent(s.substring(s.indexOf("#")+1));c.id===o&&(l&&l.classList.remove(t),e.classList.add(t),r(e),n=!0),a+=1}}}return document.addEventListener("scroll",c),document.addEventListener("resize",c),c(),()=>{document.removeEventListener("scroll",c),document.removeEventListener("resize",c)}})},i=a(52),c=a.n(i);const s="table-of-contents__link";function o({headings:e,isChild:t}){return e.length?l.a.createElement("ul",{className:t?"":"table-of-contents table-of-contents__left-border"},e.map(e=>l.a.createElement("li",{key:e.id},l.a.createElement("a",{href:"#"+e.id,className:s,dangerouslySetInnerHTML:{__html:e.value}}),l.a.createElement(o,{isChild:!0,headings:e.children})))):null}t.a=function({headings:e}){return r(s,"table-of-contents__link--active",100),l.a.createElement("div",{className:c.a.tableOfContents},l.a.createElement(o,{headings:e}))}},82:function(e,t,a){"use strict";a.r(t);var n=a(0),l=a.n(n),r=a(95),i=a(86),c=a(90),s=a(88);var o=function(e){const{metadata:t}=e;return l.a.createElement("nav",{className:"pagination-nav","aria-label":"Blog list page navigation"},l.a.createElement("div",{className:"pagination-nav__item"},t.previous&&l.a.createElement(s.a,{className:"pagination-nav__link",to:t.previous.permalink},l.a.createElement("div",{className:"pagination-nav__sublabel"},"Previous"),l.a.createElement("div",{className:"pagination-nav__label"},"\xab ",t.previous.title))),l.a.createElement("div",{className:"pagination-nav__item pagination-nav__item--next"},t.next&&l.a.createElement(s.a,{className:"pagination-nav__link",to:t.next.permalink},l.a.createElement("div",{className:"pagination-nav__sublabel"},"Next"),l.a.createElement("div",{className:"pagination-nav__label"},t.next.title," \xbb"))))},m=a(100);var d=function(){const{siteConfig:{title:e}}=Object(i.a)(),{pluginId:t}=Object(m.useActivePlugin)({failfast:!0}),a=Object(m.useActiveVersion)(t),{latestDocSuggestion:n,latestVersionSuggestion:r}=Object(m.useDocVersionSuggestions)(t);if(!r)return l.a.createElement(l.a.Fragment,null);const c=null!=n?n:(o=r).docs.find(e=>e.id===o.mainDocId);var o;return l.a.createElement("div",{className:"alert alert--warning margin-bottom--md",role:"alert"},"current"===a.name?l.a.createElement("div",null,"This is unreleased documentation for ",e," ",l.a.createElement("strong",null,a.label)," version."):l.a.createElement("div",null,"This is documentation for ",e," ",l.a.createElement("strong",null,a.label),", which is no longer actively maintained."),l.a.createElement("div",{className:"margin-top--md"},"For up-to-date documentation, see the"," ",l.a.createElement("strong",null,l.a.createElement(s.a,{to:c.path},"latest version"))," ","(",r.label,")."))},u=a(102),g=a(87),E=a(60),v=a.n(E);t.default=function(e){const{siteConfig:t={}}=Object(i.a)(),{url:a,title:n}=t,{content:s}=e,{metadata:E}=s,{description:p,title:f,permalink:b,editUrl:h,lastUpdatedAt:N,lastUpdatedBy:_}=E,{frontMatter:{image:w,keywords:y,hide_title:O,hide_table_of_contents:k}}=s,{pluginId:j}=Object(m.useActivePlugin)({failfast:!0}),C=Object(m.useVersions)(j),x=Object(m.useActiveVersion)(j),I=C.length>1,L=f?`${f} | ${n}`:n,A=Object(c.a)(w,{absolute:!0});return l.a.createElement(l.a.Fragment,null,l.a.createElement(r.a,null,l.a.createElement("title",null,L),l.a.createElement("meta",{property:"og:title",content:L}),p&&l.a.createElement("meta",{name:"description",content:p}),p&&l.a.createElement("meta",{property:"og:description",content:p}),y&&y.length&&l.a.createElement("meta",{name:"keywords",content:y.join(",")}),w&&l.a.createElement("meta",{property:"og:image",content:A}),w&&l.a.createElement("meta",{property:"twitter:image",content:A}),w&&l.a.createElement("meta",{name:"twitter:image:alt",content:"Image for "+f}),b&&l.a.createElement("meta",{property:"og:url",content:a+b}),b&&l.a.createElement("link",{rel:"canonical",href:a+b})),l.a.createElement("div",{className:Object(g.a)("container padding-vert--lg",v.a.docItemWrapper)},l.a.createElement("div",{className:"row"},l.a.createElement("div",{className:Object(g.a)("col",{[v.a.docItemCol]:!k})},l.a.createElement(d,null),l.a.createElement("div",{className:v.a.docItemContainer},l.a.createElement("article",null,I&&l.a.createElement("div",null,l.a.createElement("span",{className:"badge badge--secondary"},"Version: ",x.label)),!O&&l.a.createElement("header",null,l.a.createElement("h1",{className:v.a.docTitle},f)),l.a.createElement("div",{className:"markdown"},l.a.createElement(s,null))),(h||N||_)&&l.a.createElement("div",{className:"margin-vert--xl"},l.a.createElement("div",{className:"row"},l.a.createElement("div",{className:"col"},h&&l.a.createElement("a",{href:h,target:"_blank",rel:"noreferrer noopener"},l.a.createElement("svg",{fill:"currentColor",height:"1.2em",width:"1.2em",preserveAspectRatio:"xMidYMid meet",viewBox:"0 0 40 40",style:{marginRight:"0.3em",verticalAlign:"sub"}},l.a.createElement("g",null,l.a.createElement("path",{d:"m34.5 11.7l-3 3.1-6.3-6.3 3.1-3q0.5-0.5 1.2-0.5t1.1 0.5l3.9 3.9q0.5 0.4 0.5 1.1t-0.5 1.2z m-29.5 17.1l18.4-18.5 6.3 6.3-18.4 18.4h-6.3v-6.2z"}))),"Edit this page")),(N||_)&&l.a.createElement("div",{className:"col text--right"},l.a.createElement("em",null,l.a.createElement("small",null,"Last updated"," ",N&&l.a.createElement(l.a.Fragment,null,"on"," ",l.a.createElement("time",{dateTime:new Date(1e3*N).toISOString(),className:v.a.docLastUpdatedAt},new Date(1e3*N).toLocaleDateString()),_&&" "),_&&l.a.createElement(l.a.Fragment,null,"by ",l.a.createElement("strong",null,_)),!1))))),l.a.createElement("div",{className:"margin-vert--lg"},l.a.createElement(o,{metadata:E})))),!k&&s.rightToc&&l.a.createElement("div",{className:"col col--3"},l.a.createElement(u.a,{headings:s.rightToc})))))}}}]);