(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9828],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return m},kt:function(){return u}});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=a.createContext({}),c=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},m=function(e){var t=c(e.components);return a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},p=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,s=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),p=c(n),u=i,h=p["".concat(s,".").concat(u)]||p[u]||d[u]||o;return n?a.createElement(h,r(r({ref:t},m),{},{components:n})):a.createElement(h,r({ref:t},m))}));function u(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=p;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:i,r[1]=l;for(var c=2;c<o;c++)r[c]=n[c];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}p.displayName="MDXCreateElement"},5454:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return l},metadata:function(){return s},toc:function(){return c},default:function(){return d}});var a=n(2122),i=n(9756),o=(n(7294),n(3905)),r=["components"],l={id:"automatic_schema_matching",title:"Automatic schema-matching",hide_title:!0},s={unversionedId:"upgrades/1.0_to_1.1/automatic_schema_matching",id:"version-1.1/upgrades/1.0_to_1.1/automatic_schema_matching",isDocsHomePage:!1,title:"Automatic schema-matching",description:"In Hydra 1.0, when a config file is loaded, if a config with a matching name and group is present in the ConfigStore,",source:"@site/versioned_docs/version-1.1/upgrades/1.0_to_1.1/automatic_schema_matching.md",sourceDirName:"upgrades/1.0_to_1.1",slug:"/upgrades/1.0_to_1.1/automatic_schema_matching",permalink:"/docs/upgrades/1.0_to_1.1/automatic_schema_matching",editUrl:"https://github.com/facebookresearch/kats/edit/master/website/versioned_docs/version-1.1/upgrades/1.0_to_1.1/automatic_schema_matching.md",version:"1.1",lastUpdatedBy:"Omry Yadan",lastUpdatedAt:1623349300,formattedLastUpdatedAt:"6/10/2021",frontMatter:{id:"automatic_schema_matching",title:"Automatic schema-matching",hide_title:!0},sidebar:"version-1.1/docs",previous:{title:"Changes to Package Header",permalink:"/docs/upgrades/1.0_to_1.1/changes_to_package_header"},next:{title:"Config path changes",permalink:"/docs/upgrades/0.11_to_1.0/config_path_changes"}},c=[{value:"Migration",id:"migration",children:[{value:"Option 1: rename the Structured Config",id:"option-1-rename-the-structured-config",children:[]},{value:"Option 2: rename the config file",id:"option-2-rename-the-config-file",children:[]}]}],m={toc:c};function d(e){var t=e.components,n=(0,i.Z)(e,r);return(0,o.kt)("wrapper",(0,a.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"In Hydra 1.0, when a config file is loaded, if a config with a matching name and group is present in the ",(0,o.kt)("inlineCode",{parentName:"p"},"ConfigStore"),",\nit is used as the schema for the newly loaded config."),(0,o.kt)("p",null,"There are several problems with this approach:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Inflexible"),": This approach can only be used when a schema should validate a single config file.\nIt does not work if you want to use the same schema to validate multiple config files."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("strong",{parentName:"li"},"Unexpected"),": This behavior can be unexpected. There is no way to tell this is going to happen when looking at a given\nconfig file.")),(0,o.kt)("p",null,"Hydra 1.1 deprecates this behavior in favor of an explicit config extension via the Defaults List.",(0,o.kt)("br",{parentName:"p"}),"\n","This upgrade page aims to provide a summary of the required changes. It is highly recommended that you read the following pages:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/docs/advanced/defaults_list"},"Background: The Defaults List")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/docs/patterns/extending_configs"},"Background: Extending configs")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("a",{parentName:"li",href:"/docs/tutorials/structured_config/schema"},"Tutorial: Structured config schema"))),(0,o.kt)("h2",{id:"migration"},"Migration"),(0,o.kt)("p",null,"Before the upgrade, you have two different configs with the same name (a config file, and a Structured Config in the ",(0,o.kt)("inlineCode",{parentName:"p"},"ConfigStore"),").\nYou need to rename one of them. Depending on the circumstances and your preference you may rename one or the other."),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"If you control both configs, you can rename either of them."),(0,o.kt)("li",{parentName:"ul"},"If you only control the config file, rename it.")),(0,o.kt)("h3",{id:"option-1-rename-the-structured-config"},"Option 1: rename the Structured Config"),(0,o.kt)("p",null,"This option is less disruptive. Use it if you control the Structured Config.  "),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"Use a different name when storing the schema into the Config Store. Common choices:",(0,o.kt)("ul",{parentName:"li"},(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"base_")," prefix, e.g. ",(0,o.kt)("inlineCode",{parentName:"li"},"base_mysql"),"."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"_schema")," suffix, e.g. ",(0,o.kt)("inlineCode",{parentName:"li"},"mysql_schema"),"."))),(0,o.kt)("li",{parentName:"ol"},"Add the schema to the Defaults List of the extending config file.")),(0,o.kt)("details",null,(0,o.kt)("summary",null,"Click to show an example"),(0,o.kt)("h4",{id:"hydra-10"},"Hydra 1.0"),(0,o.kt)("div",{className:"row"},(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="db/mysql.yaml"',title:'"db/mysql.yaml"'},"# @package _group_\nhost: localhost\nport: 3306\n\n\n\n\n\n\n"))),(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="db/mysql schema in the ConfigStore"',title:'"db/mysql',schema:!0,in:!0,the:!0,'ConfigStore"':!0},'@dataclass\nclass MySQLConfig:\n    host: str\n    port: int\n\ncs = ConfigStore.instance()\ncs.store(group="db",\n         name="mysql", \n         node=MySQLConfig)\n')))),(0,o.kt)("h4",{id:"hydra-11"},"Hydra 1.1"),(0,o.kt)("div",{className:"row"},(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="db/mysql.yaml" {1,2}',title:'"db/mysql.yaml"',"{1,2}":!0},"defaults:\n  - base_mysql\n\nhost: localhost\nport: 3306\n\n\n\n\n"))),(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="db/mysql schema in the ConfigStore" {8}',title:'"db/mysql',schema:!0,in:!0,the:!0,'ConfigStore"':!0,"{8}":!0},'@dataclass\nclass MySQLConfig:\n    host: str\n    port: int\n\ncs = ConfigStore.instance()\ncs.store(group="db",\n         name="base_mysql", \n         node=MySQLConfig)\n'))))),(0,o.kt)("h3",{id:"option-2-rename-the-config-file"},"Option 2: rename the config file"),(0,o.kt)("p",null,"This option is a bit more disruptive. Use it if you only control the config file."),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"Rename the config file. Common choices are ",(0,o.kt)("inlineCode",{parentName:"li"},"custom_")," or ",(0,o.kt)("inlineCode",{parentName:"li"},"my_")," prefix, e.g. ",(0,o.kt)("inlineCode",{parentName:"li"},"custom_mysql.yaml"),". You can also use a domain specific name like ",(0,o.kt)("inlineCode",{parentName:"li"},"prod_mysql.yaml"),"."),(0,o.kt)("li",{parentName:"ol"},"Add the schema to the Defaults List of the extending config file."),(0,o.kt)("li",{parentName:"ol"},"Update references to the config name accordingly, e.g. on the command-line ",(0,o.kt)("inlineCode",{parentName:"li"},"db=mysql")," would become ",(0,o.kt)("inlineCode",{parentName:"li"},"db=custom_mysql"),", and in a defaults list ",(0,o.kt)("inlineCode",{parentName:"li"},"db: mysql")," would become ",(0,o.kt)("inlineCode",{parentName:"li"},"db: custom_mysql"),".")),(0,o.kt)("details",null,(0,o.kt)("summary",null,"Click to show an example"),(0,o.kt)("h4",{id:"hydra-10-1"},"Hydra 1.0"),(0,o.kt)("div",{className:"row"},(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="db/mysql.yaml"',title:'"db/mysql.yaml"'},"# @package _group_\nhost: localhost\nport: 3306\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="config.yaml"',title:'"config.yaml"'},"defaults:\n  - db: mysql\n"))),(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="db/mysql schema in the ConfigStore"',title:'"db/mysql',schema:!0,in:!0,the:!0,'ConfigStore"':!0},'@dataclass\nclass MySQLConfig:\n    host: str\n    port: int\n\ncs = ConfigStore.instance()\ncs.store(group="db",\n         name="mysql", \n         node=MySQLConfig)\n\n')))),(0,o.kt)("h4",{id:"hydra-11-1"},"Hydra 1.1"),(0,o.kt)("p",null,"Rename ",(0,o.kt)("inlineCode",{parentName:"p"},"db/mysql.yaml")," to ",(0,o.kt)("inlineCode",{parentName:"p"},"db/custom_mysql.yaml")," and explicitly add the schema to the Defaults List."),(0,o.kt)("div",{className:"row"},(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="db/custom_mysql.yaml" {1,2}',title:'"db/custom_mysql.yaml"',"{1,2}":!0},"defaults:\n  - mysql\n\nhost: localhost\nport: 3306\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="config.yaml" {2}',title:'"config.yaml"',"{2}":!0},"defaults:\n  - db: custom_mysql\n"))),(0,o.kt)("div",{className:"col col--6"},(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="db/mysql schema in the ConfigStore"',title:'"db/mysql',schema:!0,in:!0,the:!0,'ConfigStore"':!0},"\n\n\n\n\n                   NO CHANGES\n\n\n\n\n\n\n")))),(0,o.kt)("p",null,"Don't forget to also update your command line overrides from ",(0,o.kt)("inlineCode",{parentName:"p"},"db=mysql")," to ",(0,o.kt)("inlineCode",{parentName:"p"},"db=custom_mysql"),".")))}d.isMDXComponent=!0}}]);