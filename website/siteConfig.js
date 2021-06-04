/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// List of projects/orgs using your project for the users page.
const users = [{
  caption: 'User1',
  // You will need to prepend the image path with your baseUrl
  // if it is not '/', like: '/test-site/img/image.jpg'.
  image: '/img/undraw_open_source.svg',
  infoLink: 'https://www.facebook.com',
  pinned: true,
}, ];

const siteConfig = {
  title: '', // Title for your website.
  tagline: 'a toolKit to Analyze Time Series',
  url: 'https://home.fburl.com/',
  baseUrl: '/',
  // url: 'http://devvm938.vll0.facebook.com', // Your website URL
  // baseUrl: '/', // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'public_html',
  organizationName: 'facebook',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'
  // heroLogo : "img/Kats_logo_small.png",
  heroInfo: "Kats, a toolKit to analyze time series data, a light-weight, easy-to-use, and generalizable framework to perform Time Series analysis. Time Series analysis is an essential component of Data Science and Engineering work at Facebook, from understanding the key statistics and characteristics, detecting regressions and anomalies, to forecasting future trends. There are a few internal and external packages that can perform certain analyses, while a unified in-house framework is desired. Kats aims to provide the one-stop shop for time series analysis, like detection, forecasting, feature extraction, multivariate analysis, etc.",
  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [{
      doc: 'Forecasting_Quadratic_Model',
      label: 'Docs'
    },
    {
      page: 'team',
      label: 'Team'
    },

    {
      doc: 'doc4',
      label: 'API'
    },
    {
      page: 'help',
      label: 'Help'
    },

    // {blog: true, label: 'Blog'},
    // { search: true }
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/kats_horizontal.png',
  footerIcon: 'img/cats_logo_banner.png',
  favicon: 'img/kats_horizontal.png',

  /* Colors for website */
  colors: {
    // primaryColor: '#896913',
    // secondaryColor: '#5f490d',
    //  primaryColor: '#7596c8',
    primaryColor: '#3b5998',
    secondaryColor: '#1990ba',
  },
  //  algolia: {
  // apiKey: 'c1171163ec5f45a7d2f1d64daa4b0206',
  // indexName: 'PIDH99YAQ0',
  // algoliaOptions: {}, // Optional, if provided by Algolia
  //  apiKey: '865c35fd67cad216aee1caf2e3c3e2fe',
  //  indexName: 'demo_ecommerce',
  //  algoliaOptions: {},
  // },

  /* Custom fonts for website */

  fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },


  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright © ${new Date().getFullYear()} Kats Project @ Facebook`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ['https://buttons.github.io/buttons.js'],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/undraw_online.svg',
  twitterImage: 'img/undraw_tweetstorm.svg',

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  enableUpdateBy: true,

  // Show documentation's last update time.
  enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',
};

module.exports = siteConfig;
