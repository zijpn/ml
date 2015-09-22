window.MathJax = {
  // http://docs.mathjax.org/en/latest/configuration.html
  messageStyle: 'none',
  extensions: ['tex2jax.js'],
  TeX: {
    extensions: ['AMSmath.js', 'AMSsymbols.js']
  },
  jax: ['input/TeX', 'output/HTML-CSS'],
  tex2jax: {
    inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
    displayMath: [ ['$$', '$$'], ['\\[', '\\]'] ],
    processEscapes: true,
    // https://github.com/mathjax/mathjax-docs/wiki/Hide-render-statusbar
    preview: 'none'
  },
  'HTML-CSS': {
    availableFonts: ['TeX'],
    linebreaks: {
      automatic: true
    },
    scale: '90'
  }
};
