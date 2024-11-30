/* eslint-env node */
require('@rushstack/eslint-patch/modern-module-resolution')

module.exports = {
  root: true,
  'extends': [
    'plugin:vue/vue3-essential',
    'eslint:recommended',
    '@vue/eslint-config-typescript'
  ],
  // "rules": {
  //   "max-line-length": {
  //     "options": [160]
  //   },
  //   "semicolon":[true, "never", "ignore-interfaces"],
  //   "indent": [true, "spaces", 4],
  //   "no-console": false,
  //   "member-access": [false],
  //   "variable-name": false,
  //   "interface-name": false,
  //   "prefer-const": false,
  //   "no-string-literal": false,
  //   "no-empty": false,
  //   "arrow-parens": false,
  //   "triple-equals": false,
  //   "no-consecutive-blank-lines": false,
  //   "object-literal-sort-keys": false,
  //   "ordered-imports": false,
  //   "quotemark": false,
  //   "object-literal-key-quotes": false,
  //   "prefer-for-of": false
  // },
  parserOptions: {
    ecmaVersion: 'latest'
  }
}
