#!/usr/bin/env node
// Runs eslint over the given files with the coding-standards config.
//
// Resolve the config beside this script so repository configs cannot override it
// and plugin imports resolve within this package.

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { ESLint } from "eslint";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const configFile = join(packageRoot, "eslint.config.mjs");

const paths = process.argv.slice(2);
if (paths.length === 0) {
  console.error("usage: code-check-eslint <files...>");
  process.exit(2);
}

const eslint = new ESLint({
  overrideConfigFile: configFile,
  errorOnUnmatchedPattern: false,
});

const results = await eslint.lintFiles(paths);
const formatter = await eslint.loadFormatter("stylish");
const report = await formatter.format(results);

if (report) {
  console.log(report);
}

// Warnings are advisory, matching eslint's own CLI, so only errors fail the run.
const errorCount = results.reduce((total, result) => total + result.errorCount, 0);
process.exit(errorCount > 0 ? 1 : 0);
