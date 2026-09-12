# Security Policy

## Supported Versions

CCA-Zoo follows [semantic versioning](https://semver.org/). Only the latest
release on PyPI is actively supported with security fixes; users are
encouraged to stay up to date via `uv add --upgrade cca-zoo` / `pip install
--upgrade cca-zoo`.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately via
[GitHub Security Advisories](https://github.com/jameschapman19/cca_zoo/security/advisories/new)
for this repository. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce it (a minimal code sample where possible)
- The affected version(s)

You should receive an initial response within a few days. If the report is
confirmed, we will work on a fix and coordinate disclosure timing with you
before any public release.

## Scope

CCA-Zoo is a research library for canonical correlation analysis and related
multiview methods; it is not designed to process untrusted input. As with
any package that unpickles model objects or loads data from `numpy`/`torch`
files, do not load model checkpoints or datasets from untrusted sources.
