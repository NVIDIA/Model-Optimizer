Security Considerations
=======================

Overview
--------

NVIDIA Model Optimizer (ModelOpt) is a library used to optimize ML models and
may load and process user-provided artifacts (models, weights, configs,
calibration data) and their dependencies. Secure deployment depends on how you
source artifacts, validate inputs, and harden the environment where ModelOpt
runs.

What to Be Aware Of
-------------------

**Untrusted model and data inputs**

- Models, weights, configs and data may be malicious or corrupted.

**Deserialization and code-execution risks**

- Unsafe deserialization can lead to arbitrary code execution if fed untrusted
  inputs.
- Avoid using serialization formats/settings that can deserialize arbitrary
  objects.

**Input validation and resource exhaustion**

- Large or malformed inputs can trigger crashes or excessive CPU/GPU/memory use.
- Missing size/type checks can increase DoS risk.

**Data in transit and at rest**

- If fetching models or dependencies over the network, insecure transport can
  enable tampering.
- Stored artifacts, logs, and caches may contain sensitive data.

**Logging and observability**

- Logs may inadvertently contain sensitive inputs, paths, tokens, or proprietary
  model details.
- Overly verbose logs can leak operational and security-relevant information.

**Supply chain and third-party components**

- Dependencies may include known vulnerabilities or be compromised.
- Third-party plugins/components loaded at runtime may not have the same
  security assurances.

Example Security Approaches
---------------------------

**Artifact integrity**

- Only load artifacts from trusted sources.
- Prefer signed artifacts; verify signatures before loading.

**Safe parsing and deserialization**

- Prefer safer storage formats (avoid object deserialization for untrusted
  inputs).
- Avoid ``pickle``, ``torch.load()`` with untrusted weights, or YAML
  ``unsafe_load``.
- Treat any unverified artifact as untrusted and block/guard its loading.

**Hardening and least privilege**

- Run with least privilege and isolate workloads.

**Data protection**

- Encrypt sensitive data at rest; use TLS 1.3 for data in transit.
- Never hardcode or log credentials.

**Resilience**

- Validate inputs and enforce limits (file size, timeouts, quotas,..).
- Keep OS, containers, and dependencies patched; scan for known vulnerabilities.
