# Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub.**

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:

- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
  - Please include the following information:
    - Product/Driver name and version/branch that contains the vulnerability
    - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
    - Instructions to reproduce the vulnerability
    - Proof-of-concept or exploit code
    - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's [Product Security portal](https://www.nvidia.com/en-us/security).

---

## Security Considerations

### Overview

NVIDIA Model Optimizer (ModelOpt) is a library used to optimize ML models and may load and process user-provided artifacts (models, weights, configs, calibration data) and their dependencies. Secure deployment depends on how you source artifacts, validate inputs, and harden the environment where ModelOpt runs.

### What to Be Aware Of

#### Untrusted model and data inputs

- Models, weights, configs and data may be malicious or corrupted.

#### Deserialization and code-execution risks

- Unsafe deserialization can lead to arbitrary code execution if fed untrusted inputs.
- Avoid using serialization formats/settings that can deserialize arbitrary objects.

#### Input validation and resource exhaustion

- Large or malformed inputs can trigger crashes or excessive CPU/GPU/memory use.
- Missing size/type checks can increase DoS risk.

#### Data in transit and at rest

- If fetching models or dependencies over the network, insecure transport can enable tampering.
- Stored artifacts, logs, and caches may contain sensitive data.

#### Logging and observability

- Logs may inadvertently contain sensitive inputs, paths, tokens, or proprietary model details.
- Overly verbose logs can leak operational and security-relevant information.

#### Supply chain and third-party components

- Dependencies may include known vulnerabilities or be compromised.
- Third-party plugins/components loaded at runtime may not have the same security assurances.

### Example Security Approaches

#### Artifact integrity

- Only load artifacts from trusted sources.
- Prefer signed artifacts; verify signatures before loading.

#### Safe parsing and deserialization

- Prefer safer storage formats (avoid object deserialization for untrusted inputs).
- Avoid `pickle`, `torch.load()` with untrusted weights, or YAML `unsafe_load`.
- Treat any unverified artifact as untrusted and block/guard its loading.

#### Hardening and least privilege

- Run with least privilege and isolate workloads.

#### Data protection

- Encrypt sensitive data at rest; use TLS 1.3 for data in transit.
- Never hardcode or log credentials.

#### Resilience

- Validate inputs and enforce limits (file size, timeouts, quotas, etc.).
- Keep OS, containers, and dependencies patched; scan for known vulnerabilities.

---

## Security Coding Practices for Contributors

ModelOpt processes model checkpoints and weights from various sources. Contributors must avoid patterns that can introduce security vulnerabilities. These rules apply to all code except tests. These rules cover a few key security considerations as follows:

### Deserializing untrusted data

**Do not use `torch.load(..., weights_only=False)`** unless a documented exception is provided. It uses pickle under the hood and can execute arbitrary code from a malicious checkpoint.

```python
# Bad — allows arbitrary code execution from the checkpoint file
state = torch.load(path, weights_only=False)

# Good
state = torch.load(path, weights_only=True, map_location="cpu")

# Acceptable only with an inline comment explaining why weights_only=False
# is required and confirming the file is internally-generated / trusted.
state = torch.load(
    path,
    weights_only=False,  # loaded file is generated internally by ModelOpt and not supplied by the user
    map_location="cpu",
)
```

**Do not use `numpy.load(..., allow_pickle=True)`** unless a documented exception is provided. It uses pickle under the hood and can execute arbitrary code from a malicious checkpoint.

```python
# Bad — allows arbitrary code execution from the checkpoint file
state = numpy.load(path, allow_pickle=True)

# Good - let the caller decide; default to False
def load_data(path: str, trust_data: bool = False):
    return numpy.load(path, allow_pickle=trust_data)
```

**Do not use `yaml.load()`** — always use `yaml.safe_load()`. The default loader can execute arbitrary Python objects embedded in YAML.

### Loading transformers models with `trust_remote_code`

**Do not hardcode `trust_remote_code=True`.** This flag tells Transformers to execute arbitrary Python shipped with a checkpoint, which is an RCE vector if the model source is untrusted.

```python
# Bad — silently opts every user into remote code execution
model = AutoModel.from_pretrained(name, trust_remote_code=True)

# Good — let the caller decide; default to False
def load_model(name: str, trust_remote_code: bool = False):
    return AutoModel.from_pretrained(name, trust_remote_code=trust_remote_code)
```

### Subprocess and shell commands

**Never use `shell=True` with string interpolation or user-supplied input.** This is a command-injection vector.

```python
# Bad — command injection if model_name contains shell metacharacters
subprocess.run(f"python convert.py --model {model_name}", shell=True)

# Good — pass arguments as a list
subprocess.run(["python", "convert.py", "--model", model_name])
```

### Other patterns to avoid

- **`eval()` / `exec()`** on strings derived from external input. If you must generate and execute code dynamically, validate the input against an allowlist of safe patterns.
- **Hardcoded secrets or credentials** — never commit tokens, passwords, or API keys. Use environment variables or config files listed in `.gitignore`.

### Bandit security checks

Bandit is used as a pre-commit hook to check for security-sensitive patterns in the code. **`# nosec` comments are not allowed** as a bypass for security checks.

### Creating a security exception

If a security-sensitive pattern (e.g. `pickle`, `subprocess`) is genuinely required, the contributor must:

1. **Add an inline comment** explaining *why* the pattern is necessary and *why* it is safe in this specific context (e.g. "loaded file is generated internally by ModelOpt").
1. **Request review from [@NVIDIA/modelopt-setup-codeowners](https://github.com/orgs/NVIDIA/teams/modelopt-setup-codeowners)** and include a clear justification in the PR description.
