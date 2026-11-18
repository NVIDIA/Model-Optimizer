### What does this PR do?

Type of change: ? <!-- Use one of the following: Bug fix, new feature, new example, new tests, documentation. -->

**Problem:** <!-- What is broken, missing, or slow, and for whom? Start here so reviewers see why this PR is needed before reading the diff. -->

**Fix:** <!-- How this PR fixes it, in a sentence or two. Keep root cause and numbers to what supports the why; don't restate the diff file by file. -->

### Modules touched

<!-- Only if the PR changes more than one module (tests and docs that only follow the change don't count); otherwise delete this section. One line per module: why the fix cannot be made without changing it. A module you cannot justify belongs in a separate PR. -->

### Usage

```python
# Add a code snippet demonstrating how to use this
```

### Testing
<!-- Mention how have you tested your change if applicable. -->

### Before your PR is "*Ready for review*"

Make sure you read and follow [Contributor guidelines](https://github.com/NVIDIA/Model-Optimizer/blob/main/CONTRIBUTING.md) and your commits are signed (`git commit -s -S`).

Make sure you read and follow the [Security Best Practices](https://github.com/NVIDIA/Model-Optimizer/blob/main/SECURITY.md#security-coding-practices-for-contributors) (e.g. avoiding hardcoded `trust_remote_code=True`, `torch.load(..., weights_only=False)`, `pickle`, etc.).

- Is this change backward compatible?: ✅ / ❌ / N/A <!--- If ❌, explain why. -->
- If you copied code from any other sources or added a new PIP dependency, did you follow guidance in `CONTRIBUTING.md`: ✅ / ❌ / N/A <!--- Mandatory -->
- Did you write any new necessary tests?: ✅ / ❌ / N/A <!--- Mandatory for new features or examples. -->
- Did you update [Changelog](https://github.com/NVIDIA/Model-Optimizer/blob/main/CHANGELOG.rst)?: ✅ / ❌ / N/A <!--- Very short summary of changes only for new features, backward breaking changes, deprecations, or fixes for critical bugs present in previous releases. -->
- Did you get Claude approval on this PR?: ✅ / ❌ / N/A <!--- Run `/claude review`. NVIDIA org members can self-trigger for complex changes; orthogonal to CodeRabbit. -->

### Additional Information
<!-- E.g. related issue. -->
