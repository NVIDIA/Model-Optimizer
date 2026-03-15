# Experimental Optimization Techniques

Experimental optimization algorithms and research prototypes under active development.

## Purpose

For new optimization techniques (quantization, pruning, sparsity, etc.) that are:

- Novel or research-stage algorithms
- Not yet production-ready
- May have unstable APIs

**⚠️ Warning**: Experimental features are not guaranteed to work across releases. APIs may change or features may be removed without notice. Use at your own risk.

## Requirements

Each experimental technique must include:

- **README.md** - Explains what the technique does, how to use it, current status, model support, and references
- **Working code** - Clear, readable implementation
- **Comprehensive tests** - Good test coverage demonstrating correctness
- **Detailed documentation** - Clear docs on usage, APIs, and behavior
- **Example** - Demonstrating usage
- **Model support list** - Which models/frameworks are supported
- **Deployment info** - Supported deployment frameworks (TensorRT-LLM, vLLM, SGLang, etc.) and whether custom kernels are required
- **requirements.txt** - Additional dependencies beyond base modelopt
- **License headers** - Apache 2.0 headers on all Python files

## Example Structures

Organize your code however makes sense. Here are some examples:

**Simple flat structure:**

```text
experimental/my_technique/
├── README.md
├── requirements.txt
├── my_technique.py
├── test_my_technique.py
└── example.py
```

**Package structure:**

```text
experimental/my_technique/
├── README.md
├── requirements.txt
├── my_technique/
│   ├── __init__.py
│   ├── core.py
│   └── config.py
├── tests/
│   └── test_core.py
└── examples/
    └── example_usage.py
```

## Quality Standards

Experimental code must meet quality standards:

- Comprehensive test coverage required
- Clear documentation required
- Pass all pre-commit checks

## PR Guidelines

Keep PRs focused and reviewable:

- **Split large features**: Break complex techniques into multiple PRs if needed
- **Reasonable scope**: PRs with tens of thousands of lines are difficult to review
- **Incremental development**: Consider submitting core functionality first, then enhancements
- If your technique is large, discuss the implementation plan in an issue first

## Example Documentation Template

Your technique's README.md should include:

```markdown
# Your Technique Name

Brief description of the optimization technique.

## Model Support

| Model/Framework | Supported | Notes |
|-----------------|-----------|-------|
| LLMs (Llama, GPT, etc.) | ✅ | Tested on Llama 3.1 |
| Diffusion Models | ❌ | Not yet supported |
| Vision Models | ✅ | Experimental |

## Deployment

| Framework | Supported | Notes |
|-----------|-----------|-------|
| TensorRT-LLM | ✅ | Requires custom kernel |
| vLLM | ❌ | Not yet supported |
| SGLang | ✅ | Uses standard ops |

## Usage

\`\`\`python
from experimental.my_technique import my_optimize
...
\`\`\`

## Status

Current state: Prototype

Known issues:
- Issue 1
- Issue 2

## References

- [Paper](link)
- [Code repository](link)
- [Project page](link)
- [Related work](link)
```

## Path to Production

When a technique is ready for production (proven effective, stable API, full tests, comprehensive docs), it can be promoted to the main `modelopt` package.

**Contributors**: Open an issue proposing graduation with evidence of effectiveness and stability.

**Users**: If you find an experimental feature valuable, open a GitHub issue requesting promotion to production. User demand is a key signal for production readiness.

## Questions?

Open a GitHub issue with `[experimental]` prefix.
