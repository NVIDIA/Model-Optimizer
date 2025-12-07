# Model Optimizer End-to-End Tests

This directory contains end-to-end tests for the Model Optimizer's [examples](../../examples/).

## Adding new tests

To add a test for a new example, create a new example directory in `tests/examples` following the existing examples as guidance.
Make sure to use as small models and less data as possible to keep the tests fast. Unless needed, have tests finish under 15 minutes.

## Running the tests

To run a test, start from the recommended docker image from our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html).
Then mount your local modelopt directory to `/workspace/Model-Optimizer` and run this from the root of the repository.

```bash
cd /workspace/Model-Optimizer
pip install -e ".[all,dev-test]"
pytest tests/examples/$TEST
```

## Environment variables

The following environment variables can be set to control the behavior of the tests:

- `MODELOPT_LOCAL_MODEL_ROOT`: If set, the tests will use the local model directory instead of downloading the model from the internet. Default is not set, which means the model will be downloaded.
