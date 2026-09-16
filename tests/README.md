# Psifx Tests

This directory contains tests for the psifx library using pytest.

## Directory Structure

The test directory structure is organized as follows:

```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests @integration
└── test_structure.py # Tests for the structure of psifx @structure
```

- **Unit Tests**: Test individual functions and classes in isolation, often using mocks for dependencies.
- **Integration Tests**: Test how different components of the library work together, using real data and minimal
  mocking.

Integration tests are marked with the `@pytest.mark.integration` decorator.

## Running Tests

To run only integration tests:

```bash
pytest -m "integration"
```

To run tests with coverage report:

```bash
pytest --cov=psifx
```

To run tests for a specific module:

```bash
pytest tests/integration/io/
```
## Pose accuracy on COCO

`tests/integration/video/test_pose_coco_accuracy.py` runs each pose tool through the real CLI on 8 COCO images
(`tests/integration/data/CocoPose`) and checks its OKS / PCK@0.2 against a minimum score. This is the common
yardstick for the model and optimization branches: a new tool registers its COCO-17 adapter in
`research/benchmark/psifx_eval.py` and its minimum scores in `MIN_SCORES`.

```bash
pytest tests/integration/video/test_pose_coco_accuracy.py -s   # -s prints the measured scores
```

Tests under `tests/unit/research/` cover the benchmark metrics; `tests/conftest.py` puts `research/benchmark` on
`sys.path` for them.
