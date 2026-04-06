# Internal Development & Package Creation

This document outlines how to build, test, and package the
`litert-lm-builder` Python tools.

## Project Structure

- `litertlm_builder.py`: Core logic for building LiteRT-LM files.
- `litertlm_builder_cli.py`: Command-line interface for the builder.
- `litertlm_peek.py`: Core logic for inspecting LiteRT-LM files.
- `litertlm_peek_main.py`: Command-line interface for the peek tool.
- `pyproject.toml` / `setup.py`: Build configurations.
- `bundle_pypi_package.sh`: Script to bundle the package into a PyPI-ready wheel.

## Building and Packaging

To build the package and create a `.whl` distribution, use the helper script:

```bash
./bundle_pypi_package.sh
```

### What happens during the build?

1. **Staging**: Files are copied into a temporary directory (`/tmp/litertlm_builder/dist/`).
2. **Bazel Bindings**: It delegates to Bazel to generate Protobuf (`.pb2`) and FlatBuffer bindings for the LiteRT-LM schemas.
3. **Import Rewriting**: It searches and rewrites Protobuf/FlatBuffer internal imports so they match the expected flattened packaged namespace.
4. **Wheel Generation**: Finally, it invokes `uv build` (or standard `pip` / `build`) under PEP 517 to generate a self-contained `.whl` artifact.

## Testing locally

Once built, you can test the wheel by creating a virtual environment, installing
the wheel directly, and verifying the CLIs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install /tmp/litertlm_builder/dist/*.whl

litertlm-builder --help
litertlm-peek --help
```
