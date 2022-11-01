# individual_identification

## Setup

```
python3 -m venv .venv

# Mac
source .venv/bin/activate

# Windows
.venv/Scripts/activate

pip install -U pip

pip install poetry

poetry install & pip install -r requirements.txt

pre-commit install
```

## Usage

- First change the default parameters in the function `coordinate_batching` within iid/workflows/sort_image_library.py to the correct image input and output locations. Also adjust the batch size if required.
- Run the below statement in the command line.

```
# Run the workflow to filter to interesting images.
python iid/workflows/sort_image_library.py
```
