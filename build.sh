#!/bin/sh

(
    . "venv/bin/activate"
    python -m build --sdist
) &

for venv in benv/*; do
    (
        . "$venv/bin/activate"
        python -m build --wheel
    ) &
done

wait
