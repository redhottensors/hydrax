#!/bin/sh

rm -r dist/* wheelhouse/*

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

for wheel in dist/*.whl; do
    auditwheel repair "$wheel" &
done

wait
