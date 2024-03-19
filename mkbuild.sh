#!/bin/sh

mkdir -p benv

for pyver in "$@"; do
    "python$pyver" -m venv "benv/$pyver"
    (
        . "benv/$pyver/bin/activate"
        pip install build
    )
done
