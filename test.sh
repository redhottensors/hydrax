#!/bin/sh

mkdir -p tenv
rm -r tenv/* || :

for wheel in dist/hydrax-*-cp3*.whl; do
    pyver="$(echo "$wheel" | sed -n 's/.*-cp3\([0-9]*\)-.*/3.\1/p')"
    echo "========== setup python$pyver =========="

    "python$pyver" -m venv "tenv/$pyver"

    (
        . "tenv/$pyver/bin/activate"
        pip install jaxlib
        pip install "$wheel"'[all]'

        echo
        echo "========== test python$pyver =========="

        cd tests
        rm -r _test_cache || :

        for mod in *.py; do
            python "$mod" || exit 1
        done

        echo
    ) || exit 1
done
