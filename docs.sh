#!/bin/sh

cd sphinx || exit
rm -r _build _autosummary
make html

rm -r ../docs/*
cp -r _build/html/* ../docs/
touch ../docs/.nojekyll
