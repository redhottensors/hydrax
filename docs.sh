#!/bin/sh

cd sphinx
make html

rm -r ../docs/*
cp -r _build/html/* ../docs/
