#!/usr/bin/env bash
set -e

cp make.inc.coverage make.inc

make clean
make test
make examples

gcov -rk src/*.cpp

lcov -c --no-external --directory ./src -b . -o tracefile

genhtml -o lcov tracefile
