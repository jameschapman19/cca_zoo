#!/bin/bash

# Note that any change in this file, adding or removing steps or changing the
# printed messages, should be also reflected in the `get_comment.py` file.

# This script shouldn't exit if a command / pipeline fails
set +e
# pipefail is necessary to propagate exit codes
set -o pipefail

global_status=0

echo -e "### Running black ###\n"
black --check --diff .
status=$?

if [[ $status -eq 0 ]]
then
    echo -e "No problem detected by black\n"
else
    echo -e "Problems detected by black, please run black and commit the result\n"
    global_status=1
fi

echo -e "### Running ruff ###\n"
ruff check --show-source .
status=$?
if [[ $status -eq 0 ]]
then
    echo -e "No problem detected by ruff\n"
else
    echo -e "Problems detected by ruff, please fix them\n"
    global_status=1
fi

echo -e "### Running mypy ###\n"
mypy cca_zoo/
status=$?
if [[ $status -eq 0 ]]
then
    echo -e "No problem detected by mypy\n"
else
    echo -e "Problems detected by mypy, please fix them\n"
    global_status=1
fi

echo -e "### Linting completed ###\n"

if [[ $global_status -eq 1 ]]
then
    echo -e "Linting failed\n"
    exit 1
else
    echo -e "Linting passed\n"
    exit 0
fi