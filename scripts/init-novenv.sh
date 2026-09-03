#!/bin/bash

set -e

export PIP_BREAK_SYSTEM_PACKAGES=1

pip install -U poetry
poetry env remove --all
export POETRY_VIRTUALENVS_CREATE=false 
poetry config virtualenvs.create false
poetry install
