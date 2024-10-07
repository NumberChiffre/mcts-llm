#!/bin/bash
pytest -v -s --cov=mcts_llm --cov-report=xml --cov-config=pyproject.toml
