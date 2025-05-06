SHELL := /bin/bash


PYTHON_VERSION := 3.11.3
PYENV_VERSION_INSTALLED := $(shell pyenv versions --bare | grep -e "^$(PYTHON_VERSION)$$")

.PHONY: setup
setup:
	@if [ -z "$(PYENV_VERSION_INSTALLED)" ]; then \
		echo "Installing Python $(PYTHON_VERSION) with pyenv..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	pyenv local $(PYTHON_VERSION)
	@echo "=========|| The pyenv has successfully curated Python version $(PYTHON_VERSION) locally ||========="
	python -m venv .venv 
	@echo "=========|| The environment has been curated successfully ||========="
	.venv/bin/python -m pip install --upgrade pip 
	@echo "=========|| The pipe has been updated successfully ||========="
	.venv/bin/python -m pip install -r requirements.txt 
	@echo "=========|| The requirements have been installed successfully ||========="

