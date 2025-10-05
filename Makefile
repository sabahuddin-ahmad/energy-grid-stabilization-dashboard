PYTHON ?= $(shell command -v python3.12 >/dev/null 2>&1 && printf 'python3.12' || printf 'python3')
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_BIN := $(VENV_DIR)/bin/python
ACTIVATE := $(VENV_DIR)/bin/activate

.PHONY: help venv install run clean freeze activate

help:
	@printf "Available targets:\n"
	@printf "  make venv    # create the virtual environment\n"
	@printf "  make install # install or upgrade dependencies\n"
	@printf "  make run     # run app.py inside the virtualenv\n"
	@printf "  make activate # activate the virtualenv in a new shell\n"
	@printf "  make clean   # remove the virtual environment\n"
	@printf "  make freeze  # export pinned dependencies to requirements.txt\n"

venv:
	@if [ -d $(VENV_DIR)/bin ]; then \
		current=$$($(VENV_DIR)/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
		target=$$($(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
		if [ "$$current" != "$$target" ]; then \
			rm -rf $(VENV_DIR); \
			$(PYTHON) -m venv $(VENV_DIR); \
		fi; \
	else \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON_BIN) app.py

clean:
	rm -rf $(VENV_DIR)

freeze: install
	$(PIP) freeze > requirements.txt

activate: venv
	@. $(ACTIVATE); printf 'Virtual environment active (leave with `exit`).\n'; exec $${SHELL:-/bin/sh}
