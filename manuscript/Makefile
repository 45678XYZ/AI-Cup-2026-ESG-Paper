TECTONIC ?= tectonic
PYTHON ?= python
BUILD_DIR := build
PDF := $(BUILD_DIR)/main.pdf
LOG := $(BUILD_DIR)/main.log

.PHONY: build check check-final clean

build:
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) main.tex --outdir $(BUILD_DIR) --keep-logs --reruns 2 > $(LOG) 2>&1

check: build
	PYTHONPATH=.. $(PYTHON) check.py --root . --repo-root .. --pdf $(PDF) --log $(LOG)

check-final: build
	PYTHONPATH=.. $(PYTHON) check.py --root . --repo-root .. --pdf $(PDF) --log $(LOG) --final

clean:
	rm -rf $(BUILD_DIR)
