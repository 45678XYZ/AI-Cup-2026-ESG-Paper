TECTONIC ?= tectonic
PYTHON ?= python
SOURCE_DATE_EPOCH ?= 1787443200
BUILD_DIR := build
PDF := $(BUILD_DIR)/main.pdf
LOG := $(BUILD_DIR)/main.log

.PHONY: build check check-final clean

build:
	mkdir -p $(BUILD_DIR)
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
		$(TECTONIC) -Z deterministic-mode main.tex --outdir $(BUILD_DIR) \
		--keep-logs --reruns 2 > $(LOG) 2>&1

check: build
	PYTHONPATH=.. $(PYTHON) check.py --root . --repo-root .. --pdf $(PDF) --log $(LOG)

check-final: build
	PYTHONPATH=.. $(PYTHON) check.py --root . --repo-root .. --pdf $(PDF) --log $(LOG) --final

clean:
	rm -rf $(BUILD_DIR)
