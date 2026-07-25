# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile i18n-update i18n-check i18n-build i18n-all

i18n-update:
	@$(PYTHON) ../tools/docs/i18n.py update

i18n-check:
	@$(PYTHON) ../tools/docs/i18n.py check

i18n-build:
	@$(PYTHON) ../tools/docs/i18n.py build

i18n-all:
	@$(PYTHON) ../tools/docs/i18n.py all

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
