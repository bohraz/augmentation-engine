# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
MKDIR_P       = mkdir -p
SPHINXPROJ    = augmentation_engine
SOURCEDIR     = .
BUILDDIR      = _build
STATIC        = _static
TEMPLATES     = _templates
FILESTOBEREMOVED := $(shell find *.rst -not -name "index.rst")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@${MKDIR_P} $(BUILDDIR) $(TEMPLATES) $(STATIC)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -f ${FILESTOBEREMOVED}
	rm -rf $(BUILDDIR) $(TEMPLATES) $(STATIC)