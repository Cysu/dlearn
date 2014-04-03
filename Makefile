all:

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs dlearn
	make html -C docs
