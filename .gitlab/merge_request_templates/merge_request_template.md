# Merge request check list

This is just a reminder about the most common mistakes.
Please make sure that you tick all *appropriate* boxes.
But please read our [contribution guide](https://esa.gitlab.io/pyxel/doc/contributing.html) 
at least once, it will save you unnecessary review cycles !

If an item doesn't apply to your merge request, **check it anyway** to 
make it apparent that there's nothing to do.

 
 - [ ] Closes issue #xxxx
 - [ ] Added [**tests**](https://esa.gitlab.io/pyxel/doc/stable/references/contributing.html#running-the-test-suite) for change code
 - [ ] Updated [**documentation**](https://esa.gitlab.io/pyxel/doc/stable/references/contributing.html#contributing-to-the-documentation) for changed code
 - [ ] Documentation `.rst` files is written using [semantic newlines](https://sembr.org)
 - [ ] User visible changes (including notable bug fixes and possible deprecations) are 
       documented in `CHANGELOG.md`
 - [ ] Passes in this order
   - [ ] `isort .` or `pre-commit run -a` (preferred way)
   - [ ] `black .` or `pre-commit run -a` (preferred way)
   - [ ] `blackdoc .` or `pre-commit run -a` (preferred way)
   - [ ] `mypy .` or `tox -e mypy` or `tox -p` (preferred way)
   - [ ] `flake8 .` or `tox -e flake8` or `tox -p` (preferred way)

If you have *any* questions of the points above, just **submit and ask**!
This checklist is here to *help* you, not to deter you from contributing !
