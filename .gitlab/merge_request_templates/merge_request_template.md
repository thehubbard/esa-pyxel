# Merge request check list

This is just a reminder about the most common mistakes.
Please make sure that you tick all *appropriate* boxes.
But please read our [contribution guide](https://esa.gitlab.io/pyxel/doc/contributing.html) 
at least once, it will save you unnecessary review cycles !

If an item doesn't apply to your merge request, **check it anyway** to 
make it apparent that there's nothing to do.

 
 - [ ] Closes issue #xxxx
 - [ ] Added **tests** for change code
 - [ ] Updated **documentation** for changed code
 - [ ] Documentation `.rst` files is writtend using [semantic newlines](https://sembr.org)
 - [ ] User visible changes (including notable bug fixes and possible deprecations) are 
       documented in `CHANGELOG.rst`
 - [ ] Passes in this order
   - [ ] isort .
   - [ ] black .
   - [ ] blackdoc .
   - [ ] mypy . 
   - [ ] flake8 .

If you have *any* questions of the points above, just **submit and ask**!
This checklist is here to *help* you, not to deter you from contributing !
