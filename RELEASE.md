This is a file documenting the release process (I'd have to look it up every time otherwise).
* Increment version in `setup.py`
* Commit
* Run tests: 
  * `pytest`
* Add tag: 
  * `git tag v0.0.3`
* Push, and push tags: 
  * `git push`
  * `git push --tags`
* Upgrade packaging deps:
  * `python3 -m pip install --user --upgrade setuptools wheel twine`
* Build package:
  * `rm dist/* && python3 setup.py sdist bdist_wheel`
* Upload package
  * `python3 -m twine upload dist/*`
