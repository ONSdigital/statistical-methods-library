## Testing 

Unit tests are run when a pull request is open.

Prerequisites: Java needs to be installed.

```
poetry run pytest -x
``` 

Note: You may get a 'chispa' not installed error despite it being in the pyproject.toml. If this happens run the following:

```
cd ..
pip install chispa
cd tests/
poetry run pytest -x
``` 

