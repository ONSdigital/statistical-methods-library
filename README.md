# Statistical Methods Library
The Statistical Methods Library (S.M.L.) is a set of approved statistical
methods.

## Testing 

Unit tests are run when a pull request is open. 

You can also test locally however we advise against as it takes a long time for the tests to run.

However, here is the approach if you want to do it.

Prerequisites: Java 8 needs to be installed.

```
poetry install
pytest
``` 

Note: You may get a 'chispa' not installed error despite it being in the pyproject.toml. If this happens run the following:

```
pip install chispa
pytest
``` 

