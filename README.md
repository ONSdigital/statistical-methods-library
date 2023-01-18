# Statistical Methods Library
The Statistical Methods Library (S.M.L.) is a set of approved statistical
methods.

## Linting

If you want to run quality or linting tools you can use the following options:

### Black

Black is a PEP 8 compliant opinionated formatter. It reformats entire files in place using the following command.

```
poetry run black .
```
### Isort

Isort is a Python library to sort imports alphabetically, and automatically separated into sections and by type. 

```
poetry run isort .
```
### Flake8

For linting purposes flake8 can be used with the following command.

```
poetry run flake8 statistical_methods_library
```