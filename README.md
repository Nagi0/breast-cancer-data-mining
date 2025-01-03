# breast-cancer-data-mining
Step by step to use poetry library to build the python project:
- Install python 3.10: https://www.python.org/downloads/release/python-3100/
- Run `py -3.10 -m pip install poetry` and the following commands on the root folder of this repository
- Execute the command `py -3.10 -m poetry config virtualenvs.in-project true`
- Execute the command `py -3.10 -m poetry install`.
- Now activate the created virtual environment with `.\.venv\Scripts\activate`
- Finally install a separate library that is required to run orange with `pip install PyQt5 PyQtWebEngine`
 Now everything is set and done. If can't find the python version "3.10.9" specified in the "pyproject.toml" file to install, you can edit changing it to another version and try installing it again, keep in mind that you will have to delete the "poetry.lock" file and the ".venv" folder (if it was generated) before executing the listed commands above to the new specified version.

## Link to SSDP+:
https://github.com/tarcisiodpl/ssdp_plus