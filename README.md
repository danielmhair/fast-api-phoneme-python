# Fast API Phoneme Python

Phoneme AI trained on children voices using FastAPI in Python

## Project Structure

```
fast-api-phoneme-python
├── src
│   ├── __init__.py
│   └── main.py
├── tests
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── .flake8
├── pytest.ini
└── README.md
```

## Setup

To set up the project, follow these steps:

1. Create a new directory called `fast-api-phoneme-python`.
2. Inside the `fast-api-phoneme-python` directory, create a `src` directory.
3. Inside the `src` directory, create an empty file called `__init__.py`.
4. Inside the `src` directory, create a file called `main.py` and add your main Python code.
5. Inside the `fast-api-phoneme-python` directory, create a `tests` directory.
6. Inside the `tests` directory, create an empty file called `__init__.py`.
7. Inside the `tests` directory, create a file called `test_main.py` and add your unit tests using the `pytest` framework.
8. Create a file called `requirements.txt` and list the dependencies for your project.
9. Create a file called `.flake8` and configure the linter settings.
10. Create a file called `pytest.ini` and configure the `pytest` settings.
11. Create a file called `README.md` and add the documentation for your project.

## Usage

To run the project, you can use the following commands:

- Install the project dependencies:
  ```
  pip install -r requirements.txt
  ```

- Run in 2 terminals
  - cd src2 && uvicorn app:app --reload
  - cd src && uvicorn phoneme_api:app --reload
  - python main.py [name] [character_sound] [amount_of_tries]
