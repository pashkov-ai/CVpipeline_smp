# Code Guidance for Claude Agent

## Python Version & Type Hints
- **Use Python 3.13.12** for all code
- Refer to **library versions** defined in `environment.nobuilds.yaml` when writing code
- **Always use modern-style type hints** (PEP 585 generic types)
  - ✅ `list[str]` instead of `List[str]`
  - ✅ `dict[str, int]` instead of `Dict[str, int]`
  - Use proper function signatures: `def process(data: list[int]) -> dict[str, float]:`
  - Follow PEP 8 naming conventions

## Documentation Standards
- **Document all classes, methods, and functions** with docstrings
- Use Google-style docstrings with:
  - Clear description of purpose
  - `Args:` section with type and description for each parameter
  - `Returns:` section with type and description
  - `Raises:` section if applicable
  - `Example:` section for non-trivial functions

Example:
```python
def create_multiplication_table(n: int, m: int) -> list[list[int]]:
    """Generate an n×m multiplication table.

    Creates a table where element [i][j] contains the product of (i+1) × (j+1),
    making it a natural multiplication table starting from 1×1.

    Args:
        n: Number of rows in the multiplication table.
        m: Number of columns in the multiplication table.

    Returns:
        2D list where each inner list represents a row of the multiplication table.

    Raises:
        ValueError: If n or m is less than 1.

    Example:
        >>> table = create_multiplication_table(3, 3)
        >>> table
        [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        >>> table[2][2]
        9
    """
    if n < 1 or m < 1:
        raise ValueError("Both n and m must be at least 1")

    return [[(i + 1) * (j + 1) for j in range(m)] for i in range(n)]
```


## Code Quality Tools

- **Ruff**: Linting and formatting (rules configured in `pyproject.toml`)

- **Pyright**: Static type checking (configured in project)

- **Pre-commit hooks**: Automatically validate before commits

- Ensure your code passes all checkers before submitting

## Testing

- Write tests using **pytest** for all modules

- Aim for meaningful coverage (use `coverage` tool)

- Write tests in `tests/` folder

[//]: # (## Project Organization)


[//]: # (## Git & Commits)

[//]: # (- Write clear, descriptive commit messages)

[//]: # (- Commit logical units of work)

[//]: # (- Run code quality checks before pushing)

## Additional Resources
- Ignore commented out lines in this file
