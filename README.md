# MONET Plots

This is the plotting functionality spun off from the main MONET repository.

## Development

### Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency.

To set up pre-commit locally:

1.  Install pre-commit:
    ```bash
    pip install pre-commit
    ```

2.  Install the git hook scripts:
    ```bash
    pre-commit install
    ```

Now, pre-commit will run automatically on `git commit`. You can also run it against all files manually:

```bash
pre-commit run --all-files
```
