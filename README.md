## Word-Level Coreference Resolution was reused from [here](https://github.com/vdobrovolskii/wl-coref) under MIT licence


# Development

## Pre-Commits

Firstly, make sure that the [pre-commit](https://pypi.org/project/pre-commit/)
library is installed.
Then, run

```sh
# For each time one wants to run the pipeline without committing
> ./pre-commit.sh
```

The **pre-commit** routine is also performed automatically whenever one
attempts to commit anything. The commit will fail if one of the hooks fails.
Therefore, manual runs of the *sh* script are not necessary.

To tweak the hooks of **pre-commit**, edit
[.pre-commit-config.yaml](.pre-commit-config.yaml).
The configuration of the [mypy](https://mypy.readthedocs.io/en/stable/)
static type checker hook can be found in [mypy.ini](mypy.ini).
