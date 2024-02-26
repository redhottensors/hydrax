# Hydrax 🐉
A zero-copy multiprocess dataloader for Jax.

## Installation

Ensure you have JAX installed.
If you are using the ``hydrax.tqdm`` module, ensure [tqdm](https://github.com/tqdm/tqdm) is installed as well.

A pip wheel may be coming in the future.
For now, clone the repository and use the horribly deprecated ``python setup.py install`` in your venv.
You will need a C compiler for the extension module, as well as the appropriate python C header files.

## Usage

```python
from hydrax import Dataloader, DataGroup

def my_loader(data, arrays, seed):
    # load data from data source into arrays, optionally augmenting using 'seed'.
    # if 'seed' is None this is a validation batch
    # return any additional data for the batch

if __name__ == "main":
    my_data = ...
    array_defs = {
        "array_name": ((dim, ...), dtype),
        ...
    }

    all_data = DataGroup(batch_size, array_defs, my_data)
    valid, train = all_data.split(1000) # or however many validation items to reserve

    dataloader = Dataloader(
        my_loader,
        train,
        valid = ("epoch", 1, valid), # run validation after every epoch
        end_at = ("epoch", 5)        # run 5 epochs in total
    )

# with a vanilla for loop
    with dataloader: # a with block is required
        for batch in dataloader:
            if batch['_validation']:
                run_validation_batch(batch)
            else:
                run_training_batch(batch)

# with hydrax.tqdm.tbatches
    from hydrax.tdqm import tbatches

    with dataloader:
        for batch in tbatches(dataloader, report_interval=1000):
            ...
```

A more detailed example can be seen in ``test.py``.

## Documentation

For HTML documentation, run ``make html`` in ``/docs`` and browse the Sphinx documentation at ``/docs/_build/html/index.html``.
You will need ``pip install sphinx``.

## License

Hydrax is available under the terms of the [Mozilla Public License, version 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
