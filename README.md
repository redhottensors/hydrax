# Hydrax 🐉
A zero-copy multiprocess dataloader for Jax. Built for [Project Redrocket](https://huggingface.co/RedRocket/).

## Installation

Ensure you have JAX installed. If you are using the ``hydrax.tqdm`` module, ensure
[tqdm](https://github.com/tqdm/tqdm) is installed as well.

A pip wheel may be coming in the future.
For now, clone the repository and use the horribly deprecated ``python setup.py install`` in your venv.
You will need a C compiler for the extension module, as well as the appropriate python C header files.

## Usage

```python
from hydrax import Dataloader, DataGroup

def my_loader(data, arrays, seed):
    # load data from data source into arrays, optionally augmenting using 'seed'.
    # if 'seed' is None this is data from a validation batch
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

### Batch Structure

In Hydrax, a single Dataloader is usually responsible for producing both your training and validation batches,
in order to conserve resources and ensure perfectly smooth loading throughout.

Each batch produced by the ``Dataloader`` is either a ``TrainingBatch`` instance or a ``ValidationBatch`` instance,
which both inherit the common functionality of ``Batch``.

The most important properties of a ``Batch`` are:
* ``arrays`` -- ``{ 'array_name': jax.Array, ... }``, corresponding to each array defined by the source ``DataGroup``.
    The first dimension of the array is the batch size.
* ``additional`` -- ``{ 'key': [item_0_value, ...] }``, corresponding to any additional data returned by your loader
    function. Each list's ``len`` is the batch size. If no corresponding item was returned, the element is ``None``.
* ``data`` -- A proxy type to the original data descriptors for each item, with length equal to the batch size.

### Loader Processes

Read the documentation for your ``loader_func`` carefully. If you receive a warning from Hydrax about
your loader, you should fix your code. Failure to do this could result in your batch data changing out
from underneath you, leading to significant training issues such as NaNs.

Do not attempt to construct a Dataloader inside a loader process. Ensure your training code is guarded
with ``if __name__ == '__main__':``, or is otherwise prevented from running. As a last resort, you can
check ``hydrax.is_worker`` and bail.

### KeyboardInterrupt (Ctrl+C / SIGINT)

The Dataloader installs a handler for ``KeyboardInterrupt`` (Ctrl+C / SIGINT) which stops the flow of
batches as soon as possible. After the dataloader has completed, you can check if this occurred by
reading its ``interrupted`` property. You may want to save a checkpoint along with the numbers of the current
epoch and batch, so that you can resume from where you left off with ``start_at``.

If you send a second ``KeyboardInterrupt``, Hydrax will raise a ``KeyboardInterrupt`` at the beginning
of the next batch. This exception may cause you to lose progress unless you or a framework takes care
to save a checkpoint in response.

If you send a third ``KeyboardInterrupt``, the Python interpreter is immediately stopped and control is
returned to you. You will lose all progress since the last checkpoint.

## Documentation

For HTML documentation, run ``make html`` in ``/docs`` and browse the Sphinx documentation at
``/docs/_build/html/index.html``. You will need ``pip install furo``, which should also install Sphinx.

## License

Hydrax is available under the terms of the
[Mozilla Public License, version 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
