# Hydrax 🐉
A zero-copy multiprocess dataloader for [JAX](https://jax.readthedocs.io/en/latest/).
Built for [Project RedRocket](https://huggingface.co/RedRocket/).

## Installation

``pip install hydrax``

If you want to use any of the extra modules,
[hydrax.image](https://redhottensors.github.io/hydrax/_autosummary/hydrax.image.html),
[hydrax.tqdm](https://redhottensors.github.io/hydrax/_autosummary/hydrax.tqdm.html), or
[hydrax.pandas](https://redhottensors.github.io/hydrax/_autosummary/hydrax.pandas.html),
you can use ``pip install hydrax[image,tqdm,pandas]`` (or ``pip install hydrax[all]``).

Wheels are currently available for Linux x86_64 with CPython 3.10, 3.11, and 3.12

Ensure you have JAX installed and working. If you install via pip, the latest version of JAX will be installed if it is
not already, but jaxlib will not be.

### From Source

Clone this repository, install Python development files and a C compiler, and run:

```sh
source path/to/your/venv/bin/activate
python -m build --wheel
pip install 'dist/hydrax-<...>.whl'
```

## Documentation

Read the online documentation for the latest version at
[https://redhottensors.github.io/hydrax/](https://redhottensors.github.io/hydrax/_autosummary/hydrax.html).

For local HTML documentation, run ``make html`` in ``/sphinx`` and browse the generated Sphinx documentation in
``_build/html/index.html``. You will need ``pip install furo``, which should also install Sphinx.

## Usage

```python
from hydrax import Dataloader, DataGroup, TrainingBatch, ValidationBatch

def my_loader(data, arrays, seed):
    # load data from data source into arrays, optionally augmenting using 'seed'.
    # if 'seed' is None this is a data from a validation batch
    # return any additional data for the batch

if __name__ == "main":
    my_data = ...
    array_defs = {
        "array_name": ((dim, ...), numpy_dtype, jax_dtype),
        ...
    }

    train = DataGroup(batch_size, my_data[1000:], loader_arrays=array_defs)
    valid = DataGroup(batch_size, my_data[:1000], loader_arrays=array_defs)

    dataloader = Dataloader(
        my_loader,
        train,
        validation = ("epoch", 1, valid), # run validation after every epoch
        end_at = ("epoch", 5)             # run 5 epochs in total
    )

    with dataloader: # a with block is required
        # consider using hydrax.tqdm.tbatches instead of a vanilla for loop here
        for batch in dataloader:
            if isinstance(batch, TrainingBatch):
                run_training_batch(batch)
            elif isinstance(batch, ValidationBatch):
                run_validation_batch(batch)

            del batch # important, release batch before waiting for next one or cleaning up

# with hydrax.tqdm.tbatches
    from hydrax.tdqm import tbatches

    for batch in tbatches(dataloader, report_interval=1000): # tbatches includes a with block for the dataloader
        ...
        del batch # important, see above
```

### Deadlocks / Stalls

If you are experiencing deadlocks as a result of retaining batch or array references between iterations, consider using
[debug_batch_references](https://redhottensors.github.io/hydrax/_autosummary/hydrax.debug_batch_references.html) or
[gc.get_referrers](https://docs.python.org/3/library/gc.html#gc.get_referrers) to find out what's holding on to your
batches, though do keep in mind that JAX dispatch will retain references while running ahead. You can check your work by
running the Dataloader with ``depth = 1``, which will immediately deadlock if the first batch is not properly released.

### Batch Structure

In Hydrax, a single Dataloader is usually responsible for producing both your training and validation batches,
in order to conserve resources and ensure perfectly smooth loading throughout.

Each batch produced by the [Dataloader](https://redhottensors.github.io/hydrax/_autosummary/hydrax.Dataloader.html) is
either a [TrainingBatch](https://redhottensors.github.io/hydrax/_autosummary/hydrax.TrainingBatch.html) instance or a
[ValidationBatch](https://redhottensors.github.io/hydrax/_autosummary/hydrax.ValidationBatch.html) instance, which both
inherit the common functionality of [Batch](https://redhottensors.github.io/hydrax/_autosummary/hydrax.Batch.html).
(You can click any of the preceding links to view the
[online documentation](https://redhottensors.github.io/hydrax/_autosummary/hydrax.html).)

The most important properties of a ``Batch`` are:
* ``arrays`` -- ``{ 'array_name': jax.Array, ... }``, corresponding to each array defined by the source ``DataGroup``.
    The first dimension of the array is the batch size.
* ``additional`` -- ``{ 'key': [item_0_value, ...] }``, corresponding to any additional data returned by your loader
    function. Each list's ``len`` is the batch size. If no corresponding item was returned, the element is ``None``.
    Use ``get_additional(key[, index])`` if your loader sometimes omits returning certain keys.
* ``data`` -- A proxy type to the original data descriptors for each item, with length equal to the batch size.

As mentioned above, remember to release any references to a batch or its arrays as soon as you're done with them.

### Loader Processes

Read the [documentation](https://redhottensors.github.io/hydrax/_autosummary/hydrax.Dataloader.html) for ``loader_func``
carefully. If you receive a warning from Hydrax about your loader, you should fix your code. Failure to do this could
result in your batch data changing out from underneath you, leading to significant training issues such as NaNs.

Do not attempt to construct a Dataloader inside a loader process. Ensure your training code is guarded with
``if __name__ == '__main__':``, or is otherwise prevented from running. As a last resort, you can check
``hydrax.is_worker`` and bail.

### KeyboardInterrupt (Ctrl+C / SIGINT)

The Dataloader installs a handler for ``KeyboardInterrupt`` (Ctrl+C / SIGINT) which stops the flow of batches as soon as
possible. After the dataloader has completed, you can check if this occurred by reading its ``interrupted`` property.
You may want to save a checkpoint along with the numbers of the current epoch and batch, so that you can resume from
where you left off with ``start_at``.

If you send a second ``KeyboardInterrupt``, Hydrax will raise a ``KeyboardInterrupt`` at the beginning of the next
batch. This exception may cause you to lose progress unless you or a framework takes care to save a checkpoint in
response.

If you send a third ``KeyboardInterrupt``, the Python interpreter is immediately stopped and control is returned to you.
You will lose all progress since the last checkpoint.

## Compatibility

A convienient wrapper with [tqdm](https://tqdm.github.io/) progress bars is provided in
[hydrax.tqdm](https://redhottensors.github.io/hydrax/_autosummary/hydrax.tqdm.html). The corresponding extra is
``tqdm``.

ICC-profile-aware 8bbp image loading with [Pillow](https://python-pillow.org/) is provided in
[hydrax.image](https://redhottensors.github.io/hydrax/_autosummary/hydrax.image.html), and support is included for
[Oklab](https://bottosson.github.io/posts/oklab/) as well. The corresponding extra is ``image``.

Compatibility for [Pandas](https://pandas.pydata.org/) datasets is provided by
[hydrax.pandas](https://redhottensors.github.io/hydrax/_autosummary/hydrax.pandas.RowData.html). The corresponding extra
is ``pandas``.

## License

Hydrax is available under the terms of the
[Mozilla Public License, version 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
