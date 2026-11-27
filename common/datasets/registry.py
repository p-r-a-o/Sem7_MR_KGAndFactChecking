"""Maps dataset name -> DatasetSpec. Adding a dataset = adding a spec module and one line here."""
from common.datasets import factkg, fb15k237, wn18rr
from common.datasets.files import DatasetSpec

SPECS: dict[str, DatasetSpec] = {s.name: s for s in (fb15k237.SPEC, wn18rr.SPEC, factkg.SPEC)}


def get_spec(dataset: str) -> DatasetSpec:
    try:
        return SPECS[dataset]
    except KeyError:
        raise KeyError(f"Unknown dataset {dataset!r}; expected one of {sorted(SPECS)}")
