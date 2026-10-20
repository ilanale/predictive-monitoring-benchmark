"""Access to the (adapted) upstream code from court modules.

Upstream modules live in ``experiments/`` and import each other by bare name, and
``experiments/dataset_confs.py`` registers the court datasets only when ``PPM_COURT_DATA_DIR`` is
set at import time. ``dataset_manager(task, fs, data_dir)`` takes care of both.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path(data_dir):
    os.environ.setdefault("PPM_COURT_DATA_DIR", os.path.abspath(data_dir))
    for p in (ROOT, os.path.join(ROOT, "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)


def dataset_manager(task, fs, data_dir):
    """Upstream ``DatasetManager`` for court dataset ``<task>__<fs>`` (C1-C7 adapted)."""
    _ensure_path(data_dir)
    import dataset_confs
    from DatasetManager import DatasetManager
    name = "%s__%s" % (task, fs)
    if name not in getattr(dataset_confs, "task_type", {}):
        # dataset_confs was imported before PPM_COURT_DATA_DIR was set: register now
        from court.court_confs import register
        register(dataset_confs, os.path.join(data_dir, "court_columns.json"), data_dir, with_sensitivity=True)
    return DatasetManager(name)
