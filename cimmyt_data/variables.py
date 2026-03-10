"""Load variable metadata from `variables_details.csv` and provide helpers.

This module exposes `load_variable_metadata()` which returns an ordered
dict mapping variable name -> {description,type,unit}.
"""
from pathlib import Path
import csv
from collections import OrderedDict
from typing import Dict


def load_variable_metadata(csv_path: str = None) -> Dict[str, Dict[str, str]]:
    base = Path(__file__).resolve().parent
    csv_path = Path(csv_path) if csv_path else base / 'variables_details.csv'
    meta = OrderedDict()
    if not csv_path.exists():
        return meta

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Variable') or row.get('variable')
            if not name:
                continue
            meta[name] = {
                'description': row.get('Description', '').strip(),
                'type': row.get('Type', '').strip(),
                'unit': row.get('Unit', '').strip(),
            }

    return meta


# Load once on import for convenience
VARIABLE_METADATA = load_variable_metadata()


def get_variable_description(name: str) -> str:
    return VARIABLE_METADATA.get(name, {}).get('description', '')


def get_variable_info(name: str) -> Dict[str, str]:
    return VARIABLE_METADATA.get(name, {})


__all__ = ['load_variable_metadata', 'VARIABLE_METADATA', 'get_variable_description', 'get_variable_info']
