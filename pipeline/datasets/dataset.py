"""
PyTorch Dataset classes for SICKLE++ multi-sensor satellite data.

Supports:

This module integrates CIMMYT variable metadata when available. Use
`cimmyt_data.variables.VARIABLE_METADATA` to access per-variable descriptions,
types and units (loaded from `cimmyt_data/variables_details.csv`).
"""
from typing import Dict, Any

try:
	from cimmyt_data.variables import VARIABLE_METADATA, get_variable_info
except Exception:
	VARIABLE_METADATA = {}
	def get_variable_info(name: str) -> Dict[str, Any]:
		return {}


def get_metadata_for_columns(columns):
	"""Return a dict of metadata for the provided column list."""
	return {c: get_variable_info(c) for c in columns}


# Minimal PyTorch Dataset implementations to keep the pipeline import-safe.
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional


class SICKLEDataset(Dataset):
	"""Flexible dataset for SICKLE++.

	Supports two initialization modes:
	1. Tabular: pass a pandas DataFrame as `df` (legacy behaviour).
	   SICKLEDataset(df=..., feature_cols=[...], label_col='yield_kg_ha')

	2. Satellite directory + labels: pass `satellite_data_dir` (directory
	   containing .npy per-field arrays) and `labels_path` (CSV/NPY with
	   labels). The dataset will attempt to match files to rows using
	   a `Merged_ID` column or fallback to index alignment.
	"""

	def __init__(self,
				 df: Optional[pd.DataFrame] = None,
				 feature_cols: Optional[List[str]] = None,
				 label_col: Optional[str] = None,
				 satellite_data_dir: Optional[str] = None,
				 labels_path: Optional[str] = None,
				 task_type: str = 'crop_classification',
				 normalize: bool = True,
				 cache_in_memory: bool = False,
				 transform=None):

		self.transform = transform
		self.task_type = task_type
		self.normalize = normalize
		self.cache_in_memory = cache_in_memory

		# Mode A: DataFrame provided
		if df is not None and isinstance(df, pd.DataFrame):
			self.mode = 'table'
			self.df = df.reset_index(drop=True)
			self.feature_cols = feature_cols or [c for c in self.df.columns if c != label_col]
			self.label_col = label_col
			self.files = None

		# Mode B: Satellite data directory + labels
		elif satellite_data_dir is not None and labels_path is not None:
			self.mode = 'satellite'
			self.satellite_data_dir = Path(satellite_data_dir)
			# load labels
			if labels_path.endswith('.npy'):
				labels_arr = np.load(labels_path, allow_pickle=True)
				# if structured array or dict-like
				try:
					self.df = pd.DataFrame(labels_arr)
				except Exception:
					self.df = pd.DataFrame({'label': labels_arr})
			else:
				self.df = pd.read_csv(labels_path)

			# discover .npy files
			self.files = sorted([p for p in self.satellite_data_dir.glob('*.npy')])

			# attempt to match by Merged_ID
			if 'Merged_ID' in self.df.columns:
				id_to_row = {str(r['Merged_ID']): i for i, r in self.df.iterrows()}
				matched_files = []
				for f in self.files:
					fid = f.stem
					# if filename contains the id, map; else try exact match
					found = None
					for key in id_to_row:
						if key in fid:
							found = (f, id_to_row[key])
							break
					if found:
						matched_files.append(found)

				if matched_files:
					# reorder df to match files
					files_sorted, rows = zip(*matched_files)
					self.files = list(files_sorted)
					self.df = self.df.iloc[list(rows)].reset_index(drop=True)

			# fallback: ensure equal lengths by trimming to shorter
			if len(self.files) != len(self.df):
				n = min(len(self.files), len(self.df))
				self.files = self.files[:n]
				self.df = self.df.iloc[:n].reset_index(drop=True)

			# default label column heuristics
			if label_col is None:
				if 'yield_kg_ha' in self.df.columns:
					self.label_col = 'yield_kg_ha'
				else:
					# pick numeric column if available
					numcols = self.df.select_dtypes(include=[float, int]).columns.tolist()
					self.label_col = numcols[0] if numcols else None

		else:
			raise ValueError('Invalid SICKLEDataset initialization. Provide either df or satellite_data_dir + labels_path')

		# simple in-memory cache
		self._cache = {} if self.cache_in_memory else None

	def __len__(self):
		if self.mode == 'table':
			return len(self.df)
		else:
			return len(self.files)

	def __getitem__(self, idx):
		if self.mode == 'table':
			row = self.df.iloc[idx]
			features = row[self.feature_cols].to_numpy() if self.feature_cols is not None else row.to_numpy()
			label = row[self.label_col] if self.label_col is not None else None
			sample = {'features': features, 'label': label, 'index': idx}

		else:
			# satellite mode
			file_path = self.files[idx]
			if self._cache is not None and file_path in self._cache:
				arr = self._cache[file_path]
			else:
				arr = np.load(file_path)
				if self._cache is not None:
					self._cache[file_path] = arr

			label = None
			if self.label_col is not None and self.label_col in self.df.columns:
				label = self.df.iloc[idx][self.label_col]

			sample = {'image': arr, 'label': label, 'file': str(file_path), 'index': idx}

		if self.transform:
			sample = self.transform(sample)

		return sample


class SICKLETimeSeriesDataset(SICKLEDataset):
	"""Placeholder for time-series dataset; behaviour should be implemented
	by the pipeline's preprocessing modules. For now it behaves like
	SICKLEDataset but kept as a separate symbol for imports.
	"""
	pass


class SICKLEFieldDataset(SICKLEDataset):
	"""Placeholder dataset for field-level stacks (spatial arrays).
	This should eventually return (T,C,H,W) arrays for each field.
	"""
	pass


def create_dataloaders(dataset: Dataset,
					   batch_size: int = 8,
					   num_workers: int = 4,
					   seed: int = 42,
					   val_fraction: float = 0.1,
					   test_fraction: float = 0.1):
	"""Split dataset into train/val/test and return DataLoaders.

	Simple random split using provided fractions. Returns (train, val, test).
	"""
	import torch
	from torch.utils.data import DataLoader, Subset
	import math

	n = len(dataset)
	if n == 0:
		raise ValueError('Empty dataset')

	# compute split sizes
	test_size = int(math.floor(n * test_fraction))
	val_size = int(math.floor(n * val_fraction))
	train_size = n - val_size - test_size

	# deterministic shuffling
	rng = np.random.default_rng(seed)
	indices = np.arange(n)
	rng.shuffle(indices)

	train_idx = indices[:train_size].tolist()
	val_idx = indices[train_size:train_size + val_size].tolist()
	test_idx = indices[train_size + val_size:].tolist()

	train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return train_loader, val_loader, test_loader

