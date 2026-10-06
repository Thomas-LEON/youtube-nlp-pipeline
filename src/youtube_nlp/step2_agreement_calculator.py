"""
Agreement Calculator Module for YouTube NLP Stance Detection Pipeline.

This module provides a production-ready, object-oriented implementation for
evaluating inter-rater reliability (IRR) across multiple annotators using
Fleiss' Kappa index, refactored from notebooks/2_Agreement_Calculator.ipynb.

Academic & NLP Pipeline Context:
---------------------------------
In large-scale social media analysis—specifically analyzing public health discourse
regarding electronic cigarettes (vaping) on YouTube—annotator subjectivity is a major
source of noise and bias.

To train high-capacity deep learning classifiers (such as multilingual BERT / mBERT)
capable of stance detection (Step 4 of this pipeline), a gold-standard ground truth
dataset is required. In this research project, ~4,000 French-language YouTube comments
were independently labeled by three human annotators ('Arthur', 'Thomas', 'Gwendal') into
three categorical stances:
    1. 'F' (Favorable): Expresses explicit or implicit support, advocacy, or positive
       attitude towards e-cigarettes/vaping.
    2. 'C' (Contrary / Unfavorable): Expresses opposition, health warnings, criticism,
       or negative attitude towards e-cigarettes/vaping.
    3. 'I' (Indifferent / Inconclusive): Expresses neutral sentiment, ambiguous stance,
       unrelated remarks, or queries without a discernable opinion.

Why Fleiss' Kappa in AI Safety & NLP?
-------------------------------------
1. Beyond Raw Percent Agreement:
   Simple percentage agreement (Po) suffers from the "chance agreement paradox":
   if annotators are presented with a heavily skewed distribution (e.g. 80% indifferent
   comments), two annotators guessing randomly according to the marginal distribution
   would still achieve high apparent agreement. Fleiss' Kappa corrects for this by
   explicitly subtracting expected agreement occurring purely by chance (Pe).

2. Generalization to >2 Raters:
   While Cohen's Kappa is restricted to exactly two raters, Fleiss' Kappa (Fleiss, 1971)
   generalizes Scott's Pi to an arbitrary fixed number of raters (n >= 2) rating
   items across nominal categories (k >= 2).

3. Data Quality & Model Alignment:
   In AI Safety and NLP alignment, models trained on noisy or contradictory annotations
   can learn spurious correlations and amplify annotator bias. Evaluating inter-rater
   reliability ensures that human consensus is statistically sound before committing
   annotations to supervised training.

Mathematical Formulation:
-------------------------
Given N subjects (comments) and n raters assigning each subject into k categories:
Let n_ij be the number of raters who assigned subject i to category j.

1. Extent of agreement for subject i (P_i):
       P_i = [1 / (n * (n - 1))] * [ sum_{j=1}^k (n_ij^2) - n ]

2. Mean observed agreement across all N subjects (P_bar):
       P_bar = (1 / N) * sum_{i=1}^N P_i

3. Overall proportion of assignments to category j (p_j):
       p_j = [1 / (N * n)] * sum_{i=1}^N n_ij

4. Expected agreement by chance (P_bar_e):
       P_bar_e = sum_{j=1}^k (p_j^2)

5. Fleiss' Kappa index:
       kappa = (P_bar - P_bar_e) / (1 - P_bar_e)

6. Fixed-Marginal (Fleiss) vs. Free-Marginal (Randolph) Agreement:
       - Fleiss' Kappa (1971) is a 'fixed-marginal' measure: it assumes annotators know
         or are conditioned by the overall sample distribution (marginals) of the categories.
         When a category has extreme prevalence (e.g., 85% indifferent comments), P_bar_e is
         elevated, which artificially depresses Fleiss' Kappa despite high observed agreement
         (the well-known "prevalence paradox" documented by Feinstein & Cicchetti, 1990).
       - Randolph's Kappa (2005) is a 'free-marginal' alternative: it assumes annotators are
         free to distribute comments across categories without quota constraints. The expected
         chance agreement is uniform across categories:
             P_bar_e = 1 / k
         Randolph's kappa provides an upper bound that is robust against prevalence skew.

7. Downstream NLP Integration:
       The consensus gold-standard dataset produced by this module (FinalLabels.csv) is directly
       consumed by Step 4 of the pipeline (notebooks/4_Stance_Detection_Model.ipynb), where
       multilingual BERT (mBERT) is fine-tuned to classify comments into 3 stance categories.
       High inter-rater agreement confirms that subjective annotator noise does not corrupt
       the model training signal.

Benchmark Interpretation (Landis & Koch, 1977):
    kappa < 0.00          : Poor agreement (less than chance agreement)
    0.00 <= kappa <= 0.20 : Slight agreement
    0.21 <= kappa <= 0.40 : Fair agreement
    0.41 <= kappa <= 0.60 : Moderate agreement
    0.61 <= kappa <= 0.80 : Substantial agreement
    0.81 <= kappa <= 1.00 : Almost perfect agreement
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

__all__ = [
    "AgreementEvaluator",
    "calculate_fleiss_kappa",
    "main",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AgreementEvaluator:
    """Evaluates inter-rater agreement for multi-annotator NLP datasets using Fleiss' Kappa.

    This class encapsulates data ingestion, cleaning, validation, contingency matrix
    construction, Fleiss' Kappa computation, and majority-vote consensus aggregation.
    It strictly adheres to the mathematical methodology established in the research
    notebooks while providing a production-grade, extensible object-oriented API.

    Attributes:
        raters (List[str]): List of rater column names in the dataset.
        categories (List[str]): Valid categorical stance labels (e.g., ['I', 'C', 'F']).
        category_labels (Dict[str, str]): Human-readable mappings for each category code.
        removed_na_data (Optional[pd.DataFrame]): Rows discarded due to missing rater values.
        removed_filtered_data (Optional[pd.DataFrame]): Rows discarded due to invalid labels.
        cleaned_data (Optional[pd.DataFrame]): Cleaned dataset containing valid ratings.
        fleiss_matrix (Optional[pd.DataFrame]): Contingency matrix of counts (subjects x categories).
        last_kappa (Optional[float]): Most recently computed Fleiss' Kappa score.
    """

    DEFAULT_RATERS: List[str] = ["Arthur", "Thomas", "Gwendal"]
    DEFAULT_CATEGORIES: List[str] = ["I", "C", "F"]
    DEFAULT_CATEGORY_LABELS: Dict[str, str] = {
        "I": "Indifferent / Inconclusive",
        "C": "Contrary / Unfavorable",
        "F": "Favorable",
    }
    VALID_TIE_STRATEGIES: Sequence[str] = ("first", "inconclusive", "first_rater", "alphabetical")
    VALID_METHODS: Sequence[str] = ("fleiss", "randolph")

    @staticmethod
    def _validate_raters_list(raters: Sequence[str]) -> None:
        """Validates that a rater list contains at least 2 unique rater names."""
        if len(raters) < 2:
            raise ValueError(f"Inter-rater agreement requires at least 2 raters, got {len(raters)}.")
        if len(set(raters)) != len(raters):
            raise ValueError(f"Raters list must contain unique rater names, got duplicates: {list(raters)}")

    @staticmethod
    def _validate_categories_list(categories: Sequence[str]) -> None:
        """Validates that a category list contains at least 2 unique category labels."""
        if len(categories) < 2:
            raise ValueError(f"Inter-rater agreement requires at least 2 categories, got {len(categories)}.")
        if len(set(categories)) != len(categories):
            raise ValueError(f"Categories must be unique category labels, got duplicates: {list(categories)}")

    @staticmethod
    def _validate_dataframe_columns(df: pd.DataFrame, target_raters: Sequence[str]) -> None:
        """Validates that all target rater columns are present and unique in the DataFrame."""
        for rater in target_raters:
            if rater not in df.columns:
                raise ValueError(
                    f"Target rater column '{rater}' not present in DataFrame (Rater column '{rater}' missing)."
                )
        if any(list(df.columns).count(r) > 1 for r in target_raters):
            raise ValueError("DataFrame contains duplicate columns matching target rater names.")

    def __init__(
        self,
        raters: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        category_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the AgreementEvaluator with raters and categorical labels.

        Args:
            raters: Names of the rater columns. Defaults to ['Arthur', 'Thomas', 'Gwendal'].
            categories: Allowed category codes. Defaults to ['I', 'C', 'F'].
            category_labels: Mapping from category codes to descriptive names.

        Raises:
            ValueError: If fewer than 2 raters/categories are provided, or if duplicate
                rater/category names are detected.
        """
        self.raters: List[str] = list(raters) if raters is not None else list(self.DEFAULT_RATERS)
        self.categories: List[str] = list(categories) if categories is not None else list(self.DEFAULT_CATEGORIES)
        self.category_labels: Dict[str, str] = (
            dict(category_labels) if category_labels is not None else dict(self.DEFAULT_CATEGORY_LABELS)
        )

        self._validate_raters_list(self.raters)
        self._validate_categories_list(self.categories)

        # Audit trails and intermediate state
        self.removed_na_data: Optional[pd.DataFrame] = None
        self.removed_filtered_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.fleiss_matrix: Optional[pd.DataFrame] = None
        self.last_kappa: Optional[float] = None

        logger.info(
            "AgreementEvaluator initialized with raters=%s, categories=%s",
            self.raters,
            self.categories,
        )

    def load_data(
        self,
        filepath: Union[str, Path],
        delimiter: str = ";",
        usecols: Optional[List[str]] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads raw annotation data from a CSV file.

        Args:
            filepath: Path to the CSV file.
            delimiter: Column delimiter used in the CSV file (default is ';').
            usecols: Optional subset of columns to load.
            encoding: Character encoding for the file (e.g. 'utf-8', 'utf-8-sig', 'latin-1').

        Returns:
            pd.DataFrame: Loaded annotation dataset.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If any required rater column is missing.
        """
        path = Path(filepath)
        if not path.is_file():
            logger.error("Annotation file not found: %s", path)
            raise FileNotFoundError(f"Annotation file not found: {path}")

        logger.info("Loading annotation data from %s (delimiter='%s', encoding=%s)...", path, delimiter, encoding)

        # Check raw header line for duplicate column headers using standard csv parser
        with open(path, "r", encoding=encoding or "utf-8-sig", newline="") as f:
            header_line = f.readline()
        if not header_line.strip():
            raise ValueError(f"CSV file is empty: {path}")

        clean_header_line = header_line.lstrip("\ufeff")
        reader = csv.reader([clean_header_line], delimiter=delimiter)
        raw_headers = [c.strip() for c in next(reader) if c.strip()]
        if len(raw_headers) != len(set(raw_headers)):
            logger.error("CSV file contains duplicate column headers: %s", raw_headers)
            raise ValueError(f"CSV file contains duplicate column headers: {raw_headers}")

        df = pd.read_csv(path, delimiter=delimiter, usecols=usecols, encoding=encoding)

        # Strip surrounding whitespace from column names for robust matching
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        # Verify that expected rater columns exist if usecols was not restricted
        missing_raters = [r for r in self.raters if r not in df.columns]
        if missing_raters:
            logger.error("Missing expected rater column(s) in dataset: %s", missing_raters)
            raise ValueError(f"Missing expected rater column(s): {missing_raters}")

        logger.info("Loaded %d rows and %d columns successfully.", len(df), len(df.columns))
        return df

    def clean_and_filter(
        self,
        df: pd.DataFrame,
        raters: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        subset_only_raters_for_na: bool = True,
        strip_whitespace: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cleans and filters annotation data by handling missing and invalid category values.

        The filtering workflow preserves the two-stage exclusion logic from the original notebook:
        1. Removal of records where rater annotations contain missing values (NaN).
        2. Removal of records where any rater assigned a category outside the recognized set.

        Row partitioning is performed using pure boolean indexing, ensuring full robustness
        even when the DataFrame contains duplicate index values.

        Args:
            df: Input DataFrame containing rater annotations.
            raters: List of rater column names (defaults to self.raters).
            categories: List of valid category codes (defaults to self.categories).
            subset_only_raters_for_na: If True, only rater columns are checked for NaNs.
                If False, rows with any NaN across all DataFrame columns are dropped.
            strip_whitespace: If True, strips surrounding whitespace from string labels in
                rater columns before validation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - valid_data: Cleaned DataFrame containing valid rows.
                - removed_na_data: Rows removed due to missing values.
                - removed_filtered_data: Rows removed due to invalid/out-of-domain categories.
        """
        target_raters = list(raters) if raters is not None else self.raters
        target_categories = list(categories) if categories is not None else self.categories

        self._validate_raters_list(target_raters)
        self._validate_categories_list(target_categories)
        self._validate_dataframe_columns(df, target_raters)

        initial_count = len(df)
        original_data = df.copy()

        # Step 0: Optionally strip leading/trailing whitespace from string labels in rater columns
        df_processed = df.copy()
        if strip_whitespace:
            for rater in target_raters:
                df_processed[rater] = df_processed[rater].map(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        # Step 1: Remove rows with NaN in rater annotations (using boolean mask independent of index values)
        if subset_only_raters_for_na:
            na_mask = df_processed[target_raters].isna().any(axis=1)
        else:
            na_mask = df_processed.isna().any(axis=1)

        self.removed_na_data = original_data[na_mask].copy()
        data_no_na = df_processed[~na_mask].copy()
        na_dropped_count = len(self.removed_na_data)

        logger.info(
            "NaN cleaning: Dropped %d rows containing missing ratings (%d remaining of %d).",
            na_dropped_count,
            len(data_no_na),
            initial_count,
        )

        # Step 2: Filter valid categories across all raters
        valid_mask = pd.Series(True, index=data_no_na.index)
        for rater in target_raters:
            valid_mask &= data_no_na[rater].isin(target_categories)

        valid_data = data_no_na[valid_mask].copy()
        subset_no_na = original_data[~na_mask]
        self.removed_filtered_data = subset_no_na[~valid_mask.to_numpy()].copy()
        invalid_dropped_count = len(self.removed_filtered_data)

        logger.info(
            "Category filtering: Dropped %d rows with out-of-domain values outside %s (%d retained).",
            invalid_dropped_count,
            target_categories,
            len(valid_data),
        )

        self.cleaned_data = valid_data
        return valid_data, self.removed_na_data, self.removed_filtered_data

    def build_contingency_matrix(
        self,
        df: pd.DataFrame,
        raters: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Constructs the contingency rating matrix required for Fleiss' Kappa calculation.

        In the contingency matrix:
            - Rows correspond to subjects (comments), indexed identically to df.
            - Columns correspond to categories (e.g., 'I', 'C', 'F').
            - Matrix cell (i, j) contains the integer count of raters who assigned
              category j to subject i.
            - Each row sum strictly equals the total number of raters (n).

        Args:
            df: Cleaned DataFrame containing rater columns.
            raters: List of rater column names (defaults to self.raters).
            categories: List of categories (defaults to self.categories).

        Returns:
            pd.DataFrame: Contingency matrix of shape (N_subjects, k_categories).

        Raises:
            ValueError: If df is empty or rater columns are missing.
        """
        if df.empty:
            raise ValueError("Cannot build contingency matrix from an empty DataFrame.")

        target_raters = list(raters) if raters is not None else self.raters
        target_categories = list(categories) if categories is not None else self.categories

        self._validate_raters_list(target_raters)
        self._validate_categories_list(target_categories)
        self._validate_dataframe_columns(df, target_raters)

        # Initialize matrix with zeros for each subject and category
        fleiss_matrix = pd.DataFrame(0, index=df.index, columns=target_categories)

        # Vectorized accumulation of category counts across all raters using underlying numpy array
        for rater in target_raters:
            rater_values = df[rater].to_numpy()
            for cat in target_categories:
                fleiss_matrix[cat] += (rater_values == cat).astype(int)

        # Validate that each row sums to the total number of raters
        expected_raters = len(target_raters)
        row_sums = fleiss_matrix.sum(axis=1)
        if not (row_sums == expected_raters).all():
            mismatched_rows = fleiss_matrix[row_sums != expected_raters]
            raise ValueError(
                f"Contingency matrix row sum invariant violated! Expected {expected_raters} ratings "
                f"per row, but found discrepancies in {len(mismatched_rows)} rows."
            )

        self.fleiss_matrix = fleiss_matrix
        logger.info(
            "Constructed Fleiss contingency matrix: shape=(%d subjects, %d categories), raters=%d.",
            len(fleiss_matrix),
            len(target_categories),
            expected_raters,
        )
        return fleiss_matrix

    def _validate_matrix(
        self,
        matrix: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Union[int, float]]]],
    ) -> np.ndarray:
        """Validates and converts a contingency matrix to a 2D NumPy float array.

        Performs comprehensive input checks:
            - Validates that elements can be parsed as numeric floats.
            - Validates 2D dimensionality (N subjects x k categories).
            - Validates at least 1 subject (N >= 1).
            - Validates at least 2 categories (k >= 2).
            - Rejects NaN and infinite values.
            - Rejects negative counts.
            - Checks that each subject is evaluated by at least 2 raters.
            - Checks invariant: all subjects have identical total rater count.

        Args:
            matrix: Input contingency matrix representation.

        Returns:
            np.ndarray: Validated 2D float array of shape (N, k).

        Raises:
            TypeError: If input elements cannot be parsed as numeric.
            ValueError: If dimensions, bounds, or invariants are violated.
        """
        try:
            table = np.asarray(matrix, dtype=float)
        except (ValueError, TypeError) as err:
            raise TypeError(f"Contingency matrix elements must be numeric: {err}") from err

        if table.ndim != 2:
            raise ValueError(f"Input matrix must be 2-dimensional (subjects x categories), got ndim={table.ndim}.")

        if isinstance(matrix, pd.DataFrame) and len(matrix.columns) != len(set(matrix.columns)):
            raise ValueError(
                f"Contingency matrix columns must be unique category names, got duplicates: {list(matrix.columns)}"
            )

        n_sub, n_cat = table.shape
        if n_sub == 0:
            raise ValueError("Input matrix must contain at least one subject (N >= 1).")
        if n_cat < 2:
            raise ValueError(f"Input matrix must contain at least two categories (k >= 2), got k={n_cat}.")

        if not np.all(np.isfinite(table)):
            raise ValueError("Contingency matrix cannot contain NaN or infinite values.")

        if (table < 0).any():
            raise ValueError("Contingency matrix cannot contain negative counts.")

        if not np.all(np.isclose(table, np.round(table))):
            raise ValueError("Contingency matrix rater counts must be non-negative integers.")

        rater_counts = table.sum(axis=1)
        if not np.allclose(rater_counts, rater_counts[0]):
            raise ValueError("Inconsistent number of raters per subject. All rows must sum to the same rater count.")
        n_raters = rater_counts[0]
        if n_raters < 2:
            raise ValueError(f"Inter-rater reliability requires at least 2 raters per subject, found {n_raters}.")

        return table

    def compute_fleiss_kappa(
        self,
        matrix: Optional[Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Union[int, float]]]]] = None,
        method: str = "fleiss",
    ) -> float:
        """Computes the Fleiss' Kappa index of inter-rater reliability.

        This method accepts standard matrix inputs (NumPy array, Pandas DataFrame, or
        nested Sequence of counts) representing the subjects-by-categories contingency matrix.
        If no matrix is passed, it uses the pre-computed contingency matrix stored in the evaluator.

        Mathematical Logic:
        -------------------
        Fleiss' Kappa computes observed pairwise agreement across all raters (P_bar)
        and normalizes it against the agreement expected purely by chance (P_bar_e):
            kappa = (P_bar - P_bar_e) / (1 - P_bar_e)

        Args:
            matrix: 2D array-like input of shape (N, k), where N is the number of subjects
                and k is the number of categories. Each element (i, j) is the count of raters
                assigning subject i to category j.
            method: Underlying calculation method ('fleiss' or 'randolph'). Defaults to 'fleiss'.

        Returns:
            float: Fleiss' Kappa index, bounded within [-1.0, 1.0].

        Raises:
            ValueError: If the input matrix is invalid, empty, has fewer than 2 categories,
                has inconsistent rater counts per subject, or if method is not recognized.
            TypeError: If matrix elements are non-numeric.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Supported methods are {list(self.VALID_METHODS)}.")

        if matrix is None:
            if self.fleiss_matrix is None:
                raise ValueError(
                    "No matrix provided and no pre-computed contingency matrix found in evaluator. "
                    "Provide a matrix or call build_contingency_matrix() first."
                )
            target_matrix = self.fleiss_matrix
        else:
            target_matrix = matrix

        table = self._validate_matrix(target_matrix)

        # Check for degenerate case: all raters assign the exact same category across all subjects
        # In this scenario, chance agreement Pe = 1.0, leading to a 0/0 indeterminate form.
        p_cat = table.sum(axis=0) / table.sum()
        p_mean_exp = float((p_cat * p_cat).sum()) if method == "fleiss" else 1.0 / float(table.shape[1])
        if np.isclose(p_mean_exp, 1.0):
            logger.warning(
                "Degenerate annotation case detected: all annotations belong to a single category. "
                "Expected chance agreement Pe is 1.0; Fleiss' Kappa is undefined (returning 1.0 for perfect agreement)."
            )
            self.last_kappa = 1.0
            return 1.0

        # Compute Fleiss' Kappa using statsmodels
        kappa = float(fleiss_kappa(table, method=method))
        self.last_kappa = kappa

        interpretation = self.interpret_kappa(kappa)
        logger.info(
            "Fleiss' Kappa computed successfully: kappa=%.4f (method='%s') -> Interpretation: %s",
            kappa,
            method,
            interpretation,
        )
        return kappa

    def compute_detailed_metrics(
        self,
        matrix: Optional[Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Union[int, float]]]]] = None,
        method: str = "fleiss",
    ) -> Dict[str, Any]:
        """Calculates a comprehensive breakdown of inter-rater agreement statistics.

        Provides deep pedagogical insight into each component of Fleiss' Kappa:
            - Observed agreement (P_bar / Po): Proportion of rater pairs agreeing.
            - Expected chance agreement (P_bar_e / Pe): Agreement expected if raters assigned
              categories randomly according to the marginal category distribution (Fleiss)
              or equal random chance (Randolph).
            - Marginal category distribution: Empirical frequency of each stance class.
            - Qualitative interpretation: Landis & Koch (1977) agreement tier.

        Args:
            matrix: Standard 2D contingency matrix (defaults to self.fleiss_matrix).
            method: Underlying calculation method ('fleiss' or 'randolph'). Defaults to 'fleiss'.

        Returns:
            Dict[str, Any]: Dictionary containing statistical breakdown and metadata.

        Raises:
            ValueError: If method is invalid or matrix invariants are violated.
            TypeError: If matrix elements are non-numeric.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Supported methods are {list(self.VALID_METHODS)}.")

        if matrix is None:
            if self.fleiss_matrix is None:
                raise ValueError("No matrix provided and no contingency matrix available.")
            target_matrix = self.fleiss_matrix
        else:
            target_matrix = matrix

        table = self._validate_matrix(target_matrix)
        n_sub, n_cat = table.shape
        n_total = float(table.sum())
        n_rat = float(table.sum(axis=1)[0])

        # Marginal category frequencies (p_j)
        p_cat = table.sum(axis=0) / n_total

        # Subject-level pairwise agreement (P_i)
        table2 = table * table
        p_rat = (table2.sum(axis=1) - n_rat) / (n_rat * (n_rat - 1.0))
        p_mean = float(p_rat.mean())

        # Expected agreement by chance (P_bar_e)
        if method == "fleiss":
            p_mean_exp = float((p_cat * p_cat).sum())
        else:  # randolph free-marginal
            p_mean_exp = 1.0 / float(n_cat)

        # Fleiss' Kappa
        if np.isclose(p_mean_exp, 1.0):
            kappa = 1.0
        else:
            kappa = (p_mean - p_mean_exp) / (1.0 - p_mean_exp)

        # Format marginal distribution with column names if available
        if isinstance(target_matrix, pd.DataFrame):
            col_names = list(target_matrix.columns)
        else:
            col_names = [f"Category_{j}" for j in range(n_cat)]

        marginal_dict = {col_names[j]: float(p_cat[j]) for j in range(n_cat)}

        self.last_kappa = float(kappa)

        return {
            "kappa": float(kappa),
            "observed_agreement_Po": float(p_mean),
            "expected_agreement_Pe": float(p_mean_exp),
            "method": method,
            "n_subjects": int(n_sub),
            "n_raters": int(n_rat),
            "n_categories": int(n_cat),
            "marginal_distribution": marginal_dict,
            "interpretation": self.interpret_kappa(kappa),
        }

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Translates a Fleiss' Kappa score into the standard Landis & Koch (1977) scale.

        Benchmark Tiers:
            - < 0.00: Poor agreement (systematic disagreement / worse than chance)
            - 0.00 - 0.20: Slight agreement
            - 0.21 - 0.40: Fair agreement
            - 0.41 - 0.60: Moderate agreement
            - 0.61 - 0.80: Substantial agreement
            - 0.81 - 1.00: Almost perfect agreement

        Args:
            kappa: Computed Fleiss' Kappa score.

        Returns:
            str: Qualitative interpretation of the agreement strength.

        Raises:
            TypeError: If kappa cannot be converted to float.
            ValueError: If kappa is non-finite (NaN or Inf).
        """
        try:
            val = float(kappa)
        except (ValueError, TypeError) as err:
            raise TypeError(f"Kappa score must be numeric, got {type(kappa).__name__}: {kappa}") from err

        if not np.isfinite(val):
            raise ValueError(f"Cannot interpret non-finite kappa score: {kappa}")

        if val < 0.00:
            return "Poor agreement"
        elif val <= 0.20:
            return "Slight agreement"
        elif val <= 0.40:
            return "Fair agreement"
        elif val <= 0.60:
            return "Moderate agreement"
        elif val <= 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def determine_majority_label(
        self,
        df: pd.DataFrame,
        raters: Optional[List[str]] = None,
        label_column: str = "label",
        tie_strategy: str = "first",
    ) -> pd.DataFrame:
        """Determines the consensus majority stance label for each comment.

        Refactored directly from Cell 3 of notebooks/2_Agreement_Calculator.ipynb:
            def final_label(row):
                labels = [row['Arthur'], row['Thomas'], row['Gwendal']]
                return max(set(labels), key=labels.count)

        Consensus voting logic:
            - Unanimous agreement (e.g. 3-0): Majority label is unambiguous.
            - Plurality / simple majority (e.g. 2-1): Majority label is unambiguous.
            - Tied votes (e.g. 1-1-1 in 3 raters, or 2-2 in 4 raters): Resolved via `tie_strategy`:
                - 'first': Strict notebook parity (`max(set(labels), key=labels.count)`).
                - 'inconclusive': Assigns 'I' (Indifferent/Inconclusive) due to lack of consensus.
                - 'first_rater': Uses the annotation from the earliest designated rater whose vote
                  is among the tied top candidates.
                - 'alphabetical': Deterministically selects the alphabetically smallest tied label.

        Args:
            df: Cleaned DataFrame containing rater annotations.
            raters: List of rater column names (defaults to self.raters).
            label_column: Name of the consensus label column to create (default 'label').
            tie_strategy: Resolution strategy for ties ('first', 'inconclusive', 'first_rater', 'alphabetical').

        Returns:
            pd.DataFrame: Copy of DataFrame with the new consensus label column.

        Raises:
            ValueError: If invalid tie_strategy is specified or rater columns are missing.
        """
        if tie_strategy not in self.VALID_TIE_STRATEGIES:
            raise ValueError(
                f"Unknown tie_strategy '{tie_strategy}'. Must be one of {list(self.VALID_TIE_STRATEGIES)}."
            )
        if not isinstance(label_column, str) or not label_column.strip():
            raise ValueError("label_column must be a non-empty string.")

        target_raters = list(raters) if raters is not None else self.raters
        self._validate_raters_list(target_raters)
        self._validate_dataframe_columns(df, target_raters)

        result_df = df.copy()
        if result_df.empty:
            result_df[label_column] = pd.Series(dtype=object)
            logger.info("Majority consensus labels assigned across 0 comments: empty DataFrame.")
            return result_df

        if result_df[target_raters].isna().any().any():
            logger.warning(
                "Input DataFrame contains NaN values in rater columns. "
                "Consider calling clean_and_filter() first."
            )

        def _compute_row_label(row: pd.Series) -> Any:
            labels = [row[r] for r in target_raters]
            # Exclude missing / NaN values from candidate pool so unassigned ratings don't skew voting
            valid_votes = [lbl for lbl in labels if pd.notna(lbl)]

            if not valid_votes:
                return "I" if "I" in self.categories else None

            counts: Dict[Any, int] = {}
            for lbl in valid_votes:
                counts[lbl] = counts.get(lbl, 0) + 1

            max_count = max(counts.values())
            top_candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]

            if len(top_candidates) == 1:
                return top_candidates[0]

            # Tie detected among top vote getters
            if tie_strategy == "inconclusive":
                return "I" if "I" in self.categories else top_candidates[0]
            elif tie_strategy == "first_rater":
                # Find the earliest designated rater whose vote is among the top contenders
                for rater in target_raters:
                    if row[rater] in top_candidates:
                        return row[rater]
                return top_candidates[0]  # pragma: no cover
            elif tie_strategy == "alphabetical":
                return sorted(top_candidates, key=str)[0]
            else:  # 'first' - strict notebook parity: max(set(valid_votes), key=valid_votes.count)
                return max(set(valid_votes), key=valid_votes.count)

        result_df[label_column] = result_df.apply(_compute_row_label, axis=1)

        # Log class distribution of the consensus labels
        label_counts = result_df[label_column].value_counts().to_dict()
        total_valid = len(result_df)
        distribution_str = ", ".join(
            f"{lbl} ({self.category_labels.get(str(lbl), str(lbl))}): {cnt} ({cnt / total_valid * 100:.1f}%)"
            for lbl, cnt in label_counts.items()
        )
        logger.info("Majority consensus labels assigned across %d comments: %s", total_valid, distribution_str)

        return result_df

    def run_full_pipeline(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        output_csv: Optional[Union[str, Path]] = None,
        delimiter: str = ";",
        encoding: Optional[str] = None,
        subset_only_raters_for_na: bool = True,
        strip_whitespace: bool = True,
        tie_strategy: str = "first",
        label_column: str = "label",
        method: str = "fleiss",
    ) -> Tuple[float, pd.DataFrame]:
        """Runs the complete end-to-end agreement calculation and gold-standard generation pipeline.

        Execution Steps:
            1. Ingestion: Load raw annotation dataset.
            2. Cleaning: Exclude NaN values and out-of-domain categories, logging dropped rows.
            3. Matrix Construction: Build subjects-by-categories contingency matrix.
            4. Reliability Assessment: Compute and interpret Fleiss' Kappa index.
            5. Consensus Aggregation: Assign majority vote stance labels.
            6. Persistence: Export final gold-standard dataset to output_csv (if provided).

        Args:
            input_data: Filepath to raw CSV or existing DataFrame.
            output_csv: Optional destination path to save the final labeled dataset.
            delimiter: CSV delimiter (default ';').
            encoding: CSV encoding (e.g. 'utf-8', 'utf-8-sig', 'latin-1').
            subset_only_raters_for_na: If True, only drops rows where rater columns are NaN.
            strip_whitespace: If True, trims whitespace in rater columns before validation.
            tie_strategy: Strategy to break vote ties ('first', 'inconclusive', 'first_rater', 'alphabetical').
            label_column: Destination column name for consensus stance label.
            method: Fleiss' Kappa calculation method ('fleiss' or 'randolph').

        Returns:
            Tuple[float, pd.DataFrame]:
                - Fleiss' Kappa score.
                - Final DataFrame with consensus 'label' column.
        """
        logger.info("Starting full Agreement Calculator pipeline...")

        # Step 1: Load or accept DataFrame
        if isinstance(input_data, (str, Path)):
            df_raw = self.load_data(input_data, delimiter=delimiter, encoding=encoding)
        else:
            df_raw = input_data.copy()

        # Step 2: Clean and filter
        valid_df, _, _ = self.clean_and_filter(
            df_raw,
            subset_only_raters_for_na=subset_only_raters_for_na,
            strip_whitespace=strip_whitespace,
        )

        # Step 3: Build contingency matrix
        matrix = self.build_contingency_matrix(valid_df)

        # Step 4: Compute Fleiss' Kappa
        kappa = self.compute_fleiss_kappa(matrix, method=method)

        # Step 5: Majority consensus labeling
        final_df = self.determine_majority_label(
            valid_df,
            label_column=label_column,
            tie_strategy=tie_strategy,
        )

        # Step 6: Export to CSV if output path specified
        if output_csv is not None:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(out_path, index=False, encoding=encoding or "utf-8")
            logger.info("Saved final labeled dataset to %s (%d rows).", out_path, len(final_df))

        logger.info("Pipeline execution complete. Final Kappa = %.4f", kappa)
        return kappa, final_df


def calculate_fleiss_kappa(
    matrix: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Union[int, float]]]],
    method: str = "fleiss",
) -> float:
    """Convenience functional interface to calculate Fleiss' Kappa on a standard matrix.

    Args:
        matrix: 2D array-like contingency matrix of shape (subjects, categories).
        method: Calculation method ('fleiss' or 'randolph'). Defaults to 'fleiss'.

    Returns:
        float: Fleiss' Kappa index.
    """
    evaluator = AgreementEvaluator()
    return evaluator.compute_fleiss_kappa(matrix=matrix, method=method)


def main() -> Tuple[float, pd.DataFrame]:
    """Runs a pedagogical demonstration of the agreement pipeline on sample data."""
    print("=" * 70)
    print("YouTube NLP Stance Detection: Fleiss' Kappa Agreement Demonstration")
    print("=" * 70)

    sample_annotations = pd.DataFrame(
        {
            "comment_id": [101, 102, 103, 104, 105, 106, 107],
            "comment": [
                "Vaping really helped me quit traditional cigarettes!",
                "E-cigarettes are dangerous for youth and should be banned.",
                "Does anyone know the battery life of this mod?",
                "I prefer fruit flavors, much better than tobacco taste.",
                "Both cigarettes and vapes are harmful to lungs.",
                "Nice video thanks.",
                "Invalid row for test",
            ],
            "Arthur": ["F", "C", "I", "F", "C", "I", "NaN_entry"],
            "Thomas": ["F", "C", "I", "F", "C", "F", "F"],
            "Gwendal": ["F", "C", "I", "I", "C", "I", "C"],
        }
    )

    evaluator = AgreementEvaluator()
    demo_output = Path("FinalLabels_demo.csv")
    kappa_val, labeled_df = evaluator.run_full_pipeline(
        input_data=sample_annotations,
        output_csv=demo_output,
    )

    metrics = evaluator.compute_detailed_metrics()
    print("\nStatistical Breakdown:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nLabeled Output Sample:")
    print(labeled_df[["comment_id", "comment", "Arthur", "Thomas", "Gwendal", "label"]])
    print("=" * 70)

    if demo_output.is_file():
        demo_output.unlink()

    return kappa_val, labeled_df


if __name__ == "__main__":
    main()
