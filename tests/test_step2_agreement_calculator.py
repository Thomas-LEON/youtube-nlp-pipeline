"""Unit and integration tests for AgreementEvaluator and Fleiss' Kappa calculation."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.youtube_nlp.step2_agreement_calculator import (
    AgreementEvaluator,
    calculate_fleiss_kappa,
)


class TestAgreementEvaluator:
    """Test suite covering the AgreementEvaluator class and its methods."""

    def test_initialization_defaults(self):
        """Tests default initialization configuration."""
        evaluator = AgreementEvaluator()
        assert evaluator.raters == ["Arthur", "Thomas", "Gwendal"]
        assert evaluator.categories == ["I", "C", "F"]
        assert "I" in evaluator.category_labels
        assert "C" in evaluator.category_labels
        assert "F" in evaluator.category_labels
        assert evaluator.cleaned_data is None
        assert evaluator.fleiss_matrix is None
        assert evaluator.last_kappa is None

    def test_initialization_custom(self):
        """Tests initialization with custom raters and categories."""
        custom_raters = ["Annotator1", "Annotator2"]
        custom_categories = ["Pos", "Neg"]
        evaluator = AgreementEvaluator(raters=custom_raters, categories=custom_categories)
        assert evaluator.raters == custom_raters
        assert evaluator.categories == custom_categories

    def test_compute_fleiss_kappa_textbook_benchmark(self):
        """Tests Fleiss' Kappa against the classic Fleiss (1971) Table 1 benchmark.

        Dataset: 10 subjects, 14 raters, 5 diagnostic categories.
        Expected Fleiss' Kappa: ~0.20993 (fair agreement).
        """
        # Fleiss (1971) Table 1 contingency matrix
        textbook_matrix = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7],
            ]
        )

        evaluator = AgreementEvaluator()
        kappa = evaluator.compute_fleiss_kappa(textbook_matrix)
        assert np.isclose(kappa, 0.2099307, atol=1e-5)
        assert evaluator.last_kappa == kappa
        assert evaluator.interpret_kappa(kappa) == "Fair agreement"

    def test_compute_fleiss_kappa_input_types(self):
        """Acceptance Criteria: Accepts standard matrix inputs (DataFrame, ndarray, List)."""
        evaluator = AgreementEvaluator()
        raw_list = [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [2, 1, 0],
        ]

        # 1. Test nested Python list
        kappa_list = evaluator.compute_fleiss_kappa(raw_list)

        # 2. Test NumPy ndarray
        kappa_np = evaluator.compute_fleiss_kappa(np.array(raw_list))

        # 3. Test Pandas DataFrame
        df_matrix = pd.DataFrame(raw_list, columns=["I", "C", "F"])
        kappa_df = evaluator.compute_fleiss_kappa(df_matrix)

        # 4. Test standalone convenience function
        kappa_fn = calculate_fleiss_kappa(df_matrix)

        assert np.isclose(kappa_list, kappa_np)
        assert np.isclose(kappa_np, kappa_df)
        assert np.isclose(kappa_df, kappa_fn)
        assert 0.0 < kappa_df <= 1.0

    def test_compute_fleiss_kappa_methods(self):
        """Tests that both 'fleiss' and 'randolph' methods function."""
        evaluator = AgreementEvaluator()
        matrix = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1]])
        k_fleiss = evaluator.compute_fleiss_kappa(matrix, method="fleiss")
        k_randolph = evaluator.compute_fleiss_kappa(matrix, method="randolph")
        assert -1.0 <= k_fleiss <= 1.0
        assert -1.0 <= k_randolph <= 1.0

    def test_perfect_agreement(self):
        """Tests that unanimous agreement across balanced categories yields kappa = 1.0."""
        evaluator = AgreementEvaluator()
        perfect_matrix = np.array(
            [
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
            ]
        )
        kappa = evaluator.compute_fleiss_kappa(perfect_matrix)
        assert np.isclose(kappa, 1.0)
        assert evaluator.interpret_kappa(kappa) == "Almost perfect agreement"

    def test_clean_and_filter(self):
        """Tests NaN removal and out-of-domain category filtering matching notebook 2."""
        evaluator = AgreementEvaluator()
        raw_df = pd.DataFrame(
            {
                "comment_id": [1, 2, 3, 4, 5, 6],
                "comment": ["c1", "c2", "c3", "c4", "c5", "c6"],
                "Arthur": ["F", "C", "I", None, "X", "F"],
                "Thomas": ["F", "C", "I", "F", "C", "C"],
                "Gwendal": ["F", "C", "I", "I", "F", "I"],
            }
        )

        valid_df, removed_na, removed_filtered = evaluator.clean_and_filter(raw_df)

        # Comment 4 had NaN -> removed_na
        assert len(removed_na) == 1
        assert 3 in removed_na.index

        # Comment 5 had 'X' -> removed_filtered
        assert len(removed_filtered) == 1
        assert 4 in removed_filtered.index

        # Comments 1, 2, 3, 6 are valid
        assert len(valid_df) == 4
        assert set(valid_df["comment_id"]) == {1, 2, 3, 6}

    def test_clean_and_filter_all_columns_na(self):
        """Tests dropping rows where non-rater metadata contains NaN when subset_only_raters_for_na=False."""
        evaluator = AgreementEvaluator()
        raw_df = pd.DataFrame(
            {
                "comment_id": [1, 2],
                "meta": [None, "valid"],
                "Arthur": ["F", "F"],
                "Thomas": ["F", "F"],
                "Gwendal": ["F", "F"],
            }
        )
        valid_df, removed_na, _ = evaluator.clean_and_filter(raw_df, subset_only_raters_for_na=False)
        assert len(valid_df) == 1
        assert len(removed_na) == 1

    def test_clean_and_filter_missing_rater_error(self):
        """Tests that clean_and_filter raises ValueError if rater column is missing."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame({"Arthur": ["F"], "Thomas": ["F"]})  # missing Gwendal
        with pytest.raises(ValueError, match="Target rater column 'Gwendal' not present"):
            evaluator.clean_and_filter(df)

    def test_build_contingency_matrix(self):
        """Tests construction of contingency matrix and row sum validation."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame(
            {
                "Arthur": ["F", "I", "C"],
                "Thomas": ["F", "I", "F"],
                "Gwendal": ["F", "C", "C"],
            }
        )

        matrix = evaluator.build_contingency_matrix(df)
        assert matrix.shape == (3, 3)
        assert list(matrix.columns) == ["I", "C", "F"]

        # Row 0: F=3, C=0, I=0
        assert matrix.loc[0, "F"] == 3
        assert matrix.loc[0, "C"] == 0
        assert matrix.loc[0, "I"] == 0

        # Row 1: Arthur=I, Thomas=I, Gwendal=C -> I=2, C=1, F=0
        assert matrix.loc[1, "I"] == 2
        assert matrix.loc[1, "C"] == 1
        assert matrix.loc[1, "F"] == 0

        # Row 2: Arthur=C, Thomas=F, Gwendal=C -> I=0, C=2, F=1
        assert matrix.loc[2, "I"] == 0
        assert matrix.loc[2, "C"] == 2
        assert matrix.loc[2, "F"] == 1

        # All rows must sum to 3 (number of raters)
        assert (matrix.sum(axis=1) == 3).all()

    def test_build_contingency_matrix_empty_error(self):
        """Tests that build_contingency_matrix raises ValueError on empty DataFrame."""
        evaluator = AgreementEvaluator()
        with pytest.raises(ValueError, match="Cannot build contingency matrix from an empty"):
            evaluator.build_contingency_matrix(pd.DataFrame())

    def test_build_contingency_matrix_missing_rater_error(self):
        """Tests that build_contingency_matrix raises ValueError when rater column is missing."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame({"Arthur": ["F"], "Thomas": ["F"]})
        with pytest.raises(ValueError, match="Rater column 'Gwendal' missing"):
            evaluator.build_contingency_matrix(df)

    def test_build_contingency_matrix_row_sum_discrepancy(self):
        """Tests that passing uncleaned data with invalid categories raises invariant error."""
        evaluator = AgreementEvaluator()
        # Row 0 has an invalid category 'Z' which is not in ['I', 'C', 'F']
        df_uncleaned = pd.DataFrame(
            {
                "Arthur": ["Z"],
                "Thomas": ["F"],
                "Gwendal": ["F"],
            }
        )
        with pytest.raises(ValueError, match="Contingency matrix row sum invariant violated"):
            evaluator.build_contingency_matrix(df_uncleaned)

    def test_run_full_pipeline_dataframe_input(self):
        """Tests run_full_pipeline accepting a DataFrame directly without file I/O."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame(
            {
                "comment_id": [1, 2],
                "Arthur": ["F", "C"],
                "Thomas": ["F", "C"],
                "Gwendal": ["F", "C"],
            }
        )
        kappa, result_df = evaluator.run_full_pipeline(df)
        assert np.isclose(kappa, 1.0)
        assert len(result_df) == 2
        assert list(result_df["label"]) == ["F", "C"]

    def test_determine_majority_label(self):
        """Tests majority vote consensus assignment matching cell 3 of the notebook."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame(
            {
                "Arthur": ["F", "C", "I", "F"],
                "Thomas": ["F", "C", "I", "I"],
                "Gwendal": ["F", "I", "C", "C"],
            }
        )

        labeled_df = evaluator.determine_majority_label(df)
        assert "label" in labeled_df.columns
        assert labeled_df.loc[0, "label"] == "F"
        assert labeled_df.loc[1, "label"] == "C"
        assert labeled_df.loc[2, "label"] == "I"

        # Row 3 is a 3-way tie: F, I, C. With 'first', max(set, key) returns one of them
        assert labeled_df.loc[3, "label"] in {"F", "I", "C"}

        # Test tie_strategy='inconclusive'
        labeled_inconclusive = evaluator.determine_majority_label(df, tie_strategy="inconclusive")
        assert labeled_inconclusive.loc[3, "label"] == "I"

    def test_determine_majority_label_errors(self):
        """Tests error conditions for determine_majority_label."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame({"Arthur": ["F"], "Thomas": ["F"]})  # missing Gwendal
        with pytest.raises(ValueError, match="Rater column 'Gwendal' missing"):
            evaluator.determine_majority_label(df)

        df_valid = pd.DataFrame({"Arthur": ["F"], "Thomas": ["C"], "Gwendal": ["I"]})
        with pytest.raises(ValueError, match="Unknown tie_strategy"):
            evaluator.determine_majority_label(df_valid, tie_strategy="unsupported")

    def test_compute_detailed_metrics(self):
        """Tests detailed statistics output (Po, Pe, marginals, interpretation)."""
        evaluator = AgreementEvaluator()
        sample_matrix = np.array(
            [
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [2, 1, 0],
            ]
        )
        # Using explicit matrix
        metrics = evaluator.compute_detailed_metrics(sample_matrix)
        assert "kappa" in metrics
        assert "observed_agreement_Po" in metrics
        assert "expected_agreement_Pe" in metrics
        assert "marginal_distribution" in metrics
        assert "interpretation" in metrics
        assert metrics["n_subjects"] == 4
        assert metrics["n_raters"] == 3
        assert metrics["n_categories"] == 3
        assert "Category_0" in metrics["marginal_distribution"]

        # Using pre-computed DataFrame matrix
        df_matrix = pd.DataFrame(sample_matrix, columns=["I", "C", "F"])
        evaluator.compute_fleiss_kappa(df_matrix)
        evaluator.fleiss_matrix = df_matrix
        metrics_df = evaluator.compute_detailed_metrics()
        assert "I" in metrics_df["marginal_distribution"]

    def test_compute_detailed_metrics_no_matrix_error(self):
        """Tests error when calling compute_detailed_metrics without a matrix."""
        evaluator = AgreementEvaluator()
        with pytest.raises(ValueError, match="No matrix provided"):
            evaluator.compute_detailed_metrics()

    def test_interpret_kappa_tiers(self):
        """Tests that all Landis & Koch tiers are correctly mapped."""
        assert AgreementEvaluator.interpret_kappa(-0.1) == "Poor agreement"
        assert AgreementEvaluator.interpret_kappa(0.10) == "Slight agreement"
        assert AgreementEvaluator.interpret_kappa(0.35) == "Fair agreement"
        assert AgreementEvaluator.interpret_kappa(0.55) == "Moderate agreement"
        assert AgreementEvaluator.interpret_kappa(0.75) == "Substantial agreement"
        assert AgreementEvaluator.interpret_kappa(0.95) == "Almost perfect agreement"

    def test_run_full_pipeline_with_csv(self):
        """Tests end-to-end execution loading CSV, cleaning, computing Kappa, and exporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "FrenchLabels_test.csv"
            output_csv = Path(tmpdir) / "FinalLabels_test.csv"

            # Create mock FrenchLabels.csv matching real project format
            csv_content = (
                "comment_id;comment;Arthur;Thomas;Gwendal\n"
                "1;Vape is great;F;F;F\n"
                "2;Very harmful;C;C;C\n"
                "3;No opinion;I;I;I\n"
                "4;I think it helps;F;F;I\n"
                "5;Bad taste;C;C;F\n"
                "6;Missing entry;NaN;;F\n"
                "7;Invalid entry;X;F;F\n"
            )
            input_csv.write_text(csv_content, encoding="utf-8")

            evaluator = AgreementEvaluator()
            kappa, final_df = evaluator.run_full_pipeline(
                input_data=input_csv,
                output_csv=output_csv,
                delimiter=";",
            )

            assert output_csv.is_file()
            assert len(final_df) == 5
            assert "label" in final_df.columns
            assert final_df.loc[final_df["comment_id"] == 1, "label"].values[0] == "F"
            assert final_df.loc[final_df["comment_id"] == 2, "label"].values[0] == "C"
            assert final_df.loc[final_df["comment_id"] == 3, "label"].values[0] == "I"
            assert kappa > 0.5

    def test_compute_fleiss_kappa_using_precomputed_matrix(self):
        """Tests that compute_fleiss_kappa can use self.fleiss_matrix if matrix is None."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame(
            {
                "Arthur": ["F", "I"],
                "Thomas": ["F", "I"],
                "Gwendal": ["F", "C"],
            }
        )
        evaluator.build_contingency_matrix(df)
        kappa = evaluator.compute_fleiss_kappa()
        assert -1.0 <= kappa <= 1.0

    def test_compute_fleiss_kappa_no_matrix_error(self):
        """Tests that compute_fleiss_kappa raises ValueError if no matrix is passed and none stored."""
        evaluator = AgreementEvaluator()
        with pytest.raises(ValueError, match="No matrix provided"):
            evaluator.compute_fleiss_kappa()

    def test_load_data_missing_raters_error(self):
        """Tests that load_data raises ValueError if required rater column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_file.write_text("Arthur;Thomas\nF;F\n", encoding="utf-8")
            evaluator = AgreementEvaluator()
            with pytest.raises(ValueError, match="Missing expected rater column"):
                evaluator.load_data(csv_file, delimiter=";")

    def test_input_validation_errors(self):
        """Tests defensive input validations."""
        evaluator = AgreementEvaluator()

        # Dimension error: 1D array
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            evaluator.compute_fleiss_kappa(np.array([1, 2, 3]))

        # Empty matrix
        with pytest.raises(ValueError, match="at least one subject"):
            evaluator.compute_fleiss_kappa(np.empty((0, 3)))

        # Fewer than 2 categories
        with pytest.raises(ValueError, match="at least two categories"):
            evaluator.compute_fleiss_kappa(np.array([[3], [3]]))

        # Negative counts
        with pytest.raises(ValueError, match="negative counts"):
            evaluator.compute_fleiss_kappa(np.array([[3, -1, 1], [1, 1, 1]]))

        # Inconsistent raters per subject
        with pytest.raises(ValueError, match="Inconsistent number of raters"):
            evaluator.compute_fleiss_kappa(np.array([[3, 0, 0], [1, 1, 0]]))

        # Fewer than 2 raters
        with pytest.raises(ValueError, match="at least 2 raters"):
            evaluator.compute_fleiss_kappa(np.array([[1, 0], [0, 1]]))

        # File not found
        with pytest.raises(FileNotFoundError):
            evaluator.load_data("non_existent_file.csv")

    def test_degenerate_all_same_category(self):
        """Tests that a degenerate dataset where all ratings are in one category is handled safely."""
        evaluator = AgreementEvaluator()
        degenerate_matrix = np.array(
            [
                [3, 0, 0],
                [3, 0, 0],
                [3, 0, 0],
            ]
        )
        kappa = evaluator.compute_fleiss_kappa(degenerate_matrix)
        assert kappa == 1.0

        metrics = evaluator.compute_detailed_metrics(degenerate_matrix)
        assert metrics["kappa"] == 1.0

    def test_module_main_execution(self):
        """Tests that the module can be executed directly as a script (__main__)."""
        result = subprocess.run(
            [sys.executable, "src/youtube_nlp/agreement_calculator.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "YouTube NLP Stance Detection: Fleiss' Kappa Agreement Demonstration" in result.stdout
        assert "Fleiss' Kappa computed successfully" in result.stderr or "Fleiss' Kappa" in result.stdout

        # Clean up any generated demo file
        demo_file = Path("FinalLabels_demo.csv")
        if demo_file.is_file():
            demo_file.unlink()

    def test_clean_and_filter_duplicate_indices(self):
        """Tests that clean_and_filter preserves audit rows even with duplicate indices."""
        evaluator = AgreementEvaluator()
        # 3 rows, all sharing index 0: 1 valid, 1 invalid category, 1 NaN
        df_duplicates = pd.DataFrame(
            {
                "Arthur": ["F", "X", None],
                "Thomas": ["F", "F", "F"],
                "Gwendal": ["F", "F", "F"],
            },
            index=[0, 0, 0],
        )
        valid, removed_na, removed_filtered = evaluator.clean_and_filter(df_duplicates)
        assert len(valid) == 1
        assert len(removed_na) == 1
        assert len(removed_filtered) == 1

    def test_clean_and_filter_whitespace_stripping(self):
        """Tests handling of leading/trailing whitespace in rater labels."""
        evaluator = AgreementEvaluator()
        df_whitespace = pd.DataFrame(
            {
                "Arthur": [" F ", "C\t", "I\n"],
                "Thomas": ["F", "C", "I"],
                "Gwendal": ["F", "C", "I"],
            }
        )
        # With default strip_whitespace=True, whitespace is trimmed and all 3 rows retained
        valid, _, removed_filtered = evaluator.clean_and_filter(df_whitespace, strip_whitespace=True)
        assert len(valid) == 3
        assert len(removed_filtered) == 0
        assert list(valid["Arthur"]) == ["F", "C", "I"]

        # With strip_whitespace=False, untrimmed rows are discarded
        valid_strict, _, removed_strict = evaluator.clean_and_filter(df_whitespace, strip_whitespace=False)
        assert len(valid_strict) == 0
        assert len(removed_strict) == 3

    def test_load_data_with_column_whitespace(self):
        """Tests that load_data handles whitespace in CSV column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "spaced_headers.csv"
            csv_path.write_text("Arthur ; Thomas ; Gwendal \nF;F;F\n", encoding="utf-8")
            evaluator = AgreementEvaluator()
            df = evaluator.load_data(csv_path, delimiter=";")
            assert list(df.columns) == ["Arthur", "Thomas", "Gwendal"]
            assert len(df) == 1

    def test_load_data_encoding_support(self):
        """Tests loading CSV files with specific encodings such as latin-1 and utf-8-sig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "french_accents.csv"
            # French accented text saved with latin-1 encoding
            content = "comment;Arthur;Thomas;Gwendal\nTrès favorable!;F;F;F\n"
            csv_path.write_bytes(content.encode("latin-1"))

            evaluator = AgreementEvaluator()
            df = evaluator.load_data(csv_path, delimiter=";", encoding="latin-1")
            assert len(df) == 1
            assert df.loc[0, "comment"] == "Très favorable!"

    def test_compute_fleiss_kappa_nan_inf_rejected(self):
        """Tests that matrices containing NaN or infinite values are explicitly rejected."""
        evaluator = AgreementEvaluator()
        with pytest.raises(ValueError, match="NaN or infinite values"):
            evaluator.compute_fleiss_kappa([[np.nan, 2], [1, 1]])

        with pytest.raises(ValueError, match="NaN or infinite values"):
            evaluator.compute_fleiss_kappa([[np.inf, 2], [1, 1]])

    def test_compute_fleiss_kappa_numeric_string_conversion(self):
        """Tests that numeric string matrices are converted cleanly, while non-numeric strings raise TypeError."""
        evaluator = AgreementEvaluator()
        numeric_str_matrix = [["3", "0", "0"], ["0", "3", "0"]]
        k = evaluator.compute_fleiss_kappa(numeric_str_matrix)
        assert np.isclose(k, 1.0)

        non_numeric = [["a", "b"], ["c", "d"]]
        with pytest.raises(TypeError, match="must be numeric"):
            evaluator.compute_fleiss_kappa(non_numeric)

    def test_compute_fleiss_kappa_invalid_method(self):
        """Tests that invalid method names raise ValueError immediately."""
        evaluator = AgreementEvaluator()
        with pytest.raises(ValueError, match="Invalid method 'unsupported'"):
            evaluator.compute_fleiss_kappa([[2, 0], [0, 2]], method="unsupported")

    def test_compute_detailed_metrics_validation(self):
        """Tests that compute_detailed_metrics validates matrix inputs defensively."""
        evaluator = AgreementEvaluator()
        # Empty matrix
        with pytest.raises(ValueError, match="at least one subject"):
            evaluator.compute_detailed_metrics(np.empty((0, 3)))

        # Inconsistent raters
        with pytest.raises(ValueError, match="Inconsistent number of raters"):
            evaluator.compute_detailed_metrics(np.array([[3, 0], [1, 1]]))

        # Non-numeric input
        with pytest.raises(TypeError, match="must be numeric"):
            evaluator.compute_detailed_metrics([["a", "b"]])

        # Invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            evaluator.compute_detailed_metrics([[2, 0], [0, 2]], method="unknown")

    def test_compute_detailed_metrics_randolph(self):
        """Tests compute_detailed_metrics with Randolph free-marginal method."""
        evaluator = AgreementEvaluator()
        matrix = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1]])
        metrics = evaluator.compute_detailed_metrics(matrix, method="randolph")
        assert metrics["method"] == "randolph"
        assert np.isclose(metrics["expected_agreement_Pe"], 1.0 / 3.0)
        assert np.isclose(metrics["kappa"], evaluator.compute_fleiss_kappa(matrix, method="randolph"))

    def test_determine_majority_label_early_validation(self):
        """Tests that determine_majority_label validates tie_strategy upfront even without ties."""
        evaluator = AgreementEvaluator()
        df_no_ties = pd.DataFrame({"Arthur": ["F"], "Thomas": ["F"], "Gwendal": ["F"]})
        with pytest.raises(ValueError, match="Unknown tie_strategy 'bogus'"):
            evaluator.determine_majority_label(df_no_ties, tie_strategy="bogus")

    def test_determine_majority_label_four_raters_tie(self):
        """Tests tie detection and resolution for even number of raters (2-2 tie)."""
        evaluator = AgreementEvaluator(raters=["R1", "R2", "R3", "R4"])
        df_even = pd.DataFrame(
            {
                "R1": ["F"],
                "R2": ["F"],
                "R3": ["C"],
                "R4": ["C"],
            }
        )
        # Test 'inconclusive'
        res_inc = evaluator.determine_majority_label(df_even, tie_strategy="inconclusive")
        assert res_inc.loc[0, "label"] == "I"

        # Test 'first_rater' (R1 chose 'F')
        res_r1 = evaluator.determine_majority_label(df_even, tie_strategy="first_rater")
        assert res_r1.loc[0, "label"] == "F"

        # Test 'alphabetical' ('C' < 'F')
        res_alpha = evaluator.determine_majority_label(df_even, tie_strategy="alphabetical")
        assert res_alpha.loc[0, "label"] == "C"

    def test_run_full_pipeline_extended_options(self):
        """Tests run_full_pipeline configuring tie_strategy, label_column, and strip_whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "input.csv"
            output_csv = Path(tmpdir) / "output.csv"
            input_csv.write_text(
                "Arthur ; Thomas ; Gwendal \n"
                " F ; C ; I \n"
                "F;F;C\n",
                encoding="utf-8",
            )
            evaluator = AgreementEvaluator()
            kappa, df_out = evaluator.run_full_pipeline(
                input_data=input_csv,
                output_csv=output_csv,
                delimiter=";",
                strip_whitespace=True,
                tie_strategy="inconclusive",
                label_column="consensus",
            )
            assert output_csv.is_file()
            assert "consensus" in df_out.columns
            # Row 0 was 3-way tie (F, C, I), with inconclusive strategy should be 'I'
            assert df_out.loc[0, "consensus"] == "I"
            # Row 1 was 2-1 majority (F, F, C), should be 'F'
            assert df_out.loc[1, "consensus"] == "F"

    def test_determine_majority_label_empty_df(self):
        """Tests that passing an empty DataFrame to determine_majority_label returns an empty DataFrame with label column."""
        evaluator = AgreementEvaluator()
        empty_df = pd.DataFrame(columns=["Arthur", "Thomas", "Gwendal"])
        res = evaluator.determine_majority_label(empty_df)
        assert len(res) == 0
        assert "label" in res.columns

    def test_determine_majority_label_nan_warning(self, caplog):
        """Tests that passing data with NaNs logs a warning recommending clean_and_filter."""
        import logging
        evaluator = AgreementEvaluator()
        df_nan = pd.DataFrame(
            {
                "Arthur": [None, "F"],
                "Thomas": ["F", "F"],
                "Gwendal": ["F", "F"],
            }
        )
        with caplog.at_level(logging.WARNING):
            res = evaluator.determine_majority_label(df_nan)
            assert any("Input DataFrame contains NaN values" in record.message for record in caplog.records)
            assert len(res) == 2

    def test_determine_majority_label_first_rater_avoids_minority_candidate(self):
        """Adversarial check: first_rater tie-break must NOT select a minority candidate.

        Scenario: 5 raters where R1 voted for 'A' (1 vote), R2 and R3 voted for 'B' (2 votes),
        and R4 and R5 voted for 'C' (2 votes). The tie is strictly between 'B' and 'C'.
        Under first_rater priority among tied candidates, 'B' (the choice of R2) must win,
        not 'A' (which only received 1 vote).
        """
        evaluator = AgreementEvaluator(raters=["R1", "R2", "R3", "R4", "R5"])
        df = pd.DataFrame([{"R1": "A", "R2": "B", "R3": "B", "R4": "C", "R5": "C"}])
        res = evaluator.determine_majority_label(df, tie_strategy="first_rater")
        assert res.loc[0, "label"] == "B"

    def test_determine_majority_label_nan_ignored_in_voting(self):
        """Tests that missing/NaN rater votes do not compete as phantom consensus categories."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame(
            {
                "Arthur": [None, None],
                "Thomas": ["F", None],
                "Gwendal": [None, None],
            }
        )
        res = evaluator.determine_majority_label(df)
        # Row 0: only Thomas voted ('F'), Arthur and Gwendal are NaN -> majority is 'F'
        assert res.loc[0, "label"] == "F"
        # Row 1: all raters are NaN -> falls back to 'I'
        assert res.loc[1, "label"] == "I"

    def test_interpret_kappa_rejects_nan_and_inf_and_non_numeric(self):
        """Tests that interpret_kappa raises ValueError for NaN/Inf and TypeError for non-numeric input."""
        evaluator = AgreementEvaluator()

        with pytest.raises(ValueError, match="Cannot interpret non-finite kappa score"):
            evaluator.interpret_kappa(float("nan"))

        with pytest.raises(ValueError, match="Cannot interpret non-finite kappa score"):
            evaluator.interpret_kappa(float("inf"))

        with pytest.raises(ValueError, match="Cannot interpret non-finite kappa score"):
            evaluator.interpret_kappa(float("-inf"))

        with pytest.raises(TypeError, match="Kappa score must be numeric"):
            evaluator.interpret_kappa("not_a_number")

    def test_validate_matrix_rejects_fractional_rater_counts(self):
        """Tests that non-integer / fractional rater counts are rejected."""
        evaluator = AgreementEvaluator()
        fractional_matrix = [[2.5, 0.5], [1.5, 1.5]]
        with pytest.raises(ValueError, match="counts must be non-negative integers"):
            evaluator.compute_fleiss_kappa(fractional_matrix)

    def test_validate_matrix_rejects_duplicate_dataframe_columns(self):
        """Tests that contingency DataFrames with duplicate column names are rejected."""
        evaluator = AgreementEvaluator()
        df_duplicate_cols = pd.DataFrame([[3, 0], [0, 3]], columns=["I", "I"])
        with pytest.raises(ValueError, match="columns must be unique category names"):
            evaluator.compute_fleiss_kappa(df_duplicate_cols)

    def test_initialization_validation_errors(self):
        """Tests that AgreementEvaluator.__init__ validates minimum count and uniqueness."""
        # Fewer than 2 raters
        with pytest.raises(ValueError, match="requires at least 2 raters"):
            AgreementEvaluator(raters=["OnlyOne"])

        # Duplicate raters
        with pytest.raises(ValueError, match="must contain unique rater names"):
            AgreementEvaluator(raters=["Arthur", "Arthur"])

        # Fewer than 2 categories
        with pytest.raises(ValueError, match="requires at least 2 categories"):
            AgreementEvaluator(categories=["OnlyOne"])

        # Duplicate categories
        with pytest.raises(ValueError, match="must contain unique category labels"):
            AgreementEvaluator(categories=["F", "F"])

    def test_load_data_duplicate_columns_error(self):
        """Tests that load_data raises ValueError if the CSV file contains duplicate column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "dup_cols.csv"
            csv_path.write_text("Arthur;Arthur;Thomas;Gwendal\nF;F;F;F\n", encoding="utf-8")
            evaluator = AgreementEvaluator()
            with pytest.raises(ValueError, match="contains duplicate column headers"):
                evaluator.load_data(csv_path, delimiter=";")

    def test_clean_and_filter_duplicate_rater_columns_error(self):
        """Tests that clean_and_filter raises ValueError if the DataFrame contains duplicate target rater columns."""
        evaluator = AgreementEvaluator(raters=["Arthur", "Thomas", "Gwendal"])
        df = pd.DataFrame([["F", "F", "F", "F"]], columns=["Arthur", "Arthur", "Thomas", "Gwendal"])
        with pytest.raises(ValueError, match="duplicate columns matching target rater names"):
            evaluator.clean_and_filter(df)

    def test_build_contingency_matrix_categories_validation(self):
        """Tests that build_contingency_matrix rejects < 2 categories or duplicate categories."""
        evaluator = AgreementEvaluator()
        df = pd.DataFrame({"Arthur": ["F"], "Thomas": ["F"], "Gwendal": ["F"]})

        with pytest.raises(ValueError, match="requires at least 2 categories"):
            evaluator.build_contingency_matrix(df, categories=["F"])

        with pytest.raises(ValueError, match="Categories must be unique"):
            evaluator.build_contingency_matrix(df, categories=["F", "F"])

    def test_compute_detailed_metrics_sets_last_kappa(self):
        """Tests that compute_detailed_metrics synchronizes self.last_kappa."""
        evaluator = AgreementEvaluator()
        matrix = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1]])
        metrics = evaluator.compute_detailed_metrics(matrix)
        assert evaluator.last_kappa == metrics["kappa"]
