from typing import Dict
import os

from graphs_compare import GraphsCompare
from rank_comparison import RankComparison
from report_table_comparison import ReportTableComparison
from constants.constants import OUTPUTS_DIR, GRAPH_COMPARISON_FOLDER_NAME

class ExperimentComparison:
    """
    Aggregates and compares results across multiple experiments,
    handling both rank and graph comparisons.
    """

    def __init__(self, experiment_paths: Dict[str, str]):
        """
        Initializes ExperimentComparison with paths to experiment folders.

        Parameters:
        - experiment_paths (List[str]): List of paths to experiment folders to compare.
        """
        self.experiment_paths = experiment_paths
        self._graph_comparator = GraphsCompare(experiment_paths)
        self._rank_comparator = RankComparison(experiment_paths)
        self._report_table_comparator = ReportTableComparison(experiment_paths)

    def execute_comparisons(self):
        """
        Executes rank and graph comparisons across all specified experiments.
        """
        # Run the ranking comparison across experiments
        self._rank_comparator.run_comparisons()

        # Generate and display comparison graphs for all experiments
        self._graph_comparator.plot_all_graphs()

        # Generate and display comparison tables for all experiments
        self._report_table_comparator.compare_report_tables()

    def get_output_dir(self):
        """
        Returns the output directory where comparison results are saved.

        Returns:
        - str: Path to the output directory.
        """
        experiment_names = '-'.join([os.path.basename(folder) for folder in self.experiment_paths])
        return os.path.join(OUTPUTS_DIR, experiment_names)

