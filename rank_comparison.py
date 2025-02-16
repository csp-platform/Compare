import os
import pandas as pd
import numpy as np
from typing import List, Dict
from constants.constants import (RANKS_COMPARISON_FOLDER_NAME, OUTPUTS_DIR, COMPETITION_HISTORY_PILOT_FILE_NAME,
                                 USER_QUERY_COLUMN, RANK_CHANGE_ANALYSIS_FILE_NAME, TRANSITION_PROBABILITIES_ANALYSIS_FILE_NAME,
                                 SPERATION_LINE)


class RankComparison:
    def __init__(self, experiment_paths: Dict[str, str]):
        """
        Initializes RankComparison with specified experiment folders.

        Parameters:
        - experiment_paths (Dict[str, str]): Dictionary of experiment names and paths to compare.
        """
        self.__experiment_paths = list(experiment_paths.values())
        self.__experiment_ids = list(experiment_paths.keys())
        self.__experiments = {os.path.basename(v): k for k, v in experiment_paths.items()}
        self.__experiment_data = {}  # Stores competition history data for each experiment

        # Define output folder based on experiment names
        experiment_names = '-'.join([os.path.basename(path) for path in experiment_paths.keys()])
        self.output_dir = os.path.join(OUTPUTS_DIR, experiment_names, RANKS_COMPARISON_FOLDER_NAME)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load competition history data from each experiment folder
        for path in self.__experiment_paths:
            pivot_file_path = os.path.join(path, COMPETITION_HISTORY_PILOT_FILE_NAME)
            experiment_name = os.path.basename(path)
            if os.path.exists(pivot_file_path):
                data = pd.read_csv(pivot_file_path).set_index(USER_QUERY_COLUMN)
                self.__experiment_data[experiment_name] = data
            else:
                print(f"Warning: {pivot_file_path} not found.")

    def run_comparisons(self):
        """
        Runs full comparisons on rank changes and transition probabilities across all experiments.
        """
        self.__compare_rank_changes()
        self.__compare_transition_probabilities()

    def __compare_rank_changes(self):
        """
        Compares changes in rank positions between consecutive rounds across experiments.
        """
        for experiment, data in self.__experiment_data.items():
            analysis_df = self.__analyze_rank_changes(data, experiment)
            output_path = os.path.join(self.output_dir, f"{self.__experiments[experiment]}_{RANK_CHANGE_ANALYSIS_FILE_NAME}")
            analysis_df.to_csv(output_path, index=False)

    def __compare_transition_probabilities(self):
        """
        Compares transition probabilities, expected values, and standard deviations of rank transitions.
        """
        for experiment, data in self.__experiment_data.items():
            analysis_df = self.__analyze_transition_probabilities(data)
            output_path = os.path.join(self.output_dir, f"{self.__experiments[experiment]}_{TRANSITION_PROBABILITIES_ANALYSIS_FILE_NAME}")
            analysis_df.to_csv(output_path, index=False)

    def __analyze_rank_changes(self, df: pd.DataFrame, experiment_name: str = None) -> pd.DataFrame:
        """
        Analyzes rank position changes between consecutive rounds.

        Parameters:
        - df (DataFrame): Dataframe with rank data per round.
        - experiment_name (str, optional): Name of the experiment for display.

        Returns:
        - DataFrame: Analysis results in a DataFrame.
        """
        num_rounds = int(df.columns[-1])
        data = []

        # Analyze rank position changes across rounds
        for round_num in range(1, num_rounds):
            current_round = df[str(round_num)]
            next_round = df[str(round_num + 1)]
            changes = int((current_round != next_round).sum())
            change_magnitude = int((current_round - next_round).abs().sum())

            data.append({
                'Round': f"{round_num} to {round_num + 1}",
                'Changes': changes,
                'Sum of Changes': change_magnitude
            })

        analysis_df = pd.DataFrame(data)
        total_changes = analysis_df['Changes'].sum()
        overall_sum_of_changes = analysis_df['Sum of Changes'].sum()

        # Add total row
        total_row = pd.DataFrame({
            'Round': ['Total'],
            'Changes': [total_changes],
            'Sum of Changes': [overall_sum_of_changes]
        })
        analysis_df = pd.concat([analysis_df, total_row], ignore_index=True)

        return analysis_df

    def __calculate_transition_probabilities(self, df: pd.DataFrame, current_rank: int, max_rank: int, num_rounds: int, step: int = 1) -> np.ndarray:
        """
        Calculates transition probabilities for a specific rank across rounds.

        Parameters:
        - df (DataFrame): Dataframe with rank data per round.
        - current_rank (int): Rank from which transitions are calculated.
        - max_rank (int): Maximum rank.
        - num_rounds (int): Total number of rounds.
        - step (int): Step between rounds for transitions (default is 1 for consecutive rounds).

        Returns:
        - ndarray: Transition probability distribution for each rank from the current rank.
        """
        transitions = np.zeros(max_rank)
        for round_num in range(1, num_rounds + 1 - step):
            current_positions = df[str(round_num)] == current_rank
            for target_rank in range(1, max_rank + 1):
                transitions[target_rank - 1] += (current_positions & (df[str(round_num + step)] == target_rank)).sum()
        return transitions / transitions.sum() if transitions.sum() > 0 else transitions

    def __analyze_transition_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes transition probabilities for each rank.

        Parameters:
        - df (DataFrame): Dataframe with rank data per round.

        Returns:
        - DataFrame: Analysis of transition probabilities, expected values, and standard deviations.
        """
        max_rank = int(df.values[:, -1].max())
        num_rounds = int(df.columns[-1])
        data = []

        for rank in range(1, max_rank + 1):
            transition_probabilities = self.__calculate_transition_probabilities(df, rank, max_rank, num_rounds)
            expected_value = np.dot(transition_probabilities, np.arange(1, max_rank + 1))
            variance = np.dot(transition_probabilities, (np.arange(1, max_rank + 1) ** 2)) - expected_value ** 2
            standard_deviation = np.sqrt(variance)

            row = {
                'Rank': rank,
                'Expected Value': round(expected_value, 2),
                'Standard Deviation': round(standard_deviation, 2)
            }
            # Add transition probabilities for each possible rank
            for target_rank in range(1, max_rank + 1):
                row[f'P(Rank {target_rank})'] = round(transition_probabilities[target_rank - 1], 2)
            data.append(row)

        analysis_df = pd.DataFrame(data)
        return analysis_df
