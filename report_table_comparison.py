from typing import List, Dict
import os

import numpy as np
import pandas as pd

from constants.constants import (OUTPUTS_DIR, REPORT_TABLE_COMPARISON_FILE_NAME, EMBEDDINGS_GRAPHS_FOLDER,
                                 REPORT_TABLE_INPUT_CSV_FILE_NAME, AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_CSV_FILE_NAME,
                                 AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_CSV_FILE_NAME, RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_FILE_NAME,
                                 FINAL_TABLE_EXPERIMENT_COLUMN, FINAL_TABLE_BEST_AGENT_COLUMN,
                                 FINAL_TABLE_WINNING_HOMOGENEITY_COLUMN, FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MEAN_COLUMN,
                                 FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MIN_COLUMN,
                                 FINAL_TABLE_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_MEAN_COLUMN,
                                 FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MEAN_COLUMN,
                                 FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MIN_COLUMN)

class ReportTableComparison:
    def __init__(self, experiment_folders: Dict[str, str]):
        """
        Initializes ReportTableComparison with specified experiment folders.

        Parameters:
            experiment_folders (Dict[str, str]): Dictionary of experiment names and paths to compare.
        """
        self.__experiment_folders = experiment_folders
        self.__get_dfs()

        experiment_names = '-'.join([os.path.basename(path) for path in list(experiment_folders.keys())])
        self.__output_dir = os.path.join(OUTPUTS_DIR, experiment_names)
        os.makedirs(self.__output_dir, exist_ok=True)


    def __get_dfs(self):
        self.report_table_dfs = []
        self.avg_diameter_player_docs_dfs = []
        self.avg_diameter_player_consecutive_rounds_dfs = []
        self.rank_diameter_avg_dfs = []

        for experiment_name, experiment_folder in self.__experiment_folders.items():
            report_table_path = os.path.join(experiment_folder, REPORT_TABLE_INPUT_CSV_FILE_NAME)
            avg_diameter_player_docs_path = os.path.join(experiment_folder, EMBEDDINGS_GRAPHS_FOLDER,
                                                         AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_CSV_FILE_NAME)
            avg_diameter_player_consecutive_rounds_path = os.path.join(experiment_folder, EMBEDDINGS_GRAPHS_FOLDER,
                                                                       AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_CSV_FILE_NAME)
            rank_diameter_avg_path = os.path.join(experiment_folder, EMBEDDINGS_GRAPHS_FOLDER,
                                                  RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_FILE_NAME)

            report_table_df = pd.read_csv(report_table_path)
            avg_diameter_player_docs_df = pd.read_csv(avg_diameter_player_docs_path)
            avg_diameter_player_consecutive_rounds_df = pd.read_csv(avg_diameter_player_consecutive_rounds_path)
            rank_diameter_avg_df = pd.read_csv(rank_diameter_avg_path)

            self.report_table_dfs.append(report_table_df)
            self.avg_diameter_player_docs_dfs.append(avg_diameter_player_docs_df)
            self.avg_diameter_player_consecutive_rounds_dfs.append(avg_diameter_player_consecutive_rounds_df)
            self.rank_diameter_avg_dfs.append(rank_diameter_avg_df)

    def compare_report_tables(self):
        final_report_table = pd.DataFrame()
        for experiment_name, report_table_df, avg_diameter_player_docs_df, avg_diameter_player_consecutive_rounds_df, rank_diameter_avg_df \
                in zip(self.__experiment_folders.keys(), self.report_table_dfs, self.avg_diameter_player_docs_dfs,
                       self.avg_diameter_player_consecutive_rounds_dfs, self.rank_diameter_avg_dfs):

            report_table_df[FINAL_TABLE_EXPERIMENT_COLUMN] = experiment_name
            report_table_df = report_table_df.set_index(FINAL_TABLE_EXPERIMENT_COLUMN)

            report_table_df[FINAL_TABLE_BEST_AGENT_COLUMN] = report_table_df[FINAL_TABLE_BEST_AGENT_COLUMN].astype(str).iloc[0]
            report_table_df[FINAL_TABLE_WINNING_HOMOGENEITY_COLUMN] = report_table_df[FINAL_TABLE_WINNING_HOMOGENEITY_COLUMN].astype(np.float32).iloc[0]

            report_table_df[FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MEAN_COLUMN] = avg_diameter_player_docs_df[FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MEAN_COLUMN].astype(np.float32).iloc[0]
            report_table_df[FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MIN_COLUMN] = avg_diameter_player_docs_df[FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MIN_COLUMN].astype(np.float32).iloc[0]

            report_table_df[FINAL_TABLE_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_MEAN_COLUMN] = avg_diameter_player_consecutive_rounds_df[FINAL_TABLE_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_MEAN_COLUMN].astype(np.float32).iloc[0]

            report_table_df[FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MEAN_COLUMN] = rank_diameter_avg_df[FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MEAN_COLUMN].astype(np.float32).iloc[0]
            report_table_df[FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MIN_COLUMN] = rank_diameter_avg_df[FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MIN_COLUMN].astype(np.float32).iloc[0]

            final_report_table = pd.concat([final_report_table, report_table_df])

        final_report_table.to_csv(os.path.join(self.__output_dir, REPORT_TABLE_COMPARISON_FILE_NAME))