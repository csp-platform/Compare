import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from glob import glob
from pathlib import Path
import json


from constants.constants import (OUTPUTS_DIR, NUM_TO_STR, GRAPH_COMPARISON_FOLDER_NAME, EMBEDDINGS_GRAPHS_FOLDER,
    MEAN_METRIC, MIN_METRIC, FIGURE_SIZE, LABEL_FONT_SIZE, TITLE_FONT_SIZE, GRAPH_LEFT_MARGIN,CONFIDENCE_INTERVAL,
    SETTING_PLOT_FIRST_SECOND_DISTANCE_OVER_TIME, SETTING_PLOT_RANK_DIAMETER_AND_AVERAGE_OVER_TIME,
    SETTING_WINNER_SIMILARITY_OVER_TIME, SETTING_AVERAGE_UNIQUE_DOCUMENTS_OVER_TIME,
    SETTING_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS, SETTING_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS,
    SETTING_PLOT_DIAMETER_AND_AVERAGE_OVER_TIME,
    ROUND_X_LABEL, ROUNDS_INCLUDED_X_LABEL,
    FIRST_SECOND_SIMILARITY_OVER_TIME_FILE_NAME, RANK_DIAMETER_AND_AVERAGE_OVER_TIME_FILE_NAME,
    Y_LABEL_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME, TITLE_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME, CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME_FILE_NAME,
    Y_LABEL_AVERAGE_UNIQUE_DOCUMENTS, TITLE_LABEL_AVERAGE_UNIQUE_DOCUMENTS, AVERAGE_UNIQUE_DOCUMENTS_FILE_NAME,
    Y_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS, TITLE_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS,
    Y_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS, TITLE_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS, AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_FILE_NAME,
    Y_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS, TITLE_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS, AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUND_FILE_NAME,
    Y_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME, TITLE_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME,
    Y_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME, TITLE_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME, DIAMETER_AND_AVERAGE_OVER_TIME)


class GraphsCompare:
    def __init__(self, experiment_folders: Dict[str, str]) -> None:
        """
        Initialize the GraphsCompare object with a list of experiment folders.
        Creates an output directory for saving comparison plots.

        Parameters:
            experiment_folders (Dict[str, str]): Dictionary of experiment names and their corresponding paths.
        """
        self.__experiment_folders = list(experiment_folders.values())
        self.__experiment_ids = list(experiment_folders.keys())
        # Create output folder based on experiment names
        experiment_names = '-'.join([os.path.basename(folder) for folder in experiment_folders])
        self.__output_folder = os.path.join(OUTPUTS_DIR, experiment_names, GRAPH_COMPARISON_FOLDER_NAME)
        os.makedirs(self.__output_folder, exist_ok=True)

    def plot_all_graphs(self) -> None:
        """
        Generate and save plots for all settings across experiments.
        """
        self.__plot_first_second_similarity_over_time()
        self.__plot_rank_diameter_and_average_over_time()
        self.__plot_consecutive_winner_similarity_over_time()
        self.__plot_average_unique_documents()
        self.__plot_average_and_diameter_of_player_documents()
        self.__plot_average_similarity_of_player_documents_consecutive_rounds()
        self.__plot_diameter_and_average_over_time()


    def __get_settings(self, folder: str, subfolder: str) -> List[str]:
        """
        Retrieve unique settings based on .npy filenames in a specified subfolder.

        Parameters:
            folder (str): Folder path to check for unique settings.
            subfolder (str): Subfolder name where .npy files are located.

        Returns:
            List[str]: List of unique settings found in the directory.
        """
        path_pattern = os.path.join(folder, EMBEDDINGS_GRAPHS_FOLDER, subfolder, "*.npy")
        files = glob(path_pattern)
        # Extract settings identifiers from filenames
        settings = {os.path.basename(file).rsplit('-', 1)[0] + '-' for file in files}
        return list(settings)


    def __plot_and_save(self, X: List[np.ndarray], ys: List[np.ndarray], labels: List[str],
                        title: str, xlabel: str, ylabel: str, save_path: str,
                        yerrs: List[np.ndarray] = None) -> None:
        """
        Generalized plotting and saving function to plot multiple datasets on the same graph.

        Parameters:
            X (List[np.ndarray]): X-axis dataset to plot.
            ys (List[np.ndarray]): List of Y-axis datasets to plot.
            labels (List[str]): List of labels for each dataset.
            title (str): Title of the plot.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
            save_path (str): Path to save the plot.
            yerrs (List[np.ndarray], optional): List of Y-axis error bars for each dataset. Defaults to None.
        """
        plt.figure(figsize=FIGURE_SIZE)
        handles = []
        for i, y in enumerate(ys):
            if yerrs and yerrs[i] is not None:
                # Plot with error bars
                handle = plt.errorbar(X[i], y, yerr=yerrs[i], label=labels[i], fmt='-o')
            else:
                # Plot without error bars
                handle, = plt.plot(X[i], y, label=labels[i])
            handles.append(handle)

        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)

        title_path = save_path.replace(".png", "_title.txt")
        with open(title_path, 'w') as f:
            f.write(title)

        plt.subplots_adjust(left=GRAPH_LEFT_MARGIN)
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

        # Legend saved separately
        fig_legend = plt.figure(figsize=(4, len(labels) * 0.4))  # Adjust size for readability
        fig_legend.legend(handles, self.__experiment_ids, loc='center', fontsize=LABEL_FONT_SIZE)
        fig_legend.gca().axis('off')  # Remove axes
        fig_legend.savefig(self.__output_folder, bbox_inches='tight')
        plt.close(fig_legend)

        # Prepare the data for ECharts
        series = []
        all_y_values = []
        for i, (x, y) in enumerate(zip(X, ys)):
            data = list(zip(x.tolist(), y.tolist()))
            all_y_values.extend(y.tolist())
            series_item = {
                "type": "line",
                "smooth": True,
                "name": self.__experiment_ids[i],
                "data": data
            }
            # If there are error bars, you can incorporate them using markArea or other ECharts features
            if yerrs and yerrs[i] is not None:
                # Example: Adding markArea for error ranges
                error_data = list(zip(x.tolist(), (y - yerrs[i]).tolist()))
                x = x.tolist()
                y = y.tolist()
                series_item["markArea"] = {
                    "data": [
                        [
                            {"coord": [x[0], y[i] - yerrs[i][i]]},
                            {"coord": [x[-1], y[i] + yerrs[i][i]]}
                        ]
                    ]
                }
            series.append(series_item)

        global_min = min(all_y_values)
        global_max = max(all_y_values)
        padding = (global_max - global_min) * 0.1  # 10% padding
        y_axis_min = global_min - padding
        y_axis_max = global_max + padding

        # Define the ECharts option
        option = {
            "title": {"text": title},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": self.__experiment_ids},
            "dataZoom": [
                {
                    "type": "inside",
                    "start": 0,
                    "end": 100
                },
                {
                    "start": 0,
                    "end": 100
                }
            ],
            "xAxis": {
                "type": "category",
                "name": xlabel,
                "data": X[0].tolist()  # Assuming all X are the same
            },
            "yAxis": {
                "type": "value",
                "name": ylabel
            },
            "series": series
        }

        option["yAxis"]["min"] = y_axis_min
        option["yAxis"]["max"] = y_axis_max

        option["yAxis"]["axisLine"] = {"lineStyle": {"color": "white"}}
        option["xAxis"]["axisLine"] = {"lineStyle": {"color": "white"}}
        option["yAxis"]["splitLine"] = {"lineStyle": {"color": "white"}}
        option["xAxis"]["splitLine"] = {"lineStyle": {"color": "white"}}
        option["yAxis"]["nameTextStyle"] = {"color": "white"}
        option["xAxis"]["nameTextStyle"] = {"color": "white"}
        option["yAxis"]["axisLabel"] = {"color": "white"}
        option["xAxis"]["axisLabel"] = {"color": "white"}
        option["yAxis"]["nameTextStyle"] = {"color": "white"}
        option["title"]["textStyle"] = {"color": "white"}
        option["legend"]["textStyle"] = {"color": "white"}



        # Ensure the save_path directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the option as a JSON file
        with open(save_path.replace(".png", ".json"), 'w') as f:
            json.dump(option, f, indent=4)

    def __load_data(self, folder: str, subfolder: str, setting: str) -> np.ndarray:
        """
        Load data from .npy files for a specific setting.

        Parameters:
            folder (str): Path to the experiment folder.
            subfolder (str): Subfolder name where .npy files are located.
            setting (str): Setting identifier.

        Returns:
            np.ndarray: Loaded data.
        """
        npy_files = glob(os.path.join(folder, EMBEDDINGS_GRAPHS_FOLDER, subfolder, f"{setting}*.npy"))
        if npy_files:
            return np.load(npy_files[0], allow_pickle=True)
        return None

    def __extract_setting_info(self, setting: str) -> tuple:
        """
        Extract rank and representation information from the setting string.

        Parameters:
            setting (str): Setting identifier.

        Returns:
            tuple: (ranks, representation_name)
        """
        setting_parts = setting.rstrip('-').split('-')

        if len(setting_parts) == 3:
            ranks = list(map(int, setting_parts[:2]))
        elif len(setting_parts) == 2:
            ranks = int(setting_parts[0])
        else:
            ranks = None

        representation_name = setting_parts[-1] if setting_parts else ''
        return ranks, representation_name

    def __plot_first_second_similarity_over_time(self) -> None:
        """
        Plot the average similarity between the first and second ranked documents over time.
        """
        subfolder = SETTING_PLOT_FIRST_SECOND_DISTANCE_OVER_TIME
        settings = self.__get_settings(self.__experiment_folders[0], subfolder)

        for setting in settings:
            ys = []
            yerrs = []
            labels = []
            X = []

            for folder in self.__experiment_folders:
                data = self.__load_data(folder, subfolder, setting)
                if data is not None:
                    experiment_name = os.path.basename(folder)

                    # Compute average and standard deviation for each round
                    avg_metric = []
                    std_metric = []
                    round_lengths = []
                    for round_data in data:
                        round_data_array = np.array(round_data)
                        avg_metric.append(round_data_array.mean())
                        std_metric.append(round_data_array.std())
                        round_lengths.append(len(round_data_array))

                    avg_metric = np.array(avg_metric)
                    std_metric = np.array(std_metric)
                    round_lengths = np.array(round_lengths)

                    # Compute 95% confidence interval
                    yerr = CONFIDENCE_INTERVAL * std_metric / np.sqrt(round_lengths)

                    ys.append(avg_metric)
                    yerrs.append(yerr)
                    labels.append(experiment_name)
                    X.append(np.arange(1, len(avg_metric) + 1))

            # Extract rank range and representation from setting
            ranks, representation_name = self.__extract_setting_info(setting)

            # Set plot labels and title
            title = f"Average {NUM_TO_STR[ranks[0]]}-{NUM_TO_STR[ranks[1]]} ranked players \nsimilarity vs round"
            ylabel = f"Average {NUM_TO_STR[ranks[0]]}-{NUM_TO_STR[ranks[1]]} ranked players similarity"

            # Save plot
            subfolder_path = os.path.join(self.__output_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            output_filename = setting + FIRST_SECOND_SIMILARITY_OVER_TIME_FILE_NAME
            output_path = os.path.join(subfolder_path, output_filename)

            # Plot and save
            self.__plot_and_save(X, ys, labels, title, ROUND_X_LABEL, ylabel, output_path, yerrs=yerrs)

    def __plot_rank_diameter_and_average_over_time(self) -> None:
        """
        Plot rank-based metrics (mean and min) over time for each setting across experiments.
        """
        subfolders = {
            MEAN_METRIC: f"{SETTING_PLOT_RANK_DIAMETER_AND_AVERAGE_OVER_TIME}-{MEAN_METRIC}",
            MIN_METRIC: f"{SETTING_PLOT_RANK_DIAMETER_AND_AVERAGE_OVER_TIME}-{MIN_METRIC}",
        }
        settings = set()
        for metric in [MEAN_METRIC, MIN_METRIC]:
            settings.update(self.__get_settings(self.__experiment_folders[0], subfolders[metric]))

        for setting in settings:
            for metric in [MEAN_METRIC, MIN_METRIC]:
                ys = []
                labels = []
                X = []
                for folder in self.__experiment_folders:
                    data = self.__load_data(folder, subfolders[metric], setting)
                    if data is not None:
                        experiment_name = os.path.basename(folder)
                        avg_metric = [np.mean(round_data) for round_data in data]
                        ys.append(np.array(avg_metric))
                        labels.append(experiment_name)
                        X.append(np.arange(2, len(avg_metric) + 2))

                # Extract rank and representation from setting
                rank, representation_name = self.__extract_setting_info(setting)

                # Set plot labels and title
                if metric == MEAN_METRIC:
                    ylabel = f'Average {NUM_TO_STR[rank]}-ranked players similarity'
                    title = f"Average {NUM_TO_STR[rank]}-ranked players \nsimilarity vs round"
                else:
                    ylabel = f'Diameter {NUM_TO_STR[rank]}-ranked players similarity'
                    title = f"Diameter {NUM_TO_STR[rank]}-ranked players \nsimilarity vs round"

                # Save plot
                subfolder_path = os.path.join(self.__output_folder, subfolders[metric])
                os.makedirs(subfolder_path, exist_ok=True)
                output_filename = setting + RANK_DIAMETER_AND_AVERAGE_OVER_TIME_FILE_NAME.replace(".png", f"-{metric}.png")
                output_path = os.path.join(subfolder_path, output_filename)

                # Plot and save
                self.__plot_and_save(X, ys, labels, title, ROUNDS_INCLUDED_X_LABEL, ylabel, output_path)

    def __plot_consecutive_winner_similarity_over_time(self) -> None:
        """
        Plot the average similarity between consecutive winners over time.
        """
        subfolder = SETTING_WINNER_SIMILARITY_OVER_TIME
        settings = self.__get_settings(self.__experiment_folders[0], subfolder)

        for setting in settings:
            ys = []
            labels = []
            X = []
            for folder in self.__experiment_folders:
                data = self.__load_data(folder, subfolder, setting)
                if data is not None:
                    experiment_name = os.path.basename(folder)
                    avg_metric = []
                    for round_data in data:
                        round_data_array = np.array(round_data)
                        round_data_array = round_data_array[round_data_array < 2]
                        avg_metric.append(round_data_array.mean())
                    ys.append(np.array(avg_metric))
                    labels.append(experiment_name)
                    X.append(np.arange(2, len(avg_metric) + 2))

            # Extract representation from setting
            _, representation_name = self.__extract_setting_info(setting)

            # Save plot
            subfolder_path = os.path.join(self.__output_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            output_filename = setting + CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME_FILE_NAME
            output_path = os.path.join(subfolder_path, output_filename)

            # Plot and save
            self.__plot_and_save(X, ys, labels, TITLE_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME,
                                 ROUND_X_LABEL, Y_LABEL_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME, output_path)

    def __plot_average_unique_documents(self) -> None:
        """
        Plot the average number of unique documents per query by round.
        """
        subfolder = SETTING_AVERAGE_UNIQUE_DOCUMENTS_OVER_TIME
        ys = []
        labels = []
        X = []

        for folder in self.__experiment_folders:
            data = self.__load_data(folder, subfolder, '')
            if data is not None:
                experiment_name = os.path.basename(folder)
                avg_unique_docs = [counts.mean() for counts in data]
                ys.append(np.array(avg_unique_docs))
                labels.append(experiment_name)
                X.append(np.arange(1, len(avg_unique_docs) + 1))

        # Save plot
        subfolder_path = os.path.join(self.__output_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        output_path = os.path.join(subfolder_path, AVERAGE_UNIQUE_DOCUMENTS_FILE_NAME)

        # Plot and save
        self.__plot_and_save(X, ys, labels, TITLE_LABEL_AVERAGE_UNIQUE_DOCUMENTS, ROUND_X_LABEL,
                             Y_LABEL_AVERAGE_UNIQUE_DOCUMENTS, output_path)

    def __plot_average_and_diameter_of_player_documents(self) -> None:
        """
        Plot the average similarity and diameter of player documents over time.
        """
        subfolders = {
            MEAN_METRIC: f"{SETTING_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS}-{MEAN_METRIC}",
            MIN_METRIC: f"{SETTING_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS}-{MIN_METRIC}",
        }
        settings = set()
        for metric in [MEAN_METRIC, MIN_METRIC]:
            settings.update(self.__get_settings(self.__experiment_folders[0], subfolders[metric]))

        for setting in settings:
            for metric in [MEAN_METRIC, MIN_METRIC]:
                ys = []
                labels = []
                X = []
                for folder in self.__experiment_folders:
                    data = self.__load_data(folder, subfolders[metric], setting)
                    if data is not None:
                        experiment_name = os.path.basename(folder)
                        data_swapped = data.swapaxes(1, 2)
                        avg_metric = data_swapped.mean(axis=2).mean(axis=0)
                        ys.append(avg_metric)
                        labels.append(experiment_name)
                        X.append(np.arange(2, len(avg_metric) + 2))

                # Extract representation from setting
                _, representation_name = self.__extract_setting_info(setting)

                # Set plot labels and title
                if metric == MEAN_METRIC:
                    ylabel = Y_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS
                    title = TITLE_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS
                else:
                    ylabel = Y_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS
                    title = TITLE_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS

                # Save plot
                subfolder_path = os.path.join(self.__output_folder, subfolders[metric])
                os.makedirs(subfolder_path, exist_ok=True)
                output_filename = setting + metric + "_" + AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_FILE_NAME
                output_path = os.path.join(subfolder_path, output_filename)

                # Plot and save
                self.__plot_and_save(X, ys, labels, title, ROUNDS_INCLUDED_X_LABEL, ylabel, output_path)

    def __plot_average_similarity_of_player_documents_consecutive_rounds(self) -> None:
        """
        Plot the average similarity of player documents between consecutive rounds.
        """
        subfolder = SETTING_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS
        settings = self.__get_settings(self.__experiment_folders[0], subfolder)

        for setting in settings:
            ys = []
            labels = []
            X = []
            for folder in self.__experiment_folders:
                data = self.__load_data(folder, subfolder, setting)
                if data is not None:
                    experiment_name = os.path.basename(folder)
                    data_swapped = data.swapaxes(1, 2)
                    avg_metric = data_swapped.mean(axis=2).mean(axis=0)
                    ys.append(avg_metric)
                    X.append(np.arange(2, len(avg_metric) + 2))
                    labels.append(experiment_name)

            # Extract representation from setting
            _, representation_name = self.__extract_setting_info(setting)

            # Set plot labels and title
            ylabel = Y_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS
            title = TITLE_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS

            # Save plot
            subfolder_path = os.path.join(self.__output_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            output_filename = setting + AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUND_FILE_NAME
            output_path = os.path.join(subfolder_path, output_filename)

            # Plot and save
            self.__plot_and_save(X, ys, labels, title, ROUND_X_LABEL, ylabel, output_path)

    def __plot_diameter_and_average_over_time(self) -> None:
        """
        Plot the diameter and average similarity over time.
        """
        subfolders = {
            MEAN_METRIC: f"{SETTING_PLOT_DIAMETER_AND_AVERAGE_OVER_TIME}-{MEAN_METRIC}",
            MIN_METRIC: f"{SETTING_PLOT_DIAMETER_AND_AVERAGE_OVER_TIME}-{MIN_METRIC}",
        }
        settings = set()
        for metric in [MEAN_METRIC, MIN_METRIC]:
            settings.update(self.__get_settings(self.__experiment_folders[0], subfolders[metric]))

        for setting in settings:
            for metric in [MEAN_METRIC, MIN_METRIC]:
                ys = []
                labels = []
                X = []
                for folder in self.__experiment_folders:
                    data = self.__load_data(folder, subfolders[metric], setting)
                    if data is not None:
                        experiment_name = os.path.basename(folder)
                        avg_metric = [np.mean(round_data) for round_data in data]
                        ys.append(np.array(avg_metric))
                        labels.append(experiment_name)
                        X.append(np.arange(1, len(avg_metric) + 1))

                # Extract representation from setting
                _, representation_name = self.__extract_setting_info(setting)

                # Set plot labels and title
                if metric == MEAN_METRIC:
                    ylabel = Y_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME
                    title = TITLE_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME
                else:
                    ylabel = Y_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME
                    title = TITLE_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME
                # Save plot
                subfolder_path = os.path.join(self.__output_folder, subfolders[metric])
                os.makedirs(subfolder_path, exist_ok=True)
                output_filename = setting + metric + "_" + DIAMETER_AND_AVERAGE_OVER_TIME
                output_path = os.path.join(subfolder_path, output_filename)

                # Plot and save
                self.__plot_and_save(X, ys, labels, title, ROUND_X_LABEL, ylabel, output_path)