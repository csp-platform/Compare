<p align="center">
  <img src="extra/project_logo.png" width="200" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">LEMSS Compare</h1>
</p>
<p align="center">
    <em><code>► LEMSS Compare</code></em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
<p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#overview)
- [ Features](#features)
- [ Repository Structure](#repository-structure)
- [ Modules](#modules)
- [ Getting Started](#getting-started)
  - [ Installation](#installation)
  - [ Usage](#usage)
- [ Project Roadmap](#project-roadmap)
- [ Contributing](#contributing)
</details>
<hr>

##  Overview

<p>This project compares results from LEMSS Analyzer experiments, focusing on trends in diversity, convergence, ranking stability, and winner properties. It streamlines cross-experiment analysis with advanced visualizations, statistical comparisons, and detailed reports.</p>
---

##  Features

1. **Experiment Data Integration**  
   Facilitates the import of outputs from the LEMSS Analyzer project, including metrics, graphs, and analysis results. Automatically organizes data by experiment for streamlined comparison.

2. **Graph Comparison and Overlay**  
   Enables side-by-side and overlaid visualization of key graphs from different experiments. Supports interactive exploration of trends, allowing direct comparison of competition diversity, convergence, and other metrics.

3. **Metric Aggregation and Comparison**  
   Aggregates metrics across experiments and computes statistical comparisons. Highlights significant differences and trends between experiments.

4. **Cross-Experiment Analysis**  
   Tracks patterns and variations across experiments, focusing on key dimensions like ranking stability, diversity, and winner properties. Provides insights into the impact of experimental conditions on outcomes.

5. **Rank Comparison and Transition Analysis**  
   Compares player rankings between experiments and analyzes transitions over time. Identifies changes in player performance, stability, and convergence, shedding light on the dynamics of competition.

6. **Customizable Comparison Metrics**  
    Allows users to define custom comparison metrics, enabling tailored analysis of specific research questions or hypotheses.

7. **Interactive Website for Data Exploration**  
   Offers a user-friendly, interactive website where users can explore comparisons, trends, and insights dynamically. Visualizations and reports are accessible in an engaging and intuitive format.

8. **Detailed Comparison Reports**  
   Generates comprehensive comparison reports summarizing differences and similarities across experiments. Reports include side-by-side tables, graphs, and statistical insights.

9. **Extensible and Modular Design**  
   Provides a flexible architecture for incorporating additional metrics, visualizations, and analysis techniques. Supports seamless integration with future enhancements or external tools.

10. **Automated Data Processing Pipeline**  
   Streamlines the ingestion, cleaning, and preprocessing of competition data, ensuring efficient and accurate analysis workflows.

---

##  Repository Structure

```sh
└── Compare/
    ├── constants/
    │   └── constants.py
    ├── data/
    │   ├── <experiment_hash>/
    │   │   ├── embeddings_graphs/
    │   │   │   ├── average_and_diameter_of_player_documents–mean/
    │   │   │   ├── average_and_diameter_of_player_documents–min/
    │   │   │   ├── average_of_player_documents_consecutive_rounds/
    │   │   │   ├── average_unique_documents_over_time/
    │   │   │   ├── plot_diameter_and_average_over_time–mean/
    │   │   │   ├── plot_diameter_and_average_over_time–min/
    │   │   │   ├── plot_first_second_similarity_over_time/
    │   │   │   ├── plot_rank_diameter_and_average_over_time–mean/
    │   │   │   ├── plot_rank_diameter_and_average_over_time–min/
    │   │   │   ├── rank_diameter_and_average_last_round–mean/
    │   │   │   ├── rank_diameter_and_average_last_round–min/
    │   │   │   ├── winner_similarity_over_time/
    │   │   │   ├── average_and_diameter_of_player_documents.csv
    │   │   │   ├── average_of_player_documents_consecutive_rounds.csv
    │   │   │   └── rank_diameter_and_average_last_round.csv
    │   │   ├── competition_history_pivot.csv
    │   │   └── report_table.csv
    ├── extra/
    │   └── project_logo.png
    ├── output/
    │   ├── exp1-exp2/
    │   │   ├── graphs_comparison/
    │   │   │   ├── average_and_diameter_of_player_documents–mean/
    │   │   │   ├── average_and_diameter_of_player_documents–min/
    │   │   │   ├── average_of_player_documents_consecutive_rounds/
    │   │   │   ├── average_unique_documents_over_time/
    │   │   │   ├── plot_diameter_and_average_over_time–mean/
    │   │   │   ├── plot_diameter_and_average_over_time–min/
    │   │   │   ├── plot_first_second_similarity_over_time/
    │   │   │   ├── plot_rank_diameter_and_average_over_time–mean/
    │   │   │   ├── plot_rank_diameter_and_average_over_time–min/
    │   │   │   └── winner_similarity_over_time/
    │   │   ├── ranks_comparison/
    │   │   │   ├── rank_change_analysis.txt
    │   │   │   └── transition_probabilities_analysis.txt
    │   │   └── final_report_table.csv
    ├── .gitignore
    ├── app.py
    ├── experiment_comparison.py
    ├── graphs_compare.py
    ├── main.py
    ├── rank_comparison.py
    ├── readme.md
    ├── report_table_comparison.py
    └── requirements.txt
    
```

---

## Modules

<details closed><summary>constants</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [constants.py](constants/constants.py)      | Contains constant values and configurations used throughout the project, ensuring consistency and ease of maintenance. |

</details>

<details closed><summary>data</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [<experiment_hash>](data/)                  | Contains data generated by the competition, including embeddings, graphs, and reports for each experiment. |

</details>

<details closed><summary>extra</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [project_logo.png](extra/project_logo.png)  | Logo image used for the project. |

</details>

<details closed><summary>output</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [exp1-exp2](output/exp1-exp2/)              | Contains comparison data between two experiments, including graphs, ranks, and final reports. |

</details>

<details closed><summary>app.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [app.py](app.py)                            | Streamlit web application for interactive data exploration and visualization of comparison results. |

</details>

<details closed><summary>experiment_comparison.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [experiment_comparison.py](experiment_comparison.py) | Compares two experiments and generates a final report table with the results. |

</details>

<details closed><summary>graphs_compare.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [graphs_compare.py](graphs_compare.py)      | Compares graphs generated by two experiments and visualizes the differences between them. |

</details>

<details closed><summary>main.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [main.py](main.py)                          | Main script that runs the analysis on the competition data, generating insights and reports. |

</details>

<details closed><summary>rank_comparison.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [rank_comparison.py](rank_comparison.py)    | Compares the ranks of two experiments and analyzes the differences between them. |

</details>

<details closed><summary>report_table_comparison.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [report_table_comparison.py](report_table_comparison.py) | Compares the report tables of two experiments and highlights the differences between them. |

</details>

<details closed><summary>requirements.txt</summary>
    
| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [requirements.txt](requirements.txt)        | Contains the list of dependencies required for the project to run successfully. |
    
</details>


---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10+`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the LEMSS Compare repository:
>
> ```console
> $ git clone ../LEMSS-Compare
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd Compare
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```
> 
> 4. Update all dependencies:
> ```console
> $ pip install --upgrade -r requirements.txt
> ```

###  Usage
> 1. Update experiment_folders in main.py with the paths to the experiment folders you want to compare and the name of the experiments.
>
> 2. Run LEMSS Compare using the command below:
> ```console
> $ python main.py
> ```
> 3. Run LEMSS Compare website using the command below:
> ```console
> $ streamlit run app.py
> ```


---

##  Project Roadmap

- [ ] `► Clean the code.`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/LEMSS2025/LEMSS-Compare/issues)**: Submit bugs found or log feature requests for the `LEMSS Compare` project.
- **[Submit Pull Requests](https://github.com/LEMSS2025/LEMSS-Compare/pulls)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/LEMSS2025/LEMSS-Compare/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your local account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone ../LEMSS-Compare
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to local**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>