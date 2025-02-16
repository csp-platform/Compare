import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import json
from streamlit_echarts import st_echarts
from typing import Dict
import os

from experiment_comparison import ExperimentComparison
from constants.constants import GRAPH_COMPARISON_FOLDER_NAME, REPORT_TABLE_COMPARISON_FILE_NAME


@st.cache_data
def load_echarts_option(json_path: Path) -> Dict:
    """Load and cache ECharts option from a JSON file."""
    with open(json_path, 'r') as f:
        option = json.load(f)
    return option


def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(
        layout="wide",
        page_title="Experiment Comparison Dashboard",
        page_icon="📊"
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='font-size: 40px;'>Experiment Comparison Dashboard</h1>
            <p style='font-size: 18px;'>A tool for in-depth analysis and visualization of experiment results</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom CSS to adjust ECharts iframe height
    streamlit_style = """
        <style>
        iframe[title="streamlit_echarts.st_echarts"]{ height: 600px;} 
        </style>
        """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        @media screen and (max-width: 768px) {
            iframe[title="streamlit_echarts.st_echarts"] { height: 400px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Define the base data directory
    DATA_DIR = Path("data")

    # Input fields in the sidebar
    with st.sidebar:
        # st.image("extra/project_logo.png", use_container_width=True, caption="Research Group Name")
        st.markdown("---")
        st.header('Experiment Configuration')

        num_experiments = st.number_input(
            'Number of Experiments',
            min_value=1,
            value=2,
            help="Select the number of experiments you want to compare (minimum 1)."
        )

        # Get the list of available experiment folders in the data directory
        if DATA_DIR.exists() and DATA_DIR.is_dir():
            all_folders = [f.name for f in DATA_DIR.iterdir() if f.is_dir()]
            if not all_folders:
                st.error(f"No experiment folders found in '{DATA_DIR}'.")
                st.stop()
        else:
            st.error(f"The data directory '{DATA_DIR}' does not exist.")
            st.stop()

        experiment_folders = {}
        experiment_names = []
        selected_folders = []  # Keep track of selected folders

        for i in range(int(num_experiments)):
            exp_name = st.text_input(
                f'Experiment {i+1} Name:',
                key=f'exp_name_{i}',
                help="Enter a unique name for the experiment."
            )

            # Exclude folders already selected for previous experiments
            available_folders = [folder for folder in all_folders if folder not in selected_folders]

            if not available_folders:
                st.warning("No more available experiment folders to select.")
                break

            exp_folder_name = st.selectbox(
                f'Select Experiment {i+1} Folder:',
                options=available_folders,
                key=f'exp_folder_{i}',
                help="Select the experiment folder from the data directory."
            )

            # Add the selected folder to the list to exclude it from next selections
            if exp_folder_name:
                selected_folders.append(exp_folder_name)

            exp_folder = DATA_DIR / exp_folder_name

            # Validate the folder path
            if exp_folder.is_dir():
                if exp_name:
                    experiment_folders[exp_name] = str(exp_folder.resolve())
                    experiment_names.append(exp_name)
                else:
                    st.error(f"Please enter a name for Experiment {i+1}.")
            else:
                st.error(f"Experiment {i+1} Folder '{exp_folder}' does not exist or is not a directory.")

        run_clicked = st.button('Run Comparisons', key='run_comparisons')

        # Store the experiment folders and names in session state
        if run_clicked:
            st.session_state.experiment_folders = experiment_folders
            st.session_state.experiment_names = experiment_names

        # Add a Reset button
        reset_clicked = st.button('Reset', key='reset_comparisons')
        if reset_clicked:
            st.session_state.clear()
            st.rerun()

    if run_clicked or 'comparison_completed' in st.session_state:
        if experiment_folders:
            # Check if the inputs have changed
            if (
                'experiment_folders' in st.session_state
                and st.session_state.experiment_folders != experiment_folders
            ):
                # Inputs have changed; reset comparison
                st.session_state.compare = None
                st.session_state.output_dir = None
                st.session_state.experiment_names = experiment_names
                st.session_state.comparison_completed = False

            # Check if the comparison has already been done
            if 'comparison_completed' not in st.session_state or not st.session_state.comparison_completed:
                with st.spinner('Running comparisons...'):
                    try:
                        compare = ExperimentComparison(experiment_folders)
                        compare.execute_comparisons()
                        # Store the comparison object and output directory in session state
                        st.session_state.compare = compare
                        st.session_state.output_dir = compare.get_output_dir()
                        st.session_state.experiment_names = experiment_names
                        st.session_state.comparison_completed = True
                    except Exception as e:
                        st.error(f"Error during comparison execution: {e}")
                        st.stop()
            else:
                # Load from session state
                compare = st.session_state.compare
                output_dir = st.session_state.output_dir
                experiment_names = st.session_state.experiment_names

            # Display results
            exp_names_str = ', '.join(experiment_names)
            st.markdown(
                f"<h2 style='text-align: center;'>{exp_names_str}</h2>",
                unsafe_allow_html=True,
            )
            st.header('Comparison Results')
            display_results(Path(st.session_state.output_dir))
        else:
            st.error('Please provide valid experiment names and folder paths.')
    else:
        st.info(
            'Please input experiment names and select experiment folders in the sidebar, then click "Run Comparisons".'
        )


def display_results(output_dir: Path) -> None:
    """Display the comparison results including graphs, radar chart, and rank comparisons.

    Args:
        output_dir (Path): The directory containing the output results.
    """
    # Organize results into tabs
    tabs = st.tabs(["Graphs", "Radar Chart", "Rank Comparisons"])

    with tabs[0]:
        st.subheader('Comparison Graphs')
        display_graphs(output_dir / GRAPH_COMPARISON_FOLDER_NAME)

    with tabs[1]:
        st.subheader('Radar Chart Visualization')
        df = display_table(output_dir, REPORT_TABLE_COMPARISON_FILE_NAME)
        if not df.empty:
            display_radar_chart(df)  # Call the function to display the radar chart and Best Agent

    with tabs[2]:
        st.subheader('Rank Comparisons')
        display_rank_comparisons(output_dir)


def display_radar_chart(df: pd.DataFrame) -> None:
    """Display an interactive radar chart with additional user controls inside the Radar Chart tab."""
    if df.empty or df.shape[1] < 2:
        st.warning("Insufficient data for radar chart.")
        return

    # Extract the 'Best Agent' column
    if 'Best Agent' in df.columns:
        best_agent = df[['experiment', 'Best Agent']].set_index('experiment')
    else:
        st.warning("'Best Agent' column not found in the data.")
        best_agent = pd.DataFrame()

    # Process the DataFrame for radar chart
    df.set_index('experiment', inplace=True)

    # Remove non-numeric columns (e.g., 'Best Agent')
    df_numeric = df.select_dtypes(include=[np.number])

    # Drop rows with NaN values
    df_numeric.dropna(inplace=True)

    # Add checklist and normalization toggle inside the "Radar Chart Visualization" tab
    st.subheader("Customize Radar Chart")
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        options=df_numeric.columns.tolist(),
        default=df_numeric.columns.tolist(),
        help="Choose which metrics to include in the radar chart."
    )
    normalize = st.checkbox("Normalize Metrics", value=False)

    # Filter by selected metrics
    df_numeric = df_numeric[selected_metrics]

    # Normalize if selected
    if normalize:
        df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

    # Transpose df so that experiments are columns
    df_transposed = df_numeric.transpose()

    # Get the indicators (metrics)
    metrics = df_transposed.index.tolist()

    # Calculate 'max' per indicator (metric)
    max_values = df_transposed.max(axis=1) * 1.1

    # Create indicators with individual max values
    indicators = [{"name": metric, "max": float(max_value)} for metric, max_value in zip(metrics, max_values)]

    # Prepare data series
    series_data = []
    for col in df_transposed.columns:
        values = df_transposed[col].tolist()
        series_data.append({"value": values, "name": col})

    # Radar chart option
    option = {
        "title": {"text": "Radar Chart"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": list(df_transposed.columns)},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "radar": {"indicator": indicators},
        "series": [{"name": "Experiments Comparison", "type": "radar", "data": series_data}],
    }

    # Customize legend text color for better visibility
    option["legend"]["textStyle"] = {"color": "white"}

    # Display the radar chart and Best Agent side by side
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.spinner('Loading Radar Chart...'):
            st_echarts(option, height="500px", key="radar_chart")
        st.success('Radar Chart loaded successfully!')

    with col2:
        if not best_agent.empty:
            st.markdown("### Best Agent")
            st.table(best_agent)
        else:
            st.write("No Best Agent data available.")




def display_graphs(output_dir: Path) -> None:
    """Display comparison graphs from the specified output directory using ECharts with dropdown selection.

    Args:
        output_dir (Path): The directory containing the ECharts JSON files.
    """
    # Collect ECharts JSON files
    json_files = list(output_dir.glob('**/*.json'))
    if not json_files:
        st.warning('No ECharts JSON files found.')
        return

    # Group JSON files by subdirectories (categories)
    graph_categories = {}
    for path in json_files:
        category = path.parent.name
        graph_categories.setdefault(category, []).append(path)

    # Create an expandable section for each category
    for category, files in graph_categories.items():
        # Derive a user-friendly title from the json title
        # Find the first json file in files
        json_file = next((file for file in files if file.suffix == '.json'), None)
        options_json = load_echarts_option(json_file)
        title = options_json.get('title', {}).get('text', category).replace('_', ' ').title()

        with st.expander(f"{title} ({len(files)} graphs)", expanded=False):
            # Create a dropdown to select which graph to display
            graph_names = [file.stem.replace('_', ' ').title() for file in files]
            selected_graph = st.selectbox(
                "Select a graph to display:",
                options=graph_names,
                key=f'select_{category}'
            )

            # Add details about the graph
            st.markdown(f"TESTTTTTT")

            # Find the corresponding JSON file
            selected_file = next((file for file, name in zip(files, graph_names) if name == selected_graph), None)
            if selected_file:
                # Load the ECharts option from JSON file
                try:
                    option = load_echarts_option(selected_file)
                    img_path = str(selected_file).replace('.json', '.png')
                    # Optionally remove or customize the title within the chart
                    option.pop('title', None)  # Remove internal title if not needed
                    # Render the chart using st_echarts
                    with st.spinner(f'Loading {selected_graph}...'):
                        st_echarts(options=option, height="600px")
                        st.download_button(
                            label="Download Graph",
                            data=open(img_path, "rb").read(),
                            file_name=os.path.basename(img_path),
                            mime="image/png",
                        )
                except Exception as e:
                    st.error(f"Error loading {selected_file.name}: {e}")
            else:
                st.warning(f"Selected graph {selected_graph} not found.")


def display_table(output_dir: Path, csv_filename: str) -> pd.DataFrame:
    """Load and return the specified CSV file as a DataFrame.

    Args:
        output_dir (Path): The directory containing the CSV file.
        csv_filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The DataFrame loaded from the CSV file.
    """
    csv_path = output_dir / csv_filename
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            # Add download button
            with open(csv_path, "rb") as csv_file:
                st.download_button(
                    label="Download CSV",
                    data=csv_file,
                    file_name=csv_filename,
                    mime="text/csv",
                )
            # Optionally display the table
            st.dataframe(df, use_container_width=True)
            return df  # Return the DataFrame for further use
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return pd.DataFrame()  # Return an empty DataFrame
    else:
        st.warning(f'CSV file {csv_filename} not found in {output_dir}.')
        return pd.DataFrame()


def display_rank_comparisons(output_dir: Path) -> None:
    """Display rank comparison analysis files from the output directory.

    Args:
        output_dir (Path): The directory containing rank comparison files.
    """
    # Find CSV files for rank comparisons
    csv_files = list(output_dir.glob('**/*_analysis.csv'))

    if not csv_files:
        st.warning('No rank comparison CSV files found.')
        return

    # First do rank_change_analysis and then transition_probabilities_analysis
    csv_files = sorted(csv_files, key=lambda x: 'rank_change_analysis' not in x.name)

    for csv_path in csv_files:
        file_name = csv_path.name
        title = file_name.replace('.csv', '').replace("_", " ").title()
        st.subheader(title)

        try:
            df = pd.read_csv(csv_path)
            # Uncomment the following line for debugging purposes
            # st.write(f"Processing file: {csv_path}")
            if 'rank_change_analysis' in file_name.lower():
                display_rank_change_analysis(df, title)
            elif 'transition_probabilities_analysis' in file_name.lower():
                display_transition_probabilities_analysis(df, title)
            else:
                st.warning(f"Unrecognized data format in {file_name}.")
        except Exception as e:
            st.error(f"Error reading file {file_name}: {e}")


def display_rank_change_analysis(df: pd.DataFrame, title: str) -> None:
    """Visualize rank change analysis data.

    Args:
        df (pd.DataFrame): The DataFrame containing rank change data.
        title (str): The title for the visualization.
    """
    required_columns = ['Round', 'Changes', 'Sum of Changes']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in {title}: {', '.join(missing_columns)}")
        return

    # Clean and preprocess the data
    df = df[df['Round'] != 'Total']  # Exclude the 'Total' row for plotting
    df['Changes'] = pd.to_numeric(df['Changes'], errors='coerce')
    df['Sum of Changes'] = pd.to_numeric(df['Sum of Changes'], errors='coerce')

    # Handle possible errors in 'Round' format
    def extract_round_number(round_str):
        try:
            return int(round_str.split(' to ')[0])
        except:
            return np.nan

    df['Round Number'] = df['Round'].apply(extract_round_number)
    df = df.dropna(subset=['Round Number'])
    df['Round Number'] = df['Round Number'].astype(int)
    df = df.sort_values('Round Number')

    # Check if there is data to plot
    if df.empty:
        st.warning(f"No valid data available for {title}.")
        return

    # Plotting using Plotly
    fig = go.Figure()

    # Add 'Changes' line
    fig.add_trace(go.Scatter(
        x=df['Round Number'],
        y=df['Changes'],
        mode='lines+markers',
        name='Changes'
    ))

    # Add 'Sum of Changes' line
    fig.add_trace(go.Scatter(
        x=df['Round Number'],
        y=df['Sum of Changes'],
        mode='lines+markers',
        name='Sum of Changes'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Round',
        yaxis_title='Count',
        legend_title='Metrics',
        xaxis=dict(tickmode='linear'),
        template='plotly_dark'  # Consistent dark theme
    )

    st.plotly_chart(fig, use_container_width=True)


def display_transition_probabilities_analysis(df: pd.DataFrame, title: str) -> None:
    """Visualize transition probabilities analysis data.

    Args:
        df (pd.DataFrame): The DataFrame containing transition probabilities data.
        title (str): The title for the visualization.
    """

    required_columns = ['Rank', 'Expected Value', 'Standard Deviation']
    prob_prefix = 'P(Rank '
    prob_cols = [col for col in df.columns if col.startswith(prob_prefix)]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in {title}: {', '.join(missing_columns)}")
        return

    if not prob_cols:
        st.warning("No transition probability columns found.")
        return

    # Convert 'Rank' to numeric, coerce errors
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')

    # Drop rows where 'Rank' is NaN
    df = df.dropna(subset=['Rank'])

    # Convert 'Rank' to int
    df['Rank'] = df['Rank'].astype(int)

    # Convert probability columns to numeric
    df[prob_cols] = df[prob_cols].apply(pd.to_numeric, errors='coerce')

    # Check for missing data
    if df[['Expected Value', 'Standard Deviation'] + prob_cols].isnull().values.any():
        st.warning("Some data points contain missing or invalid values and will be excluded from the visualizations.")
        df = df.dropna(subset=['Expected Value', 'Standard Deviation'] + prob_cols)

    # Check if there's data to plot
    if df.empty:
        st.warning(f"No valid data available for {title}.")
        return

    # Plot Expected Value with Standard Deviation
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df['Rank'],
        y=df['Expected Value'],
        error_y=dict(
            type='data',
            array=df['Standard Deviation'],
            visible=True
        ),
        name='Expected Value'
    ))

    fig1.update_layout(
        title=f"{title} - Expected Value with Standard Deviation",
        xaxis_title='Rank',
        yaxis_title='Expected Value',
        legend_title='Metrics',
        template='plotly_dark'  # Consistent dark theme
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Prepare data for heatmap
    heatmap_data = df.set_index('Rank')[prob_cols]
    heatmap_data.columns = [col.replace(prob_prefix, '').replace(')', '') for col in heatmap_data.columns]

    # Check if heatmap_data is not empty
    if heatmap_data.empty:
        st.warning(f"No data available to create a heatmap for {title}.")
    else:
        fig2 = px.imshow(
            heatmap_data,
            labels=dict(x='Next Rank', y='Current Rank', color='Probability'),
            x=heatmap_data.columns.astype(int),
            y=heatmap_data.index.astype(int),
            aspect='auto',
            color_continuous_scale='Viridis',
            title=f"{title} - Transition Probabilities Heatmap",
        )

        fig2.update_layout(
            template='plotly_dark'  # Consistent dark theme
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Prepare data for stacked bar chart
    df_melted = df.melt(id_vars=['Rank'], value_vars=prob_cols,
                        var_name='Next Rank', value_name='Probability')
    df_melted['Next Rank'] = df_melted['Next Rank'].str.extract(r'P\(Rank (\d+)\)').astype(int)

    # Check if df_melted has valid data
    if df_melted.empty:
        st.warning(f"No data available to create a stacked bar chart for {title}.")
        return

    fig3 = px.bar(
        df_melted,
        x='Rank',
        y='Probability',
        color='Next Rank',
        title=f"{title} - Transition Probabilities Stacked Bar Chart",
        labels={'Rank': 'Current Rank', 'Probability': 'Transition Probability'},
        color_continuous_scale='Viridis'
    )

    fig3.update_layout(
        template='plotly_dark'  # Consistent dark theme
    )

    st.plotly_chart(fig3, use_container_width=True)


if __name__ == '__main__':
    main()
