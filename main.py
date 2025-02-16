from experiment_comparison import ExperimentComparison


def main():
    # Define the experiment folders to compare
    experiment_folders = {"exp1": "data/5e2587bfad51339c606e96652bccaf4d", "exp2": "data/78ec5035a4eaa7596581a158a65396f1"}

    # Initialize the Compare class with the experiment folders
    compare = ExperimentComparison(experiment_folders)
    compare.execute_comparisons()

if __name__ == "__main__":
    main()
