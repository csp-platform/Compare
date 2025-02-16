import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "output")

RANKS_COMPARISON_FOLDER_NAME = "ranks_comparison"
GRAPH_COMPARISON_FOLDER_NAME = "graphs_comparison"

RANK_CHANGE_ANALYSIS_FILE_NAME = "rank_change_analysis.csv"
TRANSITION_PROBABILITIES_ANALYSIS_FILE_NAME = "transition_probabilities_analysis.csv"
REPORT_TABLE_COMPARISON_FILE_NAME = "final_report_table.csv"

COMPETITION_HISTORY_PILOT_FILE_NAME = "competition_history_pivot.csv"
SETTING_PLOT_FIRST_SECOND_DISTANCE_OVER_TIME = "plot_first_second_similarity_over_time"
SETTING_PLOT_RANK_DIAMETER_AND_AVERAGE_OVER_TIME = "plot_rank_diameter_and_average_over_time"

SETTING_WINNER_SIMILARITY_OVER_TIME = 'winner_similarity_over_time'
SETTING_AVERAGE_UNIQUE_DOCUMENTS_OVER_TIME = 'average_unique_documents_over_time'
SETTING_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS = 'average_and_diameter_of_player_documents'
SETTING_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS = 'average_of_player_documents_consecutive_rounds'
SETTING_PLOT_DIAMETER_AND_AVERAGE_OVER_TIME = 'plot_diameter_and_average_over_time'

REPORT_TABLE_INPUT_CSV_FILE_NAME = "report_table.csv"
AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_CSV_FILE_NAME = "average_and_diameter_of_player_documents.csv"
AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_CSV_FILE_NAME = "average_of_player_documents_consecutive_rounds.csv"
RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_FILE_NAME = "rank_diameter_and_average_last_round.csv"

NUM_TO_STR = {1: "first", 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth', 15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', 19: 'nineteenth', 20: 'twentieth', 21: 'twenty-first', 22: 'twenty-second', 23: 'twenty-third', 24: 'twenty-fourth', 25: 'twenty-fifth', 26: 'twenty-sixth', 27: 'twenty-seventh', 28: 'twenty-eighth', 29: 'twenty-ninth', 30: 'thirtieth', 31: 'thirty-first', 32: 'thirty-second', 33: 'thirty-third', 34: 'thirty-fourth', 35: 'thirty-fifth', 36: 'thirty-sixth', 37: 'thirty-seventh', 38: 'thirty-eighth', 39: 'thirty-ninth', 40: 'fortieth', 41: 'forty-first', 42: 'forty-second', 43: 'forty-third', 44: 'forty-fourth', 45: 'forty-fifth', 46: 'forty-sixth', 47: 'forty-seventh', 48: 'forty-eighth', 49: 'forty-ninth', 50: 'fiftieth', 51: 'fifty-first', 52: 'fifty-second', 53: 'fifty-third', 54: 'fifty-fourth', 55: 'fifty-fifth', 56: 'fifty-sixth', 57: 'fifty-seventh', 58: 'fifty-eighth', 59: 'fifty-ninth', 60: 'sixtieth', 61: 'sixty-first', 62: 'sixty-second', 63: 'sixty-third', 64: 'sixty-fourth', 65: 'sixty-fifth', 66: 'sixty-sixth', 67: 'sixty-seventh', 68: 'sixty-eighth', 69: 'sixty-ninth', 70: 'seventieth', 71: 'seventy-first', 72: 'seventy-second', 73: 'seventy-third', 74: 'seventy-fourth', 75: 'seventy-fifth', 76: 'seventy-sixth', 77: 'seventy-seventh', 78: 'seventy-eighth', 79: 'seventy-ninth', 80: 'eightieth', 81: 'eighty-first', 82: 'eighty-second', 83: 'eighty-third', 84: 'eighty-fourth', 85: 'eighty-fifth', 86: 'eighty-sixth', 87: 'eighty-seventh', 88: 'eighty-eighth', 89: 'eighty-ninth', 90: 'ninetieth', 91: 'ninety-first', 92: 'ninety-second', 93: 'ninety-third', 94: 'ninety-fourth', 95: 'ninety-fifth', 96: 'ninety-sixth', 97: 'ninety-seventh', 98: 'ninety-eighth', 99: 'ninety-ninth', 100: 'one hundredth'}
NUM_TO_REPRESENTATION ={0: "tf_idf_jaccard_similarity", 1: "tf_idf_similarity", 2: "e5_embeddings_similarity", 3: "bert_embeddings_similarity"}

EMBEDDINGS_GRAPHS_FOLDER = "embeddings_graphs"

USER_QUERY_COLUMN = "user_query"
SPERATION_LINE = "-" * 50
MEAN_METRIC = "mean"
MIN_METRIC = "min"

FIGURE_SIZE = (6.5, 6.5)
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
GRAPH_LEFT_MARGIN = 0.18
CONFIDENCE_INTERVAL = 1.96

ROUND_X_LABEL = "Round"
ROUNDS_INCLUDED_X_LABEL = "Round"

FIRST_SECOND_SIMILARITY_OVER_TIME_FILE_NAME = "first_second_similarity_over_time.png"
RANK_DIAMETER_AND_AVERAGE_OVER_TIME_FILE_NAME = "rank_diameter_and_average_over_time.png"
CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME_FILE_NAME = "winner_similarity_over_time.png"
AVERAGE_UNIQUE_DOCUMENTS_FILE_NAME = "average_unique_documents_over_time.png"
AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_FILE_NAME = "player_documents.png"
AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUND_FILE_NAME = "average_similarity_consecutive_rounds.png"
DIAMETER_AND_AVERAGE_OVER_TIME = "over_time.png"

Y_LABEL_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME = "Average similarity between consecutive winners"
TITLE_CONSECUTIVE_WINNER_SIMILARITY_OVER_TIME = "Average similarity between \nconsecutive winners vs round"

Y_LABEL_AVERAGE_UNIQUE_DOCUMENTS = "Average number of unique documents"
TITLE_LABEL_AVERAGE_UNIQUE_DOCUMENTS = "Average number of unique documents vs round"

Y_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS = "Average similarity of player documents"
TITLE_LABEL_MEAN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS = "Average similarity of player documents vs round"

Y_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS = "Diameter similarity of player documents"
TITLE_LABEL_MIN_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS = "Diameter similarity of player documents vs round"

Y_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS = "Average similarity of \nplayer documents between consecutive rounds"
TITLE_LABEL_AVERAGE_SIMILARITY_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS = "Average similarity of \nplayer documents vs consecutive rounds"

Y_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME = "Average group similarity"
TITLE_LABEL_MEAN_DIAMETER_AND_AVERAGE_OVER_TIME = "Average group similarity vs round"

Y_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME = "Diameter group similarity"
TITLE_LABEL_MIN_DIAMETER_AND_AVERAGE_OVER_TIME = "Diameter group similarity vs round"

FINAL_TABLE_EXPERIMENT_COLUMN = "experiment"
FINAL_TABLE_BEST_AGENT_COLUMN = "Best Agent"
FINAL_TABLE_WINNING_HOMOGENEITY_COLUMN = "Winning Homogeneity"
FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MEAN_COLUMN = "average_and_diameter_of_player_documents_mean"
FINAL_TABLE_AVERAGE_AND_DIAMETER_OF_PLAYER_DOCUMENTS_MIN_COLUMN = "average_and_diameter_of_player_documents_min"
FINAL_TABLE_AVERAGE_OF_PLAYER_DOCUMENTS_CONSECUTIVE_ROUNDS_MEAN_COLUMN = "average_of_player_documents_consecutive_rounds_mean"
FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MEAN_COLUMN = "rank_diameter_and_average_last_round_mean"
FINAL_TABLE_RANK_DIAMETER_AND_AVERAGE_LAST_ROUND_MIN_COLUMN = "rank_diameter_and_average_last_round_min"