'''
 File to hold hardcoded constants. Examples might include:
 Paths
 Error messages
 URL's
 Max/Min values etc..
'''
# Constants
K_NEIGHBOURS = 3
LEARNING_RATE = 0.01
EPOCHS = 1000
SUM_CSV_SEP = ';'
HOUSE_CSV_SEP = ','

# Algorithms
LINEAR_REG = 'LINEAR_REGRESSION'
KNN = 'KNN'

# Split Types
SEVENTY_THIRTY = 'SEVENTY_THIRTY'
TEN_FOLD_CROSS = 'TEN_FOLD_CROSS'

# Datasets
SUM_WITHOUT_NOISE = 'SUM_WITHOUT_NOISE'
SUM_WITH_NOISE = 'SUM_WITH_NOISE'
HOUSE_PRICES = 'HOUSE_PRICES'

# All paths are relative to task directory
PATHS = {
    SUM_WITH_NOISE: "../../data/The SUM dataset, without noise.csv",
    SUM_WITH_NOISE: "../../data/The SUM dataset, with noise.csv",
    HOUSE_PRICES: "../../data/kc_house_data.csv"
}
