from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = PROJECT_ROOT / 'data/raw'
CLEANED_DATA_DIR = PROJECT_ROOT / 'data/cleaned'
FEATURES_DIR = PROJECT_ROOT / 'data/features'
MODELS_DIR = PROJECT_ROOT / 'data/models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Create directories if they do not exist
for d in [RAW_DATA_DIR, CLEANED_DATA_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Raw Data Filenames
LOYALTY_HISTORY_FILE = 'Customer Loyalty History.csv'
FLIGHT_ACTIVITY_FILE = 'Customer Flight Activity.csv'
CALENDAR_FILE = 'Calendar.csv'
DATA_DICT_FILE = 'Airline Loyalty Data Dictionary.csv'

# Primary Key Column
PK = 'Loyalty Number'

# Dataset Column Mappings
LOYALTY_COLS = {
    'pk': 'Loyalty Number',
    'country': 'Country',
    'province': 'Province',
    'city': 'City',
    'postal_code': 'Postal Code',
    'gender': 'Gender',
    'education': 'Education',
    'salary': 'Salary',
    'marital_status': 'Marital Status',
    'loyalty_card': 'Loyalty Card',
    'clv': 'CLV',
    'enrollment_type': 'Enrollment Type',
    'enrollment_year': 'Enrollment Year',
    'enrollment_month': 'Enrollment Month',
    'cancellation_year': 'Cancellation Year',
    'cancellation_month': 'Cancellation Month'
}

FLIGHT_COLS = {
    'pk': 'Loyalty Number',
    'year': 'Year',
    'month': 'Month',
    'total_flights': 'Total Flights',
    'distance': 'Distance',
    'points_accumulated': 'Points Accumulated',
    'points_redeemed': 'Points Redeemed',
    'dollar_cost_points_redeemed': 'Dollar Cost Points Redeemed'
}

# Categorical Orderings
EDUCATION_ORDER = ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor']
LOYALTY_CARD_ORDER = ['Star', 'Nova', 'Aurora']

# Model Hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CHURN_WINDOW_MONTHS = 6

# Feature Engineering Settings
ROLLING_WINDOWS = [3, 6, 12]

# Outlier Clipping Bounds
IQR_MULTIPLIER = 1.5
PERCENTILE_LOWER = 0.01
PERCENTILE_UPPER = 0.99

# Future Value Score Component Weights
FVS_WEIGHTS = {
    'clv_normalized': 0.2,
    'flight_frequency_12m': 0.2,
    'loyalty_tier': 0.15,
    'engagement_trend': 0.2,
    'redemption_behavior': 0.1,
    'tenure': 0.15
}

# Segment ID to Label Mapping
SEGMENT_NAMES = {
    0: 'Champions',
    1: 'VIP At Risk',
    2: 'Loyal Travelers',
    3: 'Growth Potential',
    4: 'Dormant Members'
}

# Streamlit Port
STREAMLIT_PORT = 8501