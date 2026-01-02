# ⚽ Football Analysis Project

A comprehensive Python-based analysis system for football statistics, featuring web scraping from FBref.com, exploratory data analysis, machine learning predictions, and interactive visualizations.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Modules Overview](#modules-overview)
- [Examples](#examples)
- [Contributing](#contributing)

## ✨ Features

### 🌐 Web Scraping
- **FBref Data Scraper**: Automated scraping of player statistics, team data, and match results
- **Big 5 European Leagues**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- **Player Profiles**: Detailed career statistics and performance data
- **Respectful Scraping**: Built-in rate limiting to respect website resources

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive statistical summaries and visualizations
- **Distribution Analysis**: Identify patterns and outliers in player performance
- **Correlation Analysis**: Discover relationships between different metrics
- **League Comparisons**: Compare statistics across different competitions

### 🤖 Machine Learning
- **Multiple Models**: Linear Regression, Random Forest, Gradient Boosting
- **Performance Prediction**: Predict goals, assists, and other metrics
- **Feature Engineering**: Advanced feature creation for better predictions
- **Model Evaluation**: Cross-validation and comprehensive metrics

### 📈 Visualization
- **Interactive Plots**: Plotly-based interactive visualizations
- **Radar Charts**: Player performance profiles
- **Heatmaps**: Performance patterns across different dimensions
- **Dashboards**: Comprehensive analytical dashboards

## 📁 Project Structure

```
Football-analysis/
├── main.py                    # Main application entry point
├── scraper.py                 # FBref web scraper
├── eda_script.py             # Exploratory data analysis
├── model.py                  # Machine learning models
├── data_processor.py         # Data cleaning and feature engineering
├── visualization.py          # Advanced visualizations
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
│
├── data/                    # Data directory (created automatically)
│   ├── raw/                # Raw scraped data
│   └── processed/          # Cleaned and processed data
│
├── outputs/                # Analysis outputs (created automatically)
│   ├── figures/           # Static visualizations
│   └── reports/           # Analysis reports
│
├── visualizations/        # Interactive visualizations (created automatically)
│   └── *.html            # Plotly interactive charts
│
└── models/               # Saved ML models (created automatically)
    └── *.pkl            # Pickled model files
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/damanjeet-singh-7z/Football-analysis.git
cd Football-analysis
```

2. **Create a virtual environment** (recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Interactive Mode

Run the main application with an interactive menu:
```bash
python main.py
```

The interactive menu provides options to:
1. Scrape latest data from FBref
2. Load existing data
3. Run exploratory data analysis
4. Train prediction models
5. Make predictions
6. Search for players

### Individual Modules

#### 1. Web Scraping

```python
from scraper import FBrefScraper

# Initialize scraper (3.5 second delay between requests)
scraper = FBrefScraper(delay=3.5)

# Scrape Big 5 leagues
data = scraper.scrape_big5_leagues(season='2023-2024')
data.to_csv('data/big5_2023_24.csv', index=False)

# Search for a specific player
players = scraper.scrape_player_search("Cristiano Ronaldo")
for player in players:
    print(f"{player['name']} - {player['url']}")

# Get player profile
if players:
    profile = scraper.scrape_player_profile(players[0]['url'])
    print(profile)
```

#### 2. Data Processing

```python
from data_processor import DataProcessor
import pandas as pd

# Load data
data = pd.read_csv('Big5_2020_21_Cleaned.csv')

# Initialize processor
processor = DataProcessor()

# Clean data
clean_data = processor.clean_data(data)

# Handle missing values
complete_data = processor.handle_missing_values(clean_data, strategy='smart')

# Create engineered features
featured_data = processor.create_features(complete_data)

# Remove outliers
final_data = processor.remove_outliers(featured_data)
```

#### 3. Exploratory Data Analysis

```python
from eda_script import FootballEDA
import pandas as pd

# Load data
data = pd.read_csv('Big5_2020_21_Cleaned.csv')

# Initialize EDA
eda = FootballEDA(data, output_dir='outputs')

# Generate comprehensive analysis
eda.generate_full_report()

# Or run individual analyses
eda.generate_summary_statistics()
eda.analyze_distributions()
eda.correlation_analysis()
eda.top_performers_analysis(top_n=20)
eda.league_comparison()
```

#### 4. Machine Learning

```python
from model import FootballPredictor
import pandas as pd

# Load data
data = pd.read_csv('Big5_2020_21_Cleaned.csv')

# Initialize predictor
predictor = FootballPredictor()

# Prepare data and train
predictor.prepare_data(data, target_col='Goals')
results = predictor.train()

# Evaluate models
predictor.evaluate()

# Get feature importance
predictor.feature_importance(top_n=10)

# Save best model
predictor.save_model('models/goal_predictor.pkl')

# Make predictions on new data
new_player_data = pd.DataFrame({
    'Matches': [30],
    'Minutes': [2500],
    'Assists': [5],
    'Age': [25]
})
prediction = predictor.predict(new_player_data)
print(f"Predicted goals: {prediction[0]:.2f}")
```

#### 5. Visualization

```python
from visualization import FootballVisualizer
import pandas as pd

# Load data
data = pd.read_csv('Big5_2020_21_Cleaned.csv')

# Initialize visualizer
viz = FootballVisualizer(output_dir='visualizations')

# Create player radar chart
player = data.iloc[0]
categories = ['Goals', 'Assists', 'Matches', 'Minutes']
viz.plot_player_radar(player, categories, player_name='Player Name')

# Create scatter matrix
numeric_cols = ['Goals', 'Assists', 'Matches', 'Age', 'Minutes']
viz.plot_scatter_matrix(data, numeric_cols, color_by='League')

# Compare teams
metrics = ['Goals', 'Assists', 'Minutes']
viz.plot_team_comparison(data, team_col='Team', metrics=metrics)
```

## 🌐 Data Sources

### FBref.com
- **URL**: https://fbref.com
- **Coverage**: Player statistics, team data, match results
- **Leagues**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, and more
- **Rate Limiting**: Please respect the 3+ second delay between requests

### Existing Dataset
- `Big5_2020_21_Cleaned.csv`: Pre-cleaned dataset of Big 5 European leagues for 2020-21 season

## 📦 Modules Overview

### `scraper.py`
Web scraping functionality for FBref.com with the following methods:
- `scrape_league_stats()`: Get player statistics for a league/season
- `scrape_team_stats()`: Get team-level statistics
- `scrape_player_search()`: Search for players
- `scrape_player_profile()`: Get detailed player information
- `scrape_match_results()`: Get match results and fixtures
- `scrape_big5_leagues()`: Scrape all Big 5 European leagues

### `data_processor.py`
Data cleaning and feature engineering:
- `clean_data()`: Remove duplicates, standardize formats
- `handle_missing_values()`: Intelligent missing value imputation
- `encode_categorical()`: Encode categorical variables
- `create_features()`: Engineer new features (goals per match, conversion rate, etc.)
- `remove_outliers()`: Outlier detection and removal
- `normalize_features()`: Feature scaling

### `eda_script.py`
Exploratory data analysis:
- `generate_summary_statistics()`: Comprehensive data summary
- `analyze_distributions()`: Distribution plots for key metrics
- `correlation_analysis()`: Correlation heatmaps
- `top_performers_analysis()`: Identify and visualize top players
- `league_comparison()`: Compare statistics across leagues

### `model.py`
Machine learning models:
- Multiple regression models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
- Automated model training and evaluation
- Feature importance analysis
- Model persistence (save/load)
- Prediction interface

### `visualization.py`
Advanced visualizations:
- `plot_player_radar()`: Radar charts for player profiles
- `plot_scatter_matrix()`: Multi-dimensional scatter plots
- `plot_performance_timeline()`: Time series visualizations
- `plot_heatmap()`: Performance heatmaps
- `plot_team_comparison()`: Team comparison charts
- `create_dashboard()`: Comprehensive analytical dashboards

### `main.py`
Main application:
- Interactive menu system
- Workflow orchestration
- Integration of all modules

## 📝 Examples

### Example 1: Complete Analysis Pipeline

```python
from scraper import FBrefScraper
from data_processor import DataProcessor
from eda_script import FootballEDA
from model import FootballPredictor

# 1. Scrape latest data
scraper = FBrefScraper()
data = scraper.scrape_big5_leagues(season='2023-2024')
data.to_csv('data/latest_data.csv', index=False)

# 2. Process data
processor = DataProcessor()
clean_data = processor.clean_data(data)
processed_data = processor.create_features(clean_data)

# 3. Analyze data
eda = FootballEDA(processed_data)
eda.generate_full_report()

# 4. Train prediction model
predictor = FootballPredictor()
predictor.prepare_data(processed_data, target_col='Goals')
predictor.train()
predictor.save_model('models/latest_model.pkl')
```

### Example 2: Player Comparison

```python
import pandas as pd
from visualization import FootballVisualizer

# Load data
data = pd.read_csv('Big5_2020_21_Cleaned.csv')

# Select players
player1 = data[data['Player'] == 'Lionel Messi'].iloc[0]
player2 = data[data['Player'] == 'Cristiano Ronaldo'].iloc[0]

# Create radar charts
viz = FootballVisualizer()
categories = ['Goals', 'Assists', 'Shots', 'Key Passes', 'Dribbles']

viz.plot_player_radar(player1, categories, 'Lionel Messi')
viz.plot_player_radar(player2, categories, 'Cristiano Ronaldo')
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<img width="1515" height="985" alt="Screenshot 2025-12-22 234553" src="https://github.com/user-attachments/assets/95a9bdfb-62ae-4586-93dd-47822edc22ec" />


<img width="1723" height="945" alt="Screenshot 2025-12-22 234619" src="https://github.com/user-attachments/assets/e7623aa4-37f6-4de5-bc62-9c5d6784323c" />

<img width="1749" height="1003" alt="Screenshot 2025-12-22 234643" src="https://github.com/user-attachments/assets/0fab8142-7817-4bcb-b0de-9f77e48262ad" />

<img width="1796" height="987" alt="Screenshot 2025-12-22 234656" src="https://github.com/user-attachments/assets/b02d12c0-bf21-45e5-a4fe-7ac9904b4d5b" />

<img width="1764" height="399" alt="Screenshot 2025-12-22 234714" src="https://github.com/user-attachments/assets/ec1f1187-71bb-46cf-b379-0359385f726f" />


## ⚖️ License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- FBref.com for providing comprehensive football statistics
- The open-source community for the amazing libraries used in this project
- All contributors and users of this project

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: When using the web scraper, please be respectful of FBref's resources. The scraper includes a default 3.5-second delay between requests. Do not reduce this delay or make excessive requests.
