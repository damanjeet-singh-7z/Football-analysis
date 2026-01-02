"""
Football Analysis - Main Application
Entry point for the football statistics analysis project
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Import custom modules
try:
    from scraper import FBrefScraper
    from eda_script import FootballEDA
    from model import FootballPredictor
    from data_processor import DataProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('football_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FootballAnalysis:
    """Main application class for football analysis"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the application
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.scraper = FBrefScraper(delay=3.5)
        self.processor = DataProcessor()
        self.eda = None
        self.predictor = None
        
        logger.info("Football Analysis application initialized")
    
    def scrape_latest_data(self, season: str = '2023-2024'):
        """
        Scrape latest data from FBref
        
        Args:
            season: Season to scrape (e.g., '2023-2024')
        """
        logger.info(f"Starting data scraping for season {season}")
        
        # Scrape Big 5 leagues
        output_file = self.data_dir / f'big5_{season.replace("-", "_")}.csv'
        
        if output_file.exists():
            logger.info(f"Data file already exists: {output_file}")
            response = input("Do you want to re-scrape? (yes/no): ")
            if response.lower() != 'yes':
                return pd.read_csv(output_file)
        
        logger.info("Scraping Big 5 European leagues...")
        data = self.scraper.scrape_big5_leagues(season)
        
        if not data.empty:
            data.to_csv(output_file, index=False)
            logger.info(f"Saved {len(data)} records to {output_file}")
            return data
        else:
            logger.error("Failed to scrape data")
            return pd.DataFrame()
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def run_eda(self, data: pd.DataFrame):
        """
        Run exploratory data analysis
        
        Args:
            data: DataFrame to analyze
        """
        logger.info("Starting exploratory data analysis...")
        
        self.eda = FootballEDA(data)
        
        # Generate comprehensive analysis
        self.eda.generate_summary_statistics()
        self.eda.analyze_distributions()
        self.eda.correlation_analysis()
        self.eda.top_performers_analysis()
        self.eda.league_comparison()
        
        logger.info("EDA completed")
    
    def train_models(self, data: pd.DataFrame):
        """
        Train machine learning models
        
        Args:
            data: DataFrame with training data
        """
        logger.info("Starting model training...")
        
        # Process data
        X, y = self.processor.prepare_training_data(data)
        
        # Initialize and train predictor
        self.predictor = FootballPredictor()
        self.predictor.train(X, y)
        
        # Evaluate models
        self.predictor.evaluate()
        
        # Save best model
        model_path = self.data_dir / 'best_model.pkl'
        self.predictor.save_model(str(model_path))
        
        logger.info(f"Model training completed. Best model saved to {model_path}")
    
    def make_predictions(self, player_data: dict):
        """
        Make predictions for a player
        
        Args:
            player_data: Dictionary with player features
        """
        if self.predictor is None:
            logger.error("No model loaded. Train or load a model first.")
            return None
        
        prediction = self.predictor.predict(player_data)
        logger.info(f"Prediction: {prediction}")
        return prediction
    
    def interactive_menu(self):
        """Display interactive menu for user"""
        while True:
            print("\n" + "="*50)
            print("Football Analysis System")
            print("="*50)
            print("1. Scrape latest data from FBref")
            print("2. Load existing data")
            print("3. Run exploratory data analysis")
            print("4. Train prediction models")
            print("5. Make predictions")
            print("6. Search for player")
            print("7. Exit")
            print("="*50)
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                season = input("Enter season (e.g., 2023-2024): ").strip()
                self.scrape_latest_data(season)
            
            elif choice == '2':
                filepath = input("Enter CSV file path: ").strip()
                data = self.load_data(filepath)
                if not data.empty:
                    self.current_data = data
                    print(f"\nLoaded data shape: {data.shape}")
                    print(f"Columns: {list(data.columns)[:10]}...")
            
            elif choice == '3':
                if hasattr(self, 'current_data'):
                    self.run_eda(self.current_data)
                else:
                    print("\nPlease load data first (option 1 or 2)")
            
            elif choice == '4':
                if hasattr(self, 'current_data'):
                    self.train_models(self.current_data)
                else:
                    print("\nPlease load data first (option 1 or 2)")
            
            elif choice == '5':
                if self.predictor is None:
                    print("\nPlease train a model first (option 4)")
                else:
                    print("\nEnter player statistics:")
                    # Get player data from user
                    # This would be expanded based on your model features
                    print("(Feature implementation needed based on your model)")
            
            elif choice == '6':
                player_name = input("Enter player name: ").strip()
                results = self.scraper.scrape_player_search(player_name)
                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, player in enumerate(results[:5], 1):
                        print(f"{i}. {player['name']}")
                        print(f"   URL: {player['url']}")
                else:
                    print("\nNo results found")
            
            elif choice == '7':
                print("\nExiting application. Goodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")


def main():
    """Main entry point"""
    print("="*60)
    print(" Football Analysis System v1.0")
    print(" Comprehensive Football Statistics Analysis & Prediction")
    print("="*60)
    
    # Initialize application
    app = FootballAnalysis(data_dir='data')
    
    # Check if we have existing data
    existing_data = list(Path('data').glob('*.csv'))
    if existing_data:
        print(f"\nFound {len(existing_data)} existing data file(s)")
        for f in existing_data:
            print(f"  - {f.name}")
    
    # Run interactive menu
    app.interactive_menu()


if __name__ == "__main__":
    main()
