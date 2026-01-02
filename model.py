"""
Machine Learning Models for Football Analysis
Prediction models for player performance, market value, and outcomes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballPredictor:
    """Machine learning models for football predictions"""
    
    def __init__(self):
        """Initialize predictor"""
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, data: pd.DataFrame, target_col: str, 
                     feature_cols: list = None, test_size: float = 0.2):
        """
        Prepare data for training
        
        Args:
            data: Input DataFrame
            target_col: Target variable column name
            feature_cols: List of feature columns (if None, uses all numeric except target)
            test_size: Proportion of test set
        """
        logger.info("Preparing data for modeling...")
        
        # Drop rows with missing target
        data_clean = data.dropna(subset=[target_col]).copy()
        
        # Select features
        if feature_cols is None:
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        for col in feature_cols:
            if data_clean[col].isnull().sum() / len(data_clean) > missing_threshold:
                feature_cols.remove(col)
                logger.info(f"Removed {col} due to missing values")
        
        # Prepare X and y
        X = data_clean[feature_cols].copy()
        y = data_clean[target_col].copy()
        
        # Handle remaining missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        self.target_name = target_col
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
    
    def train(self, data: pd.DataFrame = None, target_col: str = None):
        """
        Train all models and select the best one
        
        Args:
            data: Optional data to prepare before training
            target_col: Target column name
        """
        if data is not None and target_col is not None:
            self.prepare_data(data, target_col)
        
        if self.X_train is None:
            raise ValueError("No training data available. Call prepare_data() first.")
        
        logger.info("Training models...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                if name in ['linear', 'ridge', 'lasso']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Select best model based on R²
        best = max(results.items(), key=lambda x: x[1]['r2'])
        self.best_model_name = best[0]
        self.best_model = best[1]['model']
        
        logger.info(f"\nBest model: {self.best_model_name}")
        logger.info(f"Best R²: {best[1]['r2']:.4f}")
        
        self.results = results
        
        return results
    
    def tune_best_model(self):
        """Hyperparameter tuning for the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        logger.info(f"Tuning {self.best_model_name}...")
        
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.models['random_forest'],
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        elif self.best_model_name == 'gradient_boost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.models['gradient_boost'],
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def predict(self, X_new: pd.DataFrame):
        """
        Make predictions on new data
        
        Args:
            X_new: DataFrame with same features as training data
        
        Returns:
            Array of predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        # Ensure same features
        X_new = X_new[self.feature_names]
        
        # Handle missing values
        X_new = X_new.fillna(self.X_train.median())
        
        # Scale if needed
        if self.best_model_name in ['linear', 'ridge', 'lasso']:
            X_new_scaled = self.scaler.transform(X_new)
            predictions = self.best_model.predict(X_new_scaled)
        else:
            predictions = self.best_model.predict(X_new)
        
        return predictions
    
    def feature_importance(self, top_n: int = 10):
        """
        Get feature importance for tree-based models
        
        Args:
            top_n: Number of top features to display
        """
        if self.best_model_name not in ['random_forest', 'gradient_boost']:
            logger.warning("Feature importance only available for tree-based models")
            return None
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Feature Importances:")
        for i, idx in enumerate(indices, 1):
            print(f"{i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        return pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices]
        })
    
    def evaluate(self):
        """Evaluate all trained models"""
        if not hasattr(self, 'results'):
            raise ValueError("No models trained yet. Call train() first.")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        comparison = pd.DataFrame({
            name: {
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R²': result['r2']
            }
            for name, result in self.results.items()
        }).T
        
        print(comparison.sort_values('R²', ascending=False))
        print(f"\nBest Model: {self.best_model_name}")
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_name': self.best_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.best_model_name = model_data['model_name']
        
        logger.info(f"Model loaded from {filepath}")


class PlayerPerformancePredictor(FootballPredictor):
    """Specialized predictor for player performance metrics"""
    
    def __init__(self):
        super().__init__()
    
    def predict_goals(self, data: pd.DataFrame):
        """Predict goals scored"""
        self.prepare_data(data, target_col='Goals', 
                         feature_cols=['Matches', 'Minutes', 'Assists', 'Age'])
        self.train()
        return self.best_model
    
    def predict_assists(self, data: pd.DataFrame):
        """Predict assists"""
        self.prepare_data(data, target_col='Assists',
                         feature_cols=['Matches', 'Minutes', 'Goals', 'Age'])
        self.train()
        return self.best_model


# Example usage
if __name__ == "__main__":
    # Load data
    try:
        data = pd.read_csv('Big5_2020_21_Cleaned.csv')
        
        # Initialize predictor
        predictor = FootballPredictor()
        
        # Train models to predict goals
        predictor.prepare_data(data, target_col='Goals')
        results = predictor.train()
        
        # Evaluate
        predictor.evaluate()
        
        # Feature importance
        predictor.feature_importance()
        
        # Save model
        predictor.save_model('goal_predictor.pkl')
        
    except FileNotFoundError:
        print("Data file not found.")
