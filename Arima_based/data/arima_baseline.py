"""
ARIMA Baseline for RippleNet-TFT
Creates ARIMA models for each commodity and generates forecasts/residuals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMABaseline:
    """ARIMA baseline model for energy commodities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.forecasts = {}
        self.residuals = {}
        self.arima_params = config['model']['arima']
        
    def check_stationarity(self, series: pd.Series, title: str = "Series") -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        logger.info(f"Checking stationarity for {title}")
        
        result = adfuller(series.dropna())
        
        logger.info(f"ADF Statistic: {result[0]:.4f}")
        logger.info(f"p-value: {result[1]:.4f}")
        logger.info(f"Critical Values:")
        for key, value in result[4].items():
            logger.info(f"\t{key}: {value:.4f}")
        
        # Series is stationary if p-value < 0.05
        is_stationary = result[1] < 0.05
        
        if is_stationary:
            logger.info(f"{title} is stationary")
        else:
            logger.info(f"{title} is not stationary")
        
        return is_stationary
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """Make time series stationary by differencing"""
        logger.info("Making series stationary")
        
        diff_count = 0
        current_series = series.copy()
        
        while diff_count < max_diff:
            if self.check_stationarity(current_series, f"Series (diff={diff_count})"):
                break
            
            current_series = current_series.diff()
            diff_count += 1
        
        if diff_count == max_diff and not self.check_stationarity(current_series):
            logger.warning(f"Series still not stationary after {max_diff} differences")
        
        return current_series, diff_count
    
    def find_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using AIC"""
        logger.info("Finding optimal ARIMA order")
        
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        # Grid search over ARIMA parameters
        for p in range(self.arima_params['max_p'] + 1):
            for d in range(self.arima_params['max_d'] + 1):
                for q in range(self.arima_params['max_q'] + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def find_seasonal_order(self, series: pd.Series) -> Tuple[int, int, int, int]:
        """Find optimal seasonal ARIMA order"""
        logger.info("Finding optimal seasonal ARIMA order")
        
        best_aic = np.inf
        best_order = (0, 0, 0, 0)
        
        # Grid search over seasonal parameters
        for P in range(self.arima_params['max_P'] + 1):
            for D in range(self.arima_params['max_D'] + 1):
                for Q in range(self.arima_params['max_Q'] + 1):
                    try:
                        model = ARIMA(
                            series, 
                            order=(1, 1, 1),  # Use simple order for seasonal search
                            seasonal_order=(P, D, Q, self.arima_params['m'])
                        )
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (1, 1, 1, P, D, Q, self.arima_params['m'])
                            
                    except:
                        continue
        
        logger.info(f"Best seasonal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit_arima_model(self, series: pd.Series, symbol: str) -> Dict:
        """Fit ARIMA model for a commodity"""
        logger.info(f"Fitting ARIMA model for {symbol}")
        
        # Clean data
        series = series.dropna()
        
        if len(series) < 50:
            logger.warning(f"Insufficient data for {symbol} ({len(series)} points)")
            return None
        
        # Check stationarity
        if not self.check_stationarity(series, symbol):
            logger.info(f"Making {symbol} stationary")
            stationary_series, d = self.make_stationary(series)
        else:
            stationary_series = series
            d = 0
        
        # Find optimal order
        if self.arima_params['seasonal']:
            order = self.find_seasonal_order(stationary_series)
        else:
            order = self.find_arima_order(stationary_series)
        
        # Fit model
        try:
            if self.arima_params['seasonal']:
                model = ARIMA(
                    series, 
                    order=order[:3],
                    seasonal_order=order[3:]
                )
            else:
                model = ARIMA(series, order=order)
            
            fitted_model = model.fit()
            
            logger.info(f"ARIMA model fitted for {symbol}")
            logger.info(f"Model summary:\n{fitted_model.summary()}")
            
            return {
                'model': fitted_model,
                'order': order,
                'symbol': symbol,
                'series': series
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model for {symbol}: {e}")
            return None
    
    def generate_forecasts(self, model_info: Dict, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts and residuals"""
        if model_info is None:
            return np.array([]), np.array([])
        
        model = model_info['model']
        series = model_info['series']
        
        # Generate forecasts
        forecast = model.forecast(steps=steps)
        
        # Calculate residuals
        fitted_values = model.fittedvalues
        residuals = series - fitted_values
        
        return forecast, residuals
    
    def fit_all_models(self, data: pd.DataFrame) -> Dict:
        """Fit ARIMA models for all commodities"""
        logger.info("Fitting ARIMA models for all commodities")
        
        # Get price columns
        price_cols = [col for col in data.columns if col.endswith('_close')]
        
        if not price_cols:
            logger.error("No price columns found")
            return {}
        
        for col in price_cols:
            symbol = col.replace('_close', '')
            series = data[col].dropna()
            
            if len(series) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Fit model
            model_info = self.fit_arima_model(series, symbol)
            
            if model_info is not None:
                self.models[symbol] = model_info
                
                # Generate forecasts and residuals
                forecast, residuals = self.generate_forecasts(model_info)
                
                self.forecasts[symbol] = forecast
                self.residuals[symbol] = residuals
                
                logger.info(f"Completed ARIMA model for {symbol}")
        
        logger.info(f"Fitted {len(self.models)} ARIMA models")
        return self.models
    
    def create_arima_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ARIMA-based features for the dataset"""
        logger.info("Creating ARIMA features")
        
        # Fit all models
        self.fit_all_models(data)
        
        # Create features DataFrame
        features_df = data[['date']].copy()
        
        for symbol, model_info in self.models.items():
            if model_info is None:
                continue
            
            # Add forecasts
            if symbol in self.forecasts:
                features_df[f'{symbol}_arima_forecast'] = self.forecasts[symbol]
            
            # Add residuals
            if symbol in self.residuals:
                features_df[f'{symbol}_arima_residual'] = self.residuals[symbol]
            
            # Add model diagnostics
            model = model_info['model']
            features_df[f'{symbol}_arima_aic'] = model.aic
            features_df[f'{symbol}_arima_bic'] = model.bic
            
            # Add forecast confidence intervals
            try:
                forecast_ci = model.get_forecast(steps=1).conf_int()
                features_df[f'{symbol}_arima_ci_lower'] = forecast_ci.iloc[0, 0]
                features_df[f'{symbol}_arima_ci_upper'] = forecast_ci.iloc[0, 1]
            except:
                features_df[f'{symbol}_arima_ci_lower'] = np.nan
                features_df[f'{symbol}_arima_ci_upper'] = np.nan
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill')
        
        logger.info(f"Created ARIMA features with shape: {features_df.shape}")
        return features_df
    
    def evaluate_arima_models(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Evaluate ARIMA models on test data"""
        logger.info("Evaluating ARIMA models")
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Fit models on training data
        self.fit_all_models(train_data)
        
        # Evaluate on test data
        results = {}
        
        for symbol, model_info in self.models.items():
            if model_info is None:
                continue
            
            # Get test series
            test_col = f'{symbol}_close'
            if test_col not in test_data.columns:
                continue
            
            test_series = test_data[test_col].dropna()
            
            if len(test_series) < 10:
                continue
            
            # Generate forecasts
            try:
                forecast = model_info['model'].forecast(steps=len(test_series))
                
                # Calculate metrics
                mse = np.mean((test_series - forecast) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(test_series - forecast))
                mape = np.mean(np.abs((test_series - forecast) / test_series)) * 100
                
                results[symbol] = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'mse': mse
                }
                
                logger.info(f"{symbol} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                continue
        
        return results
    
    def save_models(self, filepath: str):
        """Save ARIMA models"""
        import pickle
        
        model_data = {
            'models': self.models,
            'forecasts': self.forecasts,
            'residuals': self.residuals
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved ARIMA models to {filepath}")
    
    def load_models(self, filepath: str):
        """Load ARIMA models"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.forecasts = model_data['forecasts']
        self.residuals = model_data['residuals']
        
        logger.info(f"Loaded ARIMA models from {filepath}")

def main():
    """Main function for ARIMA baseline"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ARIMA baseline
    arima_baseline = ARIMABaseline(config)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'CL=F_close': 100 + np.cumsum(np.random.randn(1000) * 0.02),
        'NG=F_close': 50 + np.cumsum(np.random.randn(1000) * 0.03),
        'MTF=F_close': 200 + np.cumsum(np.random.randn(1000) * 0.01)
    })
    
    # Create ARIMA features
    arima_features = arima_baseline.create_arima_features(sample_data)
    
    # Evaluate models
    results = arima_baseline.evaluate_arima_models(sample_data)
    
    print(f"ARIMA features shape: {arima_features.shape}")
    print(f"Evaluation results: {results}")
    print("ARIMA baseline completed successfully!")

if __name__ == "__main__":
    main()
