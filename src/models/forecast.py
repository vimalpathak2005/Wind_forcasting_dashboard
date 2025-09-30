import pandas as pd
import pmdarima as pm
import numpy as np
from datetime import datetime
import warnings
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()
    else:
        return obj

def calculate_dataset_statistics(data):
    """
    Calculate comprehensive statistics for the dataset with FIXED timespan calculation
    """
    if data is None or data.empty:
        return {"error": "No data available"}
    
    # FIX: Proper timespan calculation with error handling
    try:
        if 'DATE' not in data.columns:
            raise ValueError("DATE column missing")
            
        # Ensure dates are proper datetime objects
        dates = pd.to_datetime(data['DATE'], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            timespan_days = 0
            timespan_years = 0
        else:
            timespan_days = (valid_dates.max() - valid_dates.min()).days
            timespan_years = timespan_days / 365.25 if timespan_days > 0 else 0
            
    except Exception as e:
        logger.warning(f"Error calculating timespan: {e}")
        # Fallback calculation based on number of months
        timespan_years = len(data) / 12.0 if len(data) > 0 else 0
    
    # FIX: Ensure basic_info structure exists even if some calculations fail
    stats = {
        'basic_info': {
            'total_months': len(data),
            'date_range': {
                'start': data['DATE'].min().strftime('%Y-%m-%d') if 'DATE' in data.columns and len(data) > 0 else 'N/A',
                'end': data['DATE'].max().strftime('%Y-%m-%d') if 'DATE' in data.columns and len(data) > 0 else 'N/A',
                'timespan_years': float(timespan_years)  # FIX: Always ensure this key exists
            },
            'data_completeness': float(1 - data['WSPD'].isnull().sum() / len(data)) * 100 if 'WSPD' in data.columns and len(data) > 0 else 0
        }
    }
    
    # Only add wind speed statistics if WSPD column exists
    if 'WSPD' in data.columns and len(data) > 0:
        stats['wind_speed_statistics'] = {
            'mean': float(data['WSPD'].mean()),
            'median': float(data['WSPD'].median()),
            'std': float(data['WSPD'].std()),
            'min': float(data['WSPD'].min()),
            'max': float(data['WSPD'].max()),
            'percentile_25': float(data['WSPD'].quantile(0.25)) if len(data) > 1 else float(data['WSPD'].iloc[0]),
            'percentile_75': float(data['WSPD'].quantile(0.75)) if len(data) > 1 else float(data['WSPD'].iloc[0])
        }
    else:
        stats['wind_speed_statistics'] = {
            'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'percentile_25': 0, 'percentile_75': 0
        }
    
    stats = convert_numpy_types(stats)
    return stats

def validate_data_before_forecast(monthly_data):
    """
    Enhanced data validation before forecasting
    """
    if monthly_data is None or monthly_data.empty:
        raise ValueError("Monthly data is empty or None")
    
    if 'WSPD' not in monthly_data.columns:
        raise ValueError("WSPD column missing in monthly data")
    
    if len(monthly_data) < 12:
        raise ValueError(f"Insufficient data: {len(monthly_data)} months (need at least 12)")
    
    # Check for constant values
    if monthly_data['WSPD'].std() < 0.001:
        raise ValueError("Wind speed data has near-zero variance")
    
    # Check for realistic wind speed values
    wspd_mean = monthly_data['WSPD'].mean()
    if wspd_mean > 50 or wspd_mean < 0:
        raise ValueError(f"Unrealistic wind speed mean: {wspd_mean:.2f} m/s")
    
    # Check for excessive missing values
    missing_pct = monthly_data['WSPD'].isnull().sum() / len(monthly_data) * 100
    if missing_pct > 20:
        raise ValueError(f"Too many missing values: {missing_pct:.1f}%")
    
    logger.info("Data validation passed")
    return True

def fit_arima(data, column="WSPD"):
    """
    Enhanced ARIMA fitting with better validation and statistics
    """
    validate_data_before_forecast(data)
    
    data = data.sort_values("DATE").copy()
    
    logger.info(f"Fitting model on {len(data)} months of data")
    logger.info(f"Data range: {data['DATE'].min()} to {data['DATE'].max()}")
    
    series = data.set_index("DATE")[column]
    
    if series.isna().any():
        logger.warning(f"Found {series.isna().sum()} missing values, interpolating...")
        series = series.interpolate(method='linear').ffill().bfill()
    
    pre_model_stats = {
        'series_length': len(series),
        'mean': float(series.mean()),
        'std': float(series.std())
    }
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # FIX: Use simpler ARIMA configuration for better stability
            model = pm.auto_arima(series,
                                start_p=0, start_q=0,
                                max_p=2, max_q=2,  # Reduced complexity
                                seasonal=True,
                                m=12,
                                start_P=0,
                                start_Q=0,
                                max_P=1,  # Reduced seasonal complexity
                                max_Q=1,
                                stepwise=True,
                                suppress_warnings=True,
                                error_action='ignore',
                                trace=False,  # Reduced verbosity
                                n_fits=5,    # Reduced number of fits
                                information_criterion='aic')
        
        if model is None:
            raise ValueError("Auto ARIMA failed to find a suitable model")
            
        logger.info(f"Model fitted: ARIMA{model.order} Seasonal{model.seasonal_order}")
        
        model_stats = {
            'model_order': model.order,
            'seasonal_order': model.seasonal_order,
            'aic': float(model.aic()) if hasattr(model, 'aic') else None,
            'bic': float(model.bic()) if hasattr(model, 'bic') else None
        }
        
        model_stats = convert_numpy_types(model_stats)
        return model, pre_model_stats, model_stats
        
    except Exception as e:
        logger.error(f"ARIMA model fitting failed: {e}")
        
        # Fallback: try non-seasonal ARIMA with even simpler settings
        try:
            logger.info("Trying non-seasonal ARIMA as fallback...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = pm.auto_arima(series,
                                    seasonal=False,
                                    start_p=0, start_q=0,
                                    max_p=1, max_q=1,  # Very simple model
                                    stepwise=True,
                                    suppress_warnings=True,
                                    error_action='ignore')
            
            if model is not None:
                logger.info(f"Fallback model fitted: ARIMA{model.order}")
                model_stats = {
                    'model_order': model.order,
                    'seasonal_order': None,
                    'aic': float(model.aic()) if hasattr(model, 'aic') else None,
                    'bic': float(model.bic()) if hasattr(model, 'bic') else None
                }
                model_stats = convert_numpy_types(model_stats)
                return model, pre_model_stats, model_stats
            else:
                raise ValueError("Fallback ARIMA also failed")
                
        except Exception as fallback_error:
            # Ultimate fallback: use simple average model
            logger.warning("Using simple average model as last resort")
            class SimpleAverageModel:
                def __init__(self, series):
                    self.series_mean = series.mean()
                    self.order = (0, 0, 0)
                    self.seasonal_order = (0, 0, 0, 0)
                    
                def predict(self, n_periods=1):
                    return np.full(n_periods, self.series_mean)
                    
                def aic(self):
                    return 1000  # Placeholder
                    
                def bic(self):
                    return 1000  # Placeholder
            
            model = SimpleAverageModel(series)
            model_stats = {
                'model_order': model.order,
                'seasonal_order': model.seasonal_order,
                'aic': float(model.aic()),
                'bic': float(model.bic()),
                'note': 'Simple average model used as fallback'
            }
            return model, pre_model_stats, model_stats

def forecast_arima(model, periods=12, confidence_level=0.95):
    """
    Enhanced forecasting with better confidence intervals and validation
    """
    try:
        # FIX: Handle different model types
        if hasattr(model, 'predict'):
            forecast, conf_int = model.predict(
                n_periods=periods, 
                return_conf_int=True,
                alpha=1-confidence_level
            )
        else:
            # For simple average model
            forecast = model.predict(n_periods=periods)
            conf_int = np.column_stack([forecast - 1, forecast + 1])  # Simple confidence interval
        
        last_date = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods, 
            freq='MS'
        )
        
        result_df = pd.DataFrame({
            'DATE': future_dates,
            'FORECAST_WSPD': forecast,
            'LOWER_CI': conf_int[:, 0] if conf_int is not None else forecast - 1,
            'UPPER_CI': conf_int[:, 1] if conf_int is not None else forecast + 1,
            'CONFIDENCE_LEVEL': confidence_level
        })
        
        forecast_stats = {
            'forecast_mean': float(forecast.mean()),
            'forecast_std': float(forecast.std()),
            'confidence_interval_width': float((result_df['UPPER_CI'] - result_df['LOWER_CI']).mean()),
            'trend_direction': 'increasing' if forecast[-1] > forecast[0] else 'decreasing'
        }
        
        logger.info(f"Forecast created for {periods} months")
        return result_df, forecast_stats
        
    except Exception as e:
        logger.error(f"Standard forecast failed: {e}")
        return forecast_arima_simple(model, periods)

def forecast_arima_simple(model, periods=12):
    """
    Simplified forecast without confidence intervals
    """
    try:
        if hasattr(model, 'predict'):
            forecast = model.predict(n_periods=periods)
        else:
            forecast = model.predict(n_periods=periods)
        
        last_date = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods, 
            freq='MS'
        )
        
        result_df = pd.DataFrame({
            'DATE': future_dates,
            'FORECAST_WSPD': forecast
        })
        
        forecast_stats = {
            'forecast_mean': float(forecast.mean()),
            'forecast_std': float(forecast.std())
        }
        
        logger.info(f"Simple forecast created for {periods} months")
        return result_df, forecast_stats
        
    except Exception as e:
        raise ValueError(f"Simple forecasting failed: {e}")

def generate_forecast_report(raw_stats, model_stats, forecast_stats, forecast_df):
    """
    Generate comprehensive forecast report with FIXED timespan handling
    """
    # FIX: Ensure all required keys exist in raw_stats
    if 'basic_info' not in raw_stats:
        raw_stats = {'basic_info': {'total_months': 0, 'date_range': {'timespan_years': 0}, 'data_completeness': 0}}
    
    if 'date_range' not in raw_stats['basic_info']:
        raw_stats['basic_info']['date_range'] = {'timespan_years': 0}
    
    report = {
        'dataset_summary': raw_stats,
        'model_information': model_stats,
        'forecast_results': forecast_stats,
        'forecast_summary': {
            'periods': len(forecast_df),
            'date_range': {
                'start': forecast_df['DATE'].min().strftime('%Y-%m-%d') if len(forecast_df) > 0 else 'N/A',
                'end': forecast_df['DATE'].max().strftime('%Y-%m-%d') if len(forecast_df) > 0 else 'N/A'
            },
            'average_forecast': float(forecast_df['FORECAST_WSPD'].mean()) if len(forecast_df) > 0 else 0,
            'forecast_trend': 'increasing' if len(forecast_df) > 0 and forecast_df['FORECAST_WSPD'].iloc[-1] > forecast_df['FORECAST_WSPD'].iloc[0] else 'decreasing'
        },
        'recommendations': generate_recommendations(raw_stats, model_stats, forecast_stats)
    }
    
    report = convert_numpy_types(report)
    return report

def generate_recommendations(raw_stats, model_stats, forecast_stats):
    """
    Generate recommendations based on data quality and model results
    """
    recommendations = []
    
    # FIX: Safe access to nested keys
    try:
        completeness = raw_stats.get('basic_info', {}).get('data_completeness', 100)
        if completeness < 80:
            recommendations.append("Consider finding additional data sources due to low completeness")
        
        timespan_years = raw_stats.get('basic_info', {}).get('date_range', {}).get('timespan_years', 0)
        if timespan_years < 2:
            recommendations.append("Forecast reliability may be low due to short data history")
    except:
        recommendations.append("Data quality assessment unavailable")
    
    return recommendations