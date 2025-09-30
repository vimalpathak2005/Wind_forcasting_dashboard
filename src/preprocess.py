import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input_data(df):
    """
    Validate input data before preprocessing with enhanced checks
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or None")
    
    required_columns = ['DATE', 'WSPD']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < 30:
        raise ValueError(f"Insufficient data: {len(df)} records (need at least 30)")
    
    if df['WSPD'].isna().sum() == len(df):
        raise ValueError("All wind speed values are missing")
    
    # Check for realistic wind speed values
    if 'WSPD' in df.columns:
        realistic_data = df['WSPD'].dropna()
        if len(realistic_data) > 0:
            if realistic_data.max() > 50:  # Unrealistically high wind speeds
                logger.warning(f"Unrealistic wind speeds detected: max={realistic_data.max()} m/s")
    
    return True

def preprocess_monthly(df):
    """
    Enhanced preprocessing with outlier removal and better validation
    """
    logger.info("Starting monthly data preprocessing...")
    
    df_clean = df.copy()
    validate_input_data(df_clean)
    
    logger.info(f"Raw data info: {len(df_clean)} records")
    if 'YEAR' in df_clean.columns:
        logger.info(f"Year range: {df_clean['YEAR'].min()} - {df_clean['YEAR'].max()}")
    
    # Ensure DATE column is proper datetime
    df_clean['DATE'] = pd.to_datetime(df_clean['DATE'], errors='coerce')
    
    # Drop invalid dates
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['DATE'])
    dropped_count = initial_count - len(df_clean)
    
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} records with invalid dates")
    
    if df_clean.empty:
        raise ValueError("No valid dates found in the dataset")

    # Remove extreme outliers from wind speed data
    if 'WSPD' in df_clean.columns:
        # Use IQR method to detect outliers
        Q1 = df_clean['WSPD'].quantile(0.25)
        Q3 = df_clean['WSPD'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap extreme values instead of removing them
        df_clean['WSPD'] = np.where(df_clean['WSPD'] < lower_bound, lower_bound, df_clean['WSPD'])
        df_clean['WSPD'] = np.where(df_clean['WSPD'] > upper_bound, upper_bound, df_clean['WSPD'])
        
        logger.info(f"Wind speed range after outlier handling: {df_clean['WSPD'].min():.2f} to {df_clean['WSPD'].max():.2f} m/s")

    # Sort by date and set as index
    df_clean = df_clean.sort_values('DATE')
    df_clean = df_clean.set_index('DATE')

    # Handle missing wind speed values
    missing_before = df_clean['WSPD'].isna().sum()
    logger.info(f"Missing WSPD values before cleaning: {missing_before}")
    
    # Resample to monthly mean
    df_monthly = df_clean[['WSPD']].resample('MS').mean()
    
    # Handle missing values after resampling
    if df_monthly['WSPD'].isna().any():
        missing_count = df_monthly['WSPD'].isna().sum()
        logger.info(f"Missing values after resampling: {missing_count}")
        
        # Try multiple interpolation methods
        df_monthly['WSPD'] = df_monthly['WSPD'].interpolate(method='linear', limit=3)
        
        # If still missing, use seasonal means
        if df_monthly['WSPD'].isna().any():
            monthly_avg = df_monthly.groupby(df_monthly.index.month)['WSPD'].transform('mean')
            df_monthly['WSPD'] = df_monthly['WSPD'].fillna(monthly_avg)
    
    # Drop any remaining NaN values
    df_monthly = df_monthly.dropna()
    
    if df_monthly.empty:
        raise ValueError("No valid monthly data after processing")

    # Reset index for modeling
    df_monthly = df_monthly.reset_index()

    logger.info(f"Final processed data: {len(df_monthly)} months")
    logger.info(f"Date range: {df_monthly['DATE'].min()} to {df_monthly['DATE'].max()}")
    logger.info(f"Wind speed range: {df_monthly['WSPD'].min():.2f} to {df_monthly['WSPD'].max():.2f} m/s")
    
    return df_monthly[['DATE', 'WSPD']]

def generate_preprocessing_report(raw_data, processed_data):
    """
    Generate a comprehensive report on preprocessing results
    """
    report = {
        'raw_data_stats': {
            'total_records': len(raw_data),
            'date_range': f"{raw_data['DATE'].min()} to {raw_data['DATE'].max()}",
            'wind_speed_stats': {
                'mean': raw_data['WSPD'].mean(),
                'std': raw_data['WSPD'].std(),
                'min': raw_data['WSPD'].min(),
                'max': raw_data['WSPD'].max(),
                'completeness': (1 - raw_data['WSPD'].isna().sum() / len(raw_data)) * 100
            }
        },
        'processed_data_stats': {
            'total_months': len(processed_data),
            'date_range': f"{processed_data['DATE'].min()} to {processed_data['DATE'].max()}",
            'wind_speed_stats': {
                'mean': processed_data['WSPD'].mean(),
                'std': processed_data['WSPD'].std(),
                'min': processed_data['WSPD'].min(),
                'max': processed_data['WSPD'].max()
            }
        },
        'processing_notes': {
            'records_lost': len(raw_data) - (len(processed_data) * 30),
            'outlier_handling': 'Applied IQR method to cap extreme values'
        }
    }
    
    return report

def debug_data_info(df):
    """Enhanced debug information"""
    print("\n🔍 DEBUG INFO:")
    print(f"Total records: {len(df)}")
    
    if 'DATE' in df.columns:
        print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        print(f"Timespan: {(df['DATE'].max() - df['DATE'].min()).days} days")
    
    if 'YEAR' in df.columns:
        year_counts = df['YEAR'].value_counts().sort_index()
        print("Records per year:")
        for year, count in year_counts.items():
            completeness = (count / (365 * 24)) * 100
            print(f"  {year}: {count} records ({completeness:.1f}% complete)")
    
    if 'WSPD' in df.columns:
        print(f"Wind speed - Mean: {df['WSPD'].mean():.2f}, Std: {df['WSPD'].std():.2f}")
        print(f"Wind speed - Min: {df['WSPD'].min():.2f}, Max: {df['WSPD'].max():.2f}")
        missing_pct = (df['WSPD'].isna().sum() / len(df)) * 100
        print(f"Missing WSPD values: {df['WSPD'].isna().sum()} ({missing_pct:.1f}%)")
    
    print(f"Columns: {df.columns.tolist()}")
    print()

if __name__ == "__main__":
    try:
        dates = pd.date_range('2015-01-01', '2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'DATE': dates,
            'WSPD': np.random.uniform(2, 10, len(dates))
        })
        
        sample_data.loc[sample_data.sample(frac=0.1).index, 'WSPD'] = np.nan
        
        result = preprocess_monthly(sample_data)
        report = generate_preprocessing_report(sample_data, result)
        
        print("✅ Preprocessing test successful")
        print(f"Sample result: {len(result)} months")
        print(result.head())
        
        print("\n📊 PREPROCESSING REPORT:")
        print(f"Raw records: {report['raw_data_stats']['total_records']}")
        print(f"Processed months: {report['processed_data_stats']['total_months']}")
        print(f"Wind speed completeness: {report['raw_data_stats']['wind_speed_stats']['completeness']:.1f}%")
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()