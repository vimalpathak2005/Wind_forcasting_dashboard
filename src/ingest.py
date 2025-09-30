import pandas as pd
import gzip
import numpy as np
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import time
import os
from datetime import datetime

def fetch_isd_lite(station_id, year):
    """
    Fetch ISD-Lite data with enhanced error handling and data validation
    """
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz"
    
    try:
        print(f"📥 Downloading data for {year}...")
        response = urlopen(url, timeout=30)
        
        with gzip.open(response, 'rt') as f:
            df = pd.read_fwf(
                f,
                colspecs=[(0,4),(4,6),(6,8),(8,10),(10,16),(16,22),(22,28),
                         (28,34),(34,40),(40,46),(46,52)],
                names=["YEAR","MONTH","DAY","HOUR","TEMP","DEWP","SLP",
                      "WDIR","WSPD","CLDC","PRCP"],
                na_values=[-9999]
            )
        
        # Replace any remaining -9999 values with NaN
        df = df.replace(-9999, np.nan)
        
        # Validate we have meaningful data
        valid_records = len(df.dropna(subset=['WSPD']))
        if valid_records == 0:
            print(f"⚠️  No valid wind speed data for {year}")
            return None
        
        # Convert numeric columns to appropriate types
        for col in ['YEAR','MONTH','DAY','HOUR']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        numeric_columns = ['TEMP', 'DEWP', 'SLP', 'WDIR', 'WSPD', 'CLDC', 'PRCP']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # FIX: PROPER WIND SPEED SCALING - ISD-Lite stores as tenths of m/s
        if 'WSPD' in df.columns:
            df['WSPD'] = df['WSPD'] / 10.0  # Convert to m/s
        
        # FIX: Validate wind speed values are realistic
        if 'WSPD' in df.columns:
            # Remove unrealistic wind speeds (above 50 m/s = 180 km/h)
            unrealistic_mask = (df['WSPD'] > 50) | (df['WSPD'] < 0)
            if unrealistic_mask.any():
                print(f"⚠️  Removing {unrealistic_mask.sum()} unrealistic wind speed values")
                df.loc[unrealistic_mask, 'WSPD'] = np.nan
        
        # Create a datetime column safely
        df['DATE'] = pd.to_datetime(
            df[['YEAR','MONTH','DAY']].assign(HOUR=df['HOUR']), 
            errors='coerce'
        )
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['DATE'])
        
        print(f"✅ Successfully fetched {year}: {valid_records} valid records")
        return df
        
    except (URLError, HTTPError) as e:
        print(f"❌ Network error {year}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing {year}: {e}")
        return None

def validate_station_data(station_data, station_id, city):
    """
    Validate and generate statistics for station data with enhanced checks
    """
    if station_data is None or station_data.empty:
        return {
            'station_id': station_id,
            'city': city,
            'status': 'NO_DATA',
            'message': 'No data available for this station'
        }
    
    def convert_numpy_types(obj):
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
        else:
            return obj
    
    # Enhanced data validation
    if 'WSPD' in station_data.columns:
        station_data['WSPD'] = pd.to_numeric(station_data['WSPD'], errors='coerce')
        
        # Remove extreme outliers
        q1 = station_data['WSPD'].quantile(0.01)
        q99 = station_data['WSPD'].quantile(0.99)
        reasonable_mask = (station_data['WSPD'] >= q1) & (station_data['WSPD'] <= q99)
        reasonable_data = station_data[reasonable_mask]['WSPD']
        
        if len(reasonable_data) > 0:
            mean_wspd = reasonable_data.mean()
            std_wspd = reasonable_data.std()
        else:
            mean_wspd = station_data['WSPD'].mean()
            std_wspd = station_data['WSPD'].std()
    else:
        mean_wspd = 0
        std_wspd = 0
    
    stats = {
        'station_id': station_id,
        'city': city,
        'status': 'SUCCESS',
        'total_records': int(len(station_data)),
        'date_range': {
            'start': station_data['DATE'].min().strftime('%Y-%m-%d'),
            'end': station_data['DATE'].max().strftime('%Y-%m-%d')
        },
        'years_available': int(station_data['YEAR'].nunique()),
        'data_quality': {
            'wind_speed': {
                'completeness': float(1 - station_data['WSPD'].isnull().sum() / len(station_data)) * 100,
                'mean': float(mean_wspd),
                'std': float(std_wspd),
                'min': float(station_data['WSPD'].min()),
                'max': float(station_data['WSPD'].max()),
                'realistic_range': f"{q1:.1f} to {q99:.1f} m/s" if 'WSPD' in station_data else "N/A"
            },
            'wind_direction': {
                'completeness': float(1 - station_data['WDIR'].isnull().sum() / len(station_data)) * 100 if 'WDIR' in station_data else 0
            }
        }
    }
    
    # Yearly breakdown
    yearly_stats = {}
    for year in sorted(station_data['YEAR'].unique()):
        year_data = station_data[station_data['YEAR'] == year]
        yearly_stats[str(year)] = {
            'records': int(len(year_data)),
            'wind_speed_mean': float(year_data['WSPD'].mean()),
            'data_completeness': float(1 - year_data['WSPD'].isnull().sum() / len(year_data)) * 100
        }
    
    stats['yearly_breakdown'] = yearly_stats
    stats = convert_numpy_types(stats)
    
    return stats

def get_alternative_stations(main_station_id):
    """
    Provide alternative stations if main station fails
    """
    alternatives = {
        "421820-99999": ["421810-99999", "421830-99999"],
        "423690-99999": ["423680-99999", "423700-99999"],
        "421110-99999": ["421100-99999", "421120-99999"],
        "429090-99999": ["429080-99999", "429100-99999"],
        "432790-99999": ["432780-99999", "432800-99999"]
    }
    
    return alternatives.get(main_station_id, [])

# Available locations with multiple options
locations = {
    1: ("Delhi", "421820-99999", ["421810-99999", "421830-99999"]),
    2: ("Lucknow", "423690-99999", ["423680-99999", "423700-99999"]),
    3: ("Agra", "421110-99999", ["421100-99999", "421120-99999"]),
    4: ("Mumbai", "429090-99999", ["429080-99999", "429100-99999"]),
    5: ("Chennai", "432790-99999", ["432780-99999", "432800-99999"])
}

if __name__ == "__main__":
    print("Available locations:")
    for num, (city, main_id, alternatives) in locations.items():
        print(f"{num}. {city} (Primary: {main_id}, Alternatives: {', '.join(alternatives)})")

    try:
        choice = int(input("Enter number of location: "))
        city, station_id, alternative_ids = locations[choice]
        
        print(f"\nFetching data for {city}...")
        print(f"Primary station: {station_id}")
        print(f"Alternative stations: {', '.join(alternative_ids)}")

        all_data = []
        successful_years = 0
        
        for year in range(2015, 2025):
            df = fetch_isd_lite(station_id, year)
            if df is not None and not df.empty:
                all_data.append(df)
                successful_years += 1
                print(f"✅ {year}: {len(df)} records from primary station")
            else:
                for alt_id in alternative_ids:
                    print(f"🔄 Trying alternative station {alt_id} for {year}...")
                    df_alt = fetch_isd_lite(alt_id, year)
                    if df_alt is not None and not df_alt.empty:
                        df_alt['ORIGINAL_STATION'] = station_id
                        df_alt['SOURCE_STATION'] = alt_id
                        all_data.append(df_alt)
                        successful_years += 1
                        print(f"✅ {year}: {len(df_alt)} records from alternative station {alt_id}")
                        break
                else:
                    print(f"❌ No data available for {year}")
            
            time.sleep(0.3)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            stats = validate_station_data(final_df, station_id, city)
            
            print(f"\n📊 DATA SUMMARY FOR {city.upper()}:")
            print(f"✅ Downloaded {successful_years} years of data ({len(final_df)} total records)")
            print(f"📅 Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"🌬️ Wind speed - Mean: {stats['data_quality']['wind_speed']['mean']:.2f} m/s")
            print(f"📈 Data completeness: {stats['data_quality']['wind_speed']['completeness']:.1f}%")
            
            final_df.to_csv(f"raw_data_{city}_{station_id}.csv", index=False)
            print(f"💾 Raw data saved to raw_data_{city}_{station_id}.csv")
            
            import json
            with open(f"statistics_{city}_{station_id}.json", 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Statistics saved to statistics_{city}_{station_id}.json")
            
        else:
            print("❌ No data downloaded for any year")
            
    except (ValueError, KeyError) as e:
        print(f"❌ Invalid input: {e}")