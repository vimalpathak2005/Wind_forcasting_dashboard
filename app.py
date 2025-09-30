# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

# project imports
from src.ingest import fetch_isd_lite, locations, validate_station_data, get_alternative_stations
from src.preprocess import preprocess_monthly, debug_data_info, generate_preprocessing_report
from src.models.forecast import fit_arima, forecast_arima, calculate_dataset_statistics, generate_forecast_report

st.set_page_config(page_title="🌬️ Wind Forecasting Dashboard", layout="wide")

st.title("🌬️ Wind Speed Forecasting and Analysis")
st.markdown("This app fetches, preprocesses, and forecasts wind speed data using ARIMA.")

# Custom CSS for better styling with PROPER TEXT VISIBILITY
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        color: white !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 15px 0;
        color: #333333 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .info-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #17a2b8;
        margin: 15px 0;
        color: #0c5460 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
        color: #155724 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stat-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .dark-stat-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #d63384 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .card-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #FFFFFF !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .dark-card-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333333 !important;
    }
    .card-content {
        font-size: 1.1em;
        line-height: 1.8;
        color: #F0F0F0 !important;
    }
    .dark-card-content {
        font-size: 1.1em;
        line-height: 1.8;
        color: #333333 !important;
        font-weight: 500;
    }
    .info-card-content {
        font-size: 1.1em;
        line-height: 1.8;
        color: #0c5460 !important;
        font-weight: 500;
    }
    .card-divider {
        height: 3px;
        background: rgba(255,255,255,0.4);
        margin: 15px 0;
        border-radius: 2px;
    }
    .dark-card-divider {
        height: 3px;
        background: rgba(0,0,0,0.2);
        margin: 15px 0;
        border-radius: 2px;
    }
    .info-card-divider {
        height: 3px;
        background: rgba(12, 84, 96, 0.3);
        margin: 15px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar - Station selection
# ---------------------------
st.sidebar.header("🔧Forcasting Configuration")
station_choice = st.sidebar.selectbox(
    "📍 Choose Location", options=list(locations.keys()),
    format_func=lambda x: locations[x][0]
)

forecast_months = st.sidebar.slider("📅 Forecast months", min_value=3, max_value=24, value=12)
show_detailed_stats = st.sidebar.checkbox("📊 Show Detailed Statistics", value=True)
show_raw_json = st.sidebar.checkbox("🔧 Show Raw JSON Data", value=False)

if st.sidebar.button("🚀 Run Full Analysis"):
    city, station_id, alternative_ids = locations[station_choice]

    st.subheader(f"📡 Fetching Data for {city} ({station_id})")

    # ---------------------------
    # Step 1: Data ingestion
    # ---------------------------
    all_data = []
    years_with_data = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(range(2015, 2025)):
        status_text.text(f"📥 Downloading data for {year}...")
        df = fetch_isd_lite(station_id, year)
        if df is not None and not df.empty:
            all_data.append(df)
            years_with_data += 1
            st.success(f"✅ {year}: {len(df)} records")
        else:
            # Try alternative stations
            for alt_id in alternative_ids:
                df_alt = fetch_isd_lite(alt_id, year)
                if df_alt is not None and not df_alt.empty:
                    df_alt['ORIGINAL_STATION'] = station_id
                    df_alt['SOURCE_STATION'] = alt_id
                    all_data.append(df_alt)
                    years_with_data += 1
                    st.warning(f"🔄 {year}: {len(df_alt)} records from alternative station {alt_id}")
                    break
            else:
                st.error(f"❌ No data available for {year}")
        
        progress_bar.progress((i + 1) / 10)

    if not all_data:
        st.error("❌ No data downloaded. Please check the station ID and your internet connection.")
        st.stop()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.dropna(subset=['DATE'])

    # Generate comprehensive statistics
    if final_df is not None and not final_df.empty:
        stats = validate_station_data(final_df, station_id, city)
    else:
        st.error("❌ No valid data was downloaded. Please try a different station or check your connection.")
        st.stop()
    
    st.success(f"✅ Downloaded {years_with_data} years of data ({len(final_df):,} total records)")

    # ---------------------------
    # Enhanced Statistics Display
    # ---------------------------
    if stats.get('status') != 'NO_DATA':
        st.subheader("📊 Station Statistics Overview")
        
        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", f"{stats.get('total_records', 0):,}")
        with col2:
            completeness = stats.get('data_quality', {}).get('wind_speed', {}).get('completeness', 0)
            st.metric("✅ Data Completeness", f"{completeness:.1f}%")
        with col3:
            mean_wspd = stats.get('data_quality', {}).get('wind_speed', {}).get('mean', 0)
            st.metric("🌬️ Avg Wind Speed", f"{mean_wspd:.2f} m/s")
        with col4:
            st.metric("📅 Years Available", stats.get('years_available', 0))
        
        # Detailed statistics in expanders
        if show_detailed_stats:
            # Data Quality Overview
            with st.expander("📈 Data Quality Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Wind Speed Statistics
                    wspd_stats = stats.get('data_quality', {}).get('wind_speed', {})
                    st.markdown("### 🌬️ Wind Speed Statistics")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="card-title">📊 Wind Speed Analysis</div>
                        <div class="card-divider"></div>
                        <div class="card-content">
                            <div>📈 Mean: <span class="stat-value">{wspd_stats.get('mean', 0):.2f} m/s</span></div>
                            <div>📊 Std Dev: <span class="stat-value">{wspd_stats.get('std', 0):.2f} m/s</span></div>
                            <div>📉 Min: <span class="stat-value">{wspd_stats.get('min', 0):.2f} m/s</span></div>
                            <div>📈 Max: <span class="stat-value">{wspd_stats.get('max', 0):.2f} m/s</span></div>
                            <div>✅ Completeness: <span class="stat-value">{wspd_stats.get('completeness', 0):.1f}%</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Temporal Coverage
                    st.markdown("### 📅 Temporal Coverage")
                    date_range = stats.get('date_range', {})
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="dark-card-title">⏰ Data Timeline</div>
                        <div class="info-card-divider"></div>
                        <div class="info-card-content">
                            <div>📅 Start Date: <strong>{date_range.get('start', 'N/A')}</strong></div>
                            <div>📅 End Date: <strong>{date_range.get('end', 'N/A')}</strong></div>
                            <div>⏱️ Total Duration: <strong>{stats.get('years_available', 0)} years</strong></div>
                            <div>📋 Total Records: <strong>{stats.get('total_records', 0):,}</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Yearly Breakdown Visualization
            with st.expander("📊 Yearly Data Analysis", expanded=True):
                if 'yearly_breakdown' in stats:
                    yearly_data = stats['yearly_breakdown']
                    years = list(yearly_data.keys())
                    records = [yearly_data[year]['records'] for year in years]
                    means = [yearly_data[year]['wind_speed_mean'] for year in years]
                    completeness = [yearly_data[year]['data_completeness'] for year in years]
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('📊 Records per Year', '🌬️ Average Wind Speed', 
                                      '✅ Data Completeness', '📈 Wind Speed Trend'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Records per year
                    fig.add_trace(
                        go.Bar(x=years, y=records, name='Records', marker_color='#1f77b4'),
                        row=1, col=1
                    )
                    
                    # Average wind speed
                    fig.add_trace(
                        go.Scatter(x=years, y=means, name='Avg Speed', line=dict(color='#ff7f0e', width=3), 
                                 mode='lines+markers', marker=dict(size=8)),
                        row=1, col=2
                    )
                    
                    # Data completeness
                    fig.add_trace(
                        go.Bar(x=years, y=completeness, name='Completeness', marker_color='#2ca02c'),
                        row=2, col=1
                    )
                    
                    # Wind speed trend
                    fig.add_trace(
                        go.Scatter(x=years, y=means, name='Speed Trend', 
                                 line=dict(color='#d62728', width=4), mode='lines'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        height=600, 
                        showlegend=False,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(240,240,240,0.1)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data Quality Assessment
            with st.expander("🔍 Data Quality Assessment"):
                completeness = stats.get('data_quality', {}).get('wind_speed', {}).get('completeness', 0)
                years_available = stats.get('years_available', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Completeness gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = completeness,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "📊 Data Completeness", 'font': {'size': 16}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "#FF6B6B"},
                                {'range': [50, 80], 'color': "#FFD93D"},
                                {'range': [80, 100], 'color': "#6BCF7F"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Years available
                    fig = go.Figure(go.Indicator(
                        mode = "number+delta",
                        value = years_available,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "📅 Years of Data", 'font': {'size': 16}},
                        number = {'suffix': " years", 'font': {'size': 30}},
                        delta = {'reference': 5, 'position': "bottom"}
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    # Quality score
                    quality_score = min(100, (completeness * 0.6 + min(years_available, 10) * 4))
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = quality_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "⭐ Overall Quality Score", 'font': {'size': 16}},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "purple"},
                            'steps': [
                                {'range': [0, 60], 'color': "#FF6B6B"},
                                {'range': [60, 80], 'color': "#FFD93D"},
                                {'range': [80, 100], 'color': "#6BCF7F"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show raw JSON if requested
        if show_raw_json:
            with st.expander("🔧 Raw JSON Data"):
                st.json(stats)

    # ---------------------------
    # Step 2: Preprocessing
    # ---------------------------
    with st.spinner("🔄 Preprocessing data to monthly format..."):
        try:
            monthly_data = preprocess_monthly(final_df)
            preprocess_report = generate_preprocessing_report(final_df, monthly_data)
            
            st.success(f"✅ Processed {len(monthly_data)} monthly records.")
            
            # Enhanced Preprocessing Report
            if show_detailed_stats:
                with st.expander("🔄 Preprocessing Report", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📥 Input Data")
                        raw_stats = preprocess_report.get('raw_data_stats', {})
                        st.markdown(f"""
                        <div class="info-card">
                            <div class="dark-card-title">📥 Raw Data Summary</div>
                            <div class="info-card-divider"></div>
                            <div class="info-card-content">
                                <div>📊 Records: <strong>{raw_stats.get('total_records', 0):,}</strong></div>
                                <div>🌬️ Mean WSPD: <strong>{raw_stats.get('wind_speed_stats', {}).get('mean', 0):.2f} m/s</strong></div>
                                <div>✅ Completeness: <strong>{raw_stats.get('wind_speed_stats', {}).get('completeness', 0):.1f}%</strong></div>
                                <div>📅 Date Range: <strong>{raw_stats.get('date_range', 'N/A')}</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📤 Output Data")
                        proc_stats = preprocess_report.get('processed_data_stats', {})
                        # Calculate efficiency
                        records_lost = preprocess_report.get('processing_notes', {}).get('records_lost', 0)
                        efficiency = (len(monthly_data) * 30) / raw_stats.get('total_records', 1) * 100
                        
                        st.markdown(f"""
                        <div class="success-card">
                            <div class="dark-card-title">📤 Processed Data Summary</div>
                            <div class="dark-card-divider"></div>
                            <div class="dark-card-content">
                                <div>📈 Monthly Records: <strong>{proc_stats.get('total_months', 0)}</strong></div>
                                <div>🌬️ Mean WSPD: <strong>{proc_stats.get('wind_speed_stats', {}).get('mean', 0):.2f} m/s</strong></div>
                                <div>📅 Date Range: <strong>{proc_stats.get('date_range', 'N/A')}</strong></div>
                                <div>⚡ Processing Efficiency: <strong>{efficiency:.1f}%</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Processing summary
                    st.markdown("### ⚙️ Processing Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Records Processed", f"{raw_stats.get('total_records', 0):,}")
                    col2.metric("📈 Monthly Aggregates", len(monthly_data))
                    col3.metric("⚡ Processing Efficiency", f"{efficiency:.1f}%")
                    
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {e}")
            st.stop()

    # ---------------------------
    # Step 2.5: Data Validation
    # ---------------------------
    try:
        from src.models.forecast import validate_data_before_forecast
        validate_data_before_forecast(monthly_data)
        st.success("✅ Data validation passed")
    except Exception as e:
        st.error(f"❌ Data validation failed: {e}")
        st.stop()

    # Check if we have enough data for forecasting
    if len(monthly_data) < 24:
        st.warning(f"⚠️ Limited data: Only {len(monthly_data)} months available (recommended: 24+ months)")
        st.info("Forecast may be less accurate due to limited historical data")
    
    # ---------------------------
    # Step 3: Forecasting
    # ---------------------------
    with st.spinner("🤖 Fitting ARIMA model..."):
        try:
            # Calculate dataset statistics
            dataset_stats = calculate_dataset_statistics(monthly_data)
            
            # Fit model with statistics
            model, pre_model_stats, model_stats = fit_arima(monthly_data)
            
            # Generate forecast
            forecast_df, forecast_stats = forecast_arima(model, periods=forecast_months)
            
            # Generate comprehensive report
            forecast_report = generate_forecast_report(
                dataset_stats, model_stats, forecast_stats, forecast_df
            )
            
            st.success("✅ Forecasting completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error in forecasting: {e}")
            st.stop()

    # ---------------------------
    # Step 4: Enhanced Dashboard Visuals
    # ---------------------------
    st.subheader("📊 Historical vs Forecasted Wind Speed")
    
    # Use Plotly for interactive charts
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Historical data
    fig.add_trace(
        go.Scatter(x=monthly_data["DATE"], y=monthly_data["WSPD"], 
                  name="Historical", line=dict(color='#1f77b4', width=3), 
                  fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'),
        secondary_y=False,
    )
    
    # Forecast data
    fig.add_trace(
        go.Scatter(x=forecast_df["DATE"], y=forecast_df["FORECAST_WSPD"], 
                  name="Forecast", line=dict(color='#ff7f0e', width=3, dash='dash')),
        secondary_y=False,
    )
    
    # Confidence interval
    if 'LOWER_CI' in forecast_df.columns:
        fig.add_trace(
            go.Scatter(x=forecast_df["DATE"], y=forecast_df["UPPER_CI"],
                      fill=None, line=dict(color='#ff7f0e', width=1),
                      showlegend=False, name="Upper CI"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=forecast_df["DATE"], y=forecast_df["LOWER_CI"],
                      fill='tonexty', line=dict(color='#ff7f0e', width=1),
                      fillcolor='rgba(255, 127, 14, 0.2)',
                      name="95% Confidence Interval"),
            secondary_y=False,
        )
    
    fig.update_layout(
        title="🌬️ Wind Speed Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Wind Speed (m/s)",
        hovermode="x unified",
        height=500,
        plot_bgcolor='rgba(240,240,240,0.8)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Step 5: Enhanced Insights
    # ---------------------------
    st.subheader("🔎 Forecast Insights & Statistics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Historical Months", f"{len(monthly_data)}")
    col2.metric("🌬️ Last Observed WSPD", f"{monthly_data['WSPD'].iloc[-1]:.2f} m/s")
    col3.metric("📊 Next Month Forecast", f"{forecast_df['FORECAST_WSPD'].iloc[0]:.2f} m/s")
    col4.metric("📈 Forecast Period Avg", f"{forecast_df['FORECAST_WSPD'].mean():.2f} m/s")
    
    # Model information in a nice card format
    with st.expander("🤖 Model Information", expanded=True):
        if 'model_information' in forecast_report:
            model_info = forecast_report['model_information']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### ⚙️ Model Configuration")
                st.markdown(f"""
                <div class="info-card">
                    <div class="dark-card-title">⚙️ Model Configuration</div>
                    <div class="info-card-divider"></div>
                    <div class="info-card-content">
                        <div>📊 ARIMA Order: <strong>{model_info.get('model_order', 'N/A')}</strong></div>
                        <div>🔄 Seasonal Order: <strong>{model_info.get('seasonal_order', 'Non-seasonal')}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Model Quality")
                st.markdown(f"""
                <div class="success-card">
                    <div class="dark-card-title">📊 Model Quality</div>
                    <div class="dark-card-divider"></div>
                    <div class="dark-card-content">
                        <div>🎯 AIC: <strong>{model_info.get('aic', 'N/A'):.2f}</strong></div>
                        <div>📈 BIC: <strong>{model_info.get('bic', 'N/A'):.2f}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 🔮 Forecast Summary")
                forecast_summary = forecast_report.get('forecast_summary', {})
                trend_icon = "📈" if forecast_summary.get('forecast_trend') == 'increasing' else "📉"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">🔮 Forecast Summary</div>
                    <div class="card-divider"></div>
                    <div class="card-content">
                        <div>{trend_icon} Trend: <strong>{forecast_summary.get('forecast_trend', 'N/A')}</strong></div>
                        <div>📊 Avg Forecast: <strong>{forecast_summary.get('average_forecast', 0):.2f} m/s</strong></div>
                        <div>📅 Periods: <strong>{forecast_summary.get('periods', 0)} months</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Recommendations in a visually appealing way
    if 'recommendations' in forecast_report and forecast_report['recommendations']:
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(forecast_report['recommendations'], 1):
            st.markdown(f"""
            <div class="warning-card">
                <div class="dark-card-title">💡 Recommendation {i}</div>
                <div class="dark-card-divider"></div>
                <div class="dark-card-content">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------
    # Step 6: Export options
    # ---------------------------
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", csv_forecast, "wind_forecast.csv", "text/csv")
    
    with col2:
        csv_monthly = monthly_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Monthly Data CSV", csv_monthly, "monthly_data.csv", "text/csv")
    
    with col3:
        json_report = json.dumps(forecast_report, indent=2).encode("utf-8")
        st.download_button("📊 Download Full Report JSON", json_report, "forecast_report.json", "application/json")

# Add instructions when no forecast has been run
else:
    st.info("👈 Select a location and click 'Run Full Analysis' to start analysis.")
    
    st.write("**📍 Available Stations with Alternatives:**")
    for key, (city, main_id, alternatives) in locations.items():
        st.markdown(f"""
        <div style="background-color: #0f1624; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4 style="margin: 0; color: #1f77b4;">🏙️ {city}</h4>
            <p style="margin: 5px 0;">📡 Primary: <code>{main_id}</code></p>
            <p style="margin: 5px 0;">🔄 Alternatives: {', '.join(alternatives)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Add some helpful information
    st.markdown("---")
    st.subheader("ℹ️ About This App")
    st.write("""
    This wind forecasting application:
    - 🌐 Fetches historical wind data from NOAA ISD-Lite database
    - 🔄 Preprocesses data to monthly format
    - 🤖 Uses ARIMA models for forecasting
    - 📊 Provides detailed statistics and visualizations
    - 🔄 Handles missing data with alternative stations
    """)
    
    st.subheader("📊 Expected Data Coverage")
    st.write("""
    The app will attempt to fetch data for 2015-2024 (10 years). 
    If primary station data is unavailable, it will automatically try alternative stations.
    """)