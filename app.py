import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tempfile
import os
from datetime import datetime

from statsforecast import StatsForecast
from statsforecast.models import (
    HistoricAverage,
    Naive,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
    AutoETS,
    AutoARIMA
)

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *

# Function to load and process uploaded CSV
def load_data(file):
    if file is None:
        return None, "Please upload a CSV file"
    try:
        df = pd.read_csv(file)
        required_cols = ['unique_id', 'ds', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"
        
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        # Check for NaN values
        if df['y'].isna().any():
            return None, "Data contains missing values in the 'y' column"
            
        return df, "Data loaded successfully!"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Function to generate and return a plot
def create_forecast_plot(forecast_df, original_df, title="Forecasting Results"):
    plt.figure(figsize=(12, 7))
    unique_ids = forecast_df['unique_id'].unique()
    forecast_cols = [col for col in forecast_df.columns if col not in ['unique_id', 'ds', 'cutoff']]
    
    colors = plt.cm.tab10.colors
    
    for i, unique_id in enumerate(unique_ids):
        original_data = original_df[original_df['unique_id'] == unique_id]
        plt.plot(original_data['ds'], original_data['y'], 'k-', linewidth=2, label=f'{unique_id} (Actual)')
        
        forecast_data = forecast_df[forecast_df['unique_id'] == unique_id]
        for j, col in enumerate(forecast_cols):
            if col in forecast_data.columns:
                plt.plot(forecast_data['ds'], forecast_data[col], 
                         color=colors[j % len(colors)], 
                         linestyle='--', 
                         linewidth=1.5,
                         label=f'{col}')

    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Format date labels better
    fig = plt.gcf()
    ax = plt.gca()
    fig.autofmt_xdate()
    
    return fig

# Function to create a plot for future forecasts
def create_future_forecast_plot(forecast_df, original_df):
    plt.figure(figsize=(12, 7))
    unique_ids = forecast_df['unique_id'].unique()
    forecast_cols = [col for col in forecast_df.columns if col not in ['unique_id', 'ds']]
    
    colors = plt.cm.tab10.colors
    
    for i, unique_id in enumerate(unique_ids):
        # Plot historical data
        original_data = original_df[original_df['unique_id'] == unique_id]
        plt.plot(original_data['ds'], original_data['y'], 'k-', linewidth=2, label=f'{unique_id} (Historical)')
        
        # Plot forecast data with shaded vertical line separator
        forecast_data = forecast_df[forecast_df['unique_id'] == unique_id]
        
        # Add vertical line at the forecast start
        if not forecast_data.empty and not original_data.empty:
            forecast_start = forecast_data['ds'].min()
            plt.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.5)
            
        for j, col in enumerate(forecast_cols):
            if col in forecast_data.columns:
                plt.plot(forecast_data['ds'], forecast_data[col], 
                         color=colors[j % len(colors)], 
                         linestyle='--', 
                         linewidth=1.5,
                         label=f'{col}')

    plt.title('Future Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Format date labels better
    fig = plt.gcf()
    ax = plt.gca()
    fig.autofmt_xdate()
    
    return fig

# Function to export results to CSV
def export_results(eval_df, cv_results, future_forecasts):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create temp directory if it doesn't exist
    temp_dir = tempfile.mkdtemp()
    
    result_files = []
    
    if eval_df is not None:
        eval_path = os.path.join(temp_dir, f"evaluation_metrics_{timestamp}.csv")
        eval_df.to_csv(eval_path, index=False)
        result_files.append(eval_path)
        
    if cv_results is not None:
        cv_path = os.path.join(temp_dir, f"cross_validation_results_{timestamp}.csv")
        cv_results.to_csv(cv_path, index=False)
        result_files.append(cv_path)
        
    if future_forecasts is not None:
        forecast_path = os.path.join(temp_dir, f"forecasts_{timestamp}.csv")
        future_forecasts.to_csv(forecast_path, index=False)
        result_files.append(forecast_path)
    
    return result_files

# Main forecasting logic
def run_forecast(
    file,
    frequency,
    eval_strategy,
    horizon,
    step_size,
    num_windows,
    use_historical_avg,
    use_naive,
    use_seasonal_naive,
    seasonality,
    use_window_avg,
    window_size,
    use_seasonal_window_avg,
    seasonal_window_size,
    use_autoets,
    use_autoarima,
    future_horizon
):
    df, message = load_data(file)
    if df is None:
        return None, None, None, None, None, None, message

    models = []
    model_aliases = []

    if use_historical_avg:
        models.append(HistoricAverage(alias='historical_average'))
        model_aliases.append('historical_average')
    if use_naive:
        models.append(Naive(alias='naive'))
        model_aliases.append('naive')
    if use_seasonal_naive:
        models.append(SeasonalNaive(season_length=seasonality, alias='seasonal_naive'))
        model_aliases.append('seasonal_naive')
    if use_window_avg:
        models.append(WindowAverage(window_size=window_size, alias='window_average'))
        model_aliases.append('window_average')
    if use_seasonal_window_avg:
        models.append(SeasonalWindowAverage(season_length=seasonality, window_size=seasonal_window_size, alias='seasonal_window_average'))
        model_aliases.append('seasonal_window_average')
    if use_autoets:
        models.append(AutoETS(alias='autoets', season_length=seasonality))
        model_aliases.append('autoets')
    if use_autoarima:
        models.append(AutoARIMA(alias='autoarima', season_length=seasonality))
        model_aliases.append('autoarima')

    if not models:
        return None, None, None, None, None, None, "Please select at least one forecasting model"

    sf = StatsForecast(models=models, freq=frequency, n_jobs=-1)

    try:
        # Run cross-validation
        if eval_strategy == "Cross Validation":
            cv_results = sf.cross_validation(df=df, h=horizon, step_size=step_size, n_windows=num_windows)
            evaluation = evaluate(df=cv_results, metrics=[bias, mae, rmse, mape], models=model_aliases)
            eval_df = pd.DataFrame(evaluation).reset_index()
            fig_validation = create_forecast_plot(cv_results, df, "Cross Validation Results")
        else:  # Fixed window
            cv_results = sf.cross_validation(df=df, h=horizon, step_size=10, n_windows=1)  # any step size for 1 window
            evaluation = evaluate(df=cv_results, metrics=[bias, mae, rmse, mape], models=model_aliases)
            eval_df = pd.DataFrame(evaluation).reset_index()
            fig_validation = create_forecast_plot(cv_results, df, "Fixed Window Validation Results")

        # Generate future forecasts
        future_forecasts = sf.forecast(df=df, h=future_horizon)
        fig_future = create_future_forecast_plot(future_forecasts, df)
        
        # Export results
        export_files = export_results(eval_df, cv_results, future_forecasts)
        
        return eval_df, cv_results, fig_validation, future_forecasts, fig_future, export_files, "Analysis completed successfully!"

    except Exception as e:
        return None, None, None, None, None, None, f"Error during forecasting: {str(e)}"

# Sample CSV file generation
def download_sample():
    sample_data = """unique_id,ds,y
^GSPC,2023-01-03,3824.139892578125
^GSPC,2023-01-04,3852.969970703125
^GSPC,2023-01-05,3808.10009765625
# rest of the sample data...
^GSPC,2024-10-31,5705.4501953125"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='')
    temp.write(sample_data)
    temp.close()
    return temp.name

# Global theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray"
)

# Gradio interface
with gr.Blocks(title="Time Series Forecasting App", theme=theme) as app:
    gr.Markdown("# 📈 Time Series Forecasting App")
    gr.Markdown("Upload a CSV with `unique_id`, `ds`, and `y` columns to apply forecasting models.")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload CSV file", file_types=[".csv"])

            download_btn = gr.Button("Download Sample Data", variant="secondary")
            download_output = gr.File(label="Click to download", visible=True)
            download_btn.click(fn=download_sample, outputs=download_output)

            with gr.Accordion("Data & Validation Settings", open=True):
                frequency = gr.Dropdown(
                    choices=[
                        ("Hourly", "H"), 
                        ("Daily", "D"), 
                        ("Weekly", "WS"), 
                        ("Monthly", "MS"), 
                        ("Quarterly", "QS"), 
                        ("Yearly", "YS")
                    ], 
                    label="Data Frequency", 
                    value="D"
                )
                
                # Evaluation Strategy
                eval_strategy = gr.Radio(
                    choices=["Fixed Window", "Cross Validation"], 
                    label="Evaluation Strategy", 
                    value="Cross Validation"
                )
                
                # Fixed Window settings
                with gr.Box(visible=True) as fixed_window_box:
                    gr.Markdown("### Fixed Window Settings")
                    horizon = gr.Slider(1, 100, value=10, step=1, label="Validation Horizon (steps ahead to predict)")
                
                # Cross Validation settings
                with gr.Box(visible=True) as cv_box:
                    gr.Markdown("### Cross Validation Settings")
                    with gr.Row():
                        step_size = gr.Slider(1, 50, value=10, step=1, label="Step Size (distance between windows)")
                        num_windows = gr.Slider(1, 20, value=5, step=1, label="Number of Windows")
                
                # Future forecast settings (always visible)
                with gr.Box():
                    gr.Markdown("### Future Forecast Settings")
                    future_horizon = gr.Slider(1, 100, value=10, step=1, label="Future Forecast Horizon (steps to predict)")

            with gr.Accordion("Model Configuration", open=True):
                gr.Markdown("## Basic Models")
                with gr.Row():
                    use_historical_avg = gr.Checkbox(label="Historical Average", value=True)
                    use_naive = gr.Checkbox(label="Naive", value=True)
                
                # Common seasonality parameter at the top level
                with gr.Box():
                    gr.Markdown("### Seasonality Configuration")
                    gr.Markdown("This seasonality period affects Seasonal Naive, Seasonal Window Average, AutoETS, and AutoARIMA models")
                    seasonality = gr.Number(label="Seasonality Period", value=5)
                
                gr.Markdown("### Seasonal Models")
                with gr.Row():
                    use_seasonal_naive = gr.Checkbox(label="Seasonal Naive", value=True)
                
                gr.Markdown("### Window-based Models")
                with gr.Row():
                    use_window_avg = gr.Checkbox(label="Window Average", value=True)
                    window_size = gr.Number(label="Window Size", value=10)
                
                with gr.Row():
                    use_seasonal_window_avg = gr.Checkbox(label="Seasonal Window Average", value=True)
                    seasonal_window_size = gr.Number(label="Seasonal Window Size", value=2)
                
                gr.Markdown("### Advanced Models (use seasonality from above)")
                with gr.Row():
                    use_autoets = gr.Checkbox(label="AutoETS (Exponential Smoothing)", value=True)
                    use_autoarima = gr.Checkbox(label="AutoARIMA", value=True)

        with gr.Column(scale=3):
            message_output = gr.Textbox(label="Status Message")
            
            with gr.Tabs() as tabs:
                with gr.TabItem("Validation Results"):
                    eval_output = gr.Dataframe(label="Evaluation Metrics")
                    validation_plot = gr.Plot(label="Validation Plot")
                    validation_output = gr.Dataframe(label="Validation Data", visible=False)
                    
                    with gr.Row():
                        show_data_btn = gr.Button("Show Validation Data")
                        hide_data_btn = gr.Button("Hide Validation Data", visible=False)
                    
                    def show_data():
                        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                    
                    def hide_data():
                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                    
                    show_data_btn.click(
                        fn=show_data,
                        outputs=[validation_output, hide_data_btn, show_data_btn]
                    )
                    
                    hide_data_btn.click(
                        fn=hide_data,
                        outputs=[validation_output, hide_data_btn, show_data_btn]
                    )
                
                with gr.TabItem("Future Forecast"):
                    forecast_plot = gr.Plot(label="Future Forecast Plot")
                    forecast_output = gr.Dataframe(label="Future Forecast Data", visible=False)
                    
                    with gr.Row():
                        show_forecast_btn = gr.Button("Show Forecast Data")
                        hide_forecast_btn = gr.Button("Hide Forecast Data", visible=False)
                    
                    def show_forecast():
                        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                    
                    def hide_forecast():
                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                    
                    show_forecast_btn.click(
                        fn=show_forecast,
                        outputs=[forecast_output, hide_forecast_btn, show_forecast_btn]
                    )
                    
                    hide_forecast_btn.click(
                        fn=hide_forecast,
                        outputs=[forecast_output, hide_forecast_btn, show_forecast_btn]
                    )
                
                with gr.TabItem("Export Results"):
                    export_files = gr.Files(label="Download Results")

    with gr.Row(visible=True) as run_row:
        submit_btn = gr.Button("Run Validation and Forecast", variant="primary", size="lg")
    
    # Update visibility of the appropriate box based on evaluation strategy
    def update_eval_boxes(strategy):
        return (gr.update(visible=strategy == "Fixed Window"), 
                gr.update(visible=strategy == "Cross Validation"))
        
    eval_strategy.change(
        fn=update_eval_boxes,
        inputs=[eval_strategy],
        outputs=[fixed_window_box, cv_box]
    )

    # Run forecast when button is clicked
    submit_btn.click(
        fn=run_forecast,
        inputs=[
            file_input, frequency, eval_strategy, horizon, step_size, num_windows,
            use_historical_avg, use_naive, use_seasonal_naive, seasonality,
            use_window_avg, window_size, use_seasonal_window_avg, seasonal_window_size,
            use_autoets, use_autoarima, future_horizon
        ],
        outputs=[eval_output, validation_output, validation_plot, forecast_output, forecast_plot, export_files, message_output]
    )

if __name__ == "__main__":
    app.launch(share=False)