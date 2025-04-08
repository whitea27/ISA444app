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
    
    files = {}
    
    if eval_df is not None:
        eval_path = os.path.join(temp_dir, f"evaluation_metrics_{timestamp}.csv")
        eval_df.to_csv(eval_path, index=False)
        files["evaluation"] = eval_path
        
    if cv_results is not None:
        cv_path = os.path.join(temp_dir, f"cross_validation_results_{timestamp}.csv")
        cv_results.to_csv(cv_path, index=False)
        files["validation"] = cv_path
        
    if future_forecasts is not None:
        forecast_path = os.path.join(temp_dir, f"forecasts_{timestamp}.csv")
        future_forecasts.to_csv(forecast_path, index=False)
        files["forecast"] = forecast_path
    
    return files

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
        models.append(AutoETS(alias='autoets'))
        model_aliases.append('autoets')
    if use_autoarima:
        models.append(AutoARIMA(alias='autoarima'))
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
series1,2023-01-01,100
series1,2023-01-02,105
series1,2023-01-03,102
series1,2023-01-04,107
series1,2023-01-05,104
series1,2023-01-06,110
series1,2023-01-07,108
series1,2023-01-08,112
series1,2023-01-09,115
series1,2023-01-10,118
series1,2023-01-11,120
series1,2023-01-12,123
series1,2023-01-13,126
series1,2023-01-14,129
series1,2023-01-15,131
series2,2023-01-01,200
series2,2023-01-02,195
series2,2023-01-03,205
series2,2023-01-04,210
series2,2023-01-05,215
series2,2023-01-06,212
series2,2023-01-07,208
series2,2023-01-08,215
series2,2023-01-09,220
series2,2023-01-10,218
series2,2023-01-11,225
series2,2023-01-12,230
series2,2023-01-13,235
series2,2023-01-14,232
series2,2023-01-15,240
"""
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
                
                eval_strategy = gr.Radio(
                    choices=["Fixed Window", "Cross Validation"], 
                    label="Evaluation Strategy", 
                    value="Cross Validation"
                )
                
                with gr.Row():
                    horizon = gr.Slider(1, 100, value=10, step=1, label="Validation Horizon")
                    future_horizon = gr.Slider(1, 100, value=20, step=1, label="Future Forecast Horizon")
                
                # Cross validation settings will be defined after the main UI elements

            with gr.Accordion("Model Configuration", open=True):
                gr.Markdown("### Basic Models")
                with gr.Row():
                    use_historical_avg = gr.Checkbox(label="Historical Average", value=True)
                    use_naive = gr.Checkbox(label="Naive", value=True)
                
                gr.Markdown("### Seasonal Models")
                with gr.Row():
                    use_seasonal_naive = gr.Checkbox(label="Seasonal Naive")
                    seasonality = gr.Number(label="Seasonality Period", value=7)
                
                gr.Markdown("### Window-based Models")
                with gr.Row():
                    use_window_avg = gr.Checkbox(label="Window Average")
                    window_size = gr.Number(label="Window Size", value=3)
                
                with gr.Row():
                    use_seasonal_window_avg = gr.Checkbox(label="Seasonal Window Average")
                    seasonal_window_size = gr.Number(label="Seasonal Window Size", value=2)
                
                gr.Markdown("### Advanced Models")
                with gr.Row():
                    use_autoets = gr.Checkbox(label="AutoETS (Exponential Smoothing)")
                    use_autoarima = gr.Checkbox(label="AutoARIMA")

            submit_btn = gr.Button("Run Forecast", variant="primary", size="lg")

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

    # Create a special Row for cross-validation settings
    with gr.Row(visible=True) as cv_row:
        step_size = gr.Slider(1, 50, value=10, step=1, label="Step Size")
        num_windows = gr.Slider(1, 20, value=3, step=1, label="Number of Windows")
    
    # Update visibility of step_size and num_windows based on eval_strategy
    def update_cv_visibility(strategy):
        return gr.update(visible=strategy == "Cross Validation")
        
    eval_strategy.change(
        fn=update_cv_visibility,
        inputs=[eval_strategy],
        outputs=[cv_row]
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