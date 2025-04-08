import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tempfile

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
from utilsforecast.losses import *  # Assuming you need the metrics like bias, mae, rmse, mape

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
        df = df.sort_values(['unique_id', 'ds'])
        return df, "Data loaded successfully!"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Function to generate and return a plot for a specific cross-validation window
def create_forecast_plot(forecast_df, original_df, window=None):
    plt.figure(figsize=(10, 6))
    unique_ids = forecast_df['unique_id'].unique()
    forecast_cols = [col for col in forecast_df.columns if col not in ['unique_id', 'ds', 'cutoff']]

    if window is not None and 'cutoff' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['cutoff'] == window]

    for unique_id in unique_ids:
        original_data = original_df[original_df['unique_id'] == unique_id]
        plt.plot(original_data['ds'], original_data['y'], 'k-', label='Actual')
        forecast_data = forecast_df[forecast_df['unique_id'] == unique_id]
        for col in forecast_cols:
            if col in forecast_data.columns:
                plt.plot(forecast_data['ds'], forecast_data[col], label=col)

    plt.title(f'Forecasting Results{" (Window: " + str(window) + ")" if window else ""}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    return fig

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
    use_autoarima
):
    df, message = load_data(file)
    if df is None:
        return None, None, None, message, []

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
        return None, None, None, "Please select at least one forecasting model", []

    sf = StatsForecast(models=models, freq=frequency, n_jobs=-1)

    try:
        if eval_strategy == "Cross Validation":
            cv_results = sf.cross_validation(df=df, h=horizon, step_size=step_size, n_windows=num_windows)
            evaluation = evaluate(df=cv_results, metrics=[bias, mae, rmse, mape], models=model_aliases)
            eval_df = pd.DataFrame(evaluation).reset_index()
            unique_cutoffs = sorted(cv_results['cutoff'].unique())
            fig_forecast = create_forecast_plot(cv_results, df, window=unique_cutoffs[0])
            return eval_df, cv_results, fig_forecast, "Cross validation completed successfully!", unique_cutoffs

        else:  # Fixed window
            train_size = len(df) - horizon
            if train_size <= 0:
                return None, None, None, f"Not enough data for horizon={horizon}", []

            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            sf.fit(train_df)
            forecast = sf.predict(h=horizon)
            evaluation = evaluate(df=forecast, metrics=[bias, mae, rmse, mape], models=model_aliases)
            eval_df = pd.DataFrame(evaluation).reset_index()
            fig_forecast = create_forecast_plot(forecast, df)
            return eval_df, forecast, fig_forecast, "Fixed window evaluation completed successfully!", []

    except Exception as e:
        return None, None, None, f"Error during forecasting: {str(e)}", []

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
"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='')
    temp.write(sample_data)
    temp.close()
    return temp.name

# Gradio interface
with gr.Blocks(title="StatsForecast Demo") as app:
    gr.Markdown("# 📈 StatsForecast Demo App")
    gr.Markdown("Upload a CSV with `unique_id`, `ds`, and `y` columns to apply forecasting models.")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload CSV file", file_types=[".csv"])

            download_btn = gr.Button("Download Sample Data")
            download_output = gr.File(label="Click to download", visible=True)
            download_btn.click(fn=download_sample, outputs=download_output)

            frequency = gr.Dropdown(choices=["H", "D", "WS", "MS", "QS", "YS"], label="Frequency", value="D")
            eval_strategy = gr.Radio(choices=["Fixed Window", "Cross Validation"], label="Evaluation Strategy", value="Cross Validation")
            horizon = gr.Slider(1, 100, value=14, step=1, label="Horizon")
            step_size = gr.Slider(1, 50, value=5, step=1, label="Step Size")
            num_windows = gr.Slider(1, 20, value=3, step=1, label="Number of Windows")


            gr.Markdown("### Model Configuration")
            use_historical_avg = gr.Checkbox(label="Use Historical Average", value=True)
            use_naive = gr.Checkbox(label="Use Naive", value=True)
            use_seasonal_naive = gr.Checkbox(label="Use Seasonal Naive")
            seasonality = gr.Number(label="Seasonality", value=7)
            use_window_avg = gr.Checkbox(label="Use Window Average")
            window_size = gr.Number(label="Window Size", value=3)
            use_seasonal_window_avg = gr.Checkbox(label="Use Seasonal Window Average")
            seasonal_window_size = gr.Number(label="Seasonal Window Size", value=2)
            use_autoets = gr.Checkbox(label="Use AutoETS")
            use_autoarima = gr.Checkbox(label="Use AutoARIMA")

            submit_btn = gr.Button("Run Forecast")

        with gr.Column(scale=3):
            eval_output = gr.Dataframe(label="Evaluation Results")
            forecast_output = gr.Dataframe(label="Forecast Data")
            plot_output = gr.Plot(label="Forecast Plot")
            message_output = gr.Textbox(label="Message")
            window_selector = gr.Dropdown(label="Select Forecast Window", choices=[], visible=False)

    submit_btn.click(
        fn=run_forecast,
        inputs=[
            file_input, frequency, eval_strategy, horizon, step_size, num_windows,
            use_historical_avg, use_naive, use_seasonal_naive, seasonality,
            use_window_avg, window_size, use_seasonal_window_avg, seasonal_window_size,
            use_autoets, use_autoarima
        ],
        outputs=[eval_output, forecast_output, plot_output, message_output, window_selector]
    )

    window_selector.change(fn=create_forecast_plot, inputs=window_selector, outputs=plot_output)

if __name__ == "__main__":
    app.launch(share=False)
