# Bayesian Time Series Forecasting using Orbit-ML and Prophet

Orbit is an open-source Python package developed by Uber for Bayesian time series forecasting. It is designed to handle real-world forecasting scenarios with features like trend analysis, seasonality modeling, and uncertainty quantification. Orbit supports models like the Bayesian Structural Time Series (BSTS) and Generalized Additive Models (GAM), making it versatile for a variety of time series problems.

Orbit provides a Bayesian approach to time series forecasting that has built-in support for Uncertainty Quantification and Trend and Seasonality Decomposition. It works with univariate and multivariate time series, and the API is simple to use.

Like all Bayesian approaches, it is computationally intensive. So expect it to take longer to get your forecasts than ARIMA.

You can install Orbit using pip:

    pip install orbit-ml

Let's use the sunspot data from WDC-SILSO, Royal Observatory of Belgium, Brussels.

    from orbit.utils.dataset import load_iclaims
    from orbit.models import DLT
    from orbit.diagnostics.plot import plot_predicted_data
    from orbit.diagnostics.metrics import smape
    from prophet import Prophet

    # Define RMSE function
    def rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))

    # Load sample data
    data = pd.read_csv("SN_m_tot_V2.0.csv")
    data["Month"] = pd.to_datetime(data["Month"])

    # Create separate dataframes for Orbit and Prophet with correct column names
    orbit_data = data.copy()
    prophet_data = data.copy()

    # Rename columns for Orbit
    orbit_data = orbit_data.rename(columns={
        "Month": "date",
        "Sunspot": "response"  # Changed from "value" to "response"
    })

    # Rename columns for Prophet
    prophet_data = prophet_data.rename(columns={
        "Month": "ds",
        "Sunspot": "y"
    })

    # Split the data
    train_size = len(data) - 48

    # Orbit train/test split
    orbit_train = orbit_data.iloc[:train_size]
    orbit_test = orbit_data.iloc[train_size:]

    # Prophet train/test split
    prophet_train = prophet_data.iloc[:train_size]
    prophet_test = prophet_data.iloc[train_size:]

    # --- Orbit Model ---
    model_orbit = DLT(
        response_col="response",
        date_col="date",
        seasonality=12,
    )

    # Fit Orbit model
    model_orbit.fit(df=orbit_train)
    predictions_orbit = model_orbit.predict(df=orbit_data)

    # --- Prophet Model ---
    # Initialize and fit Prophet model
    model_prophet = Prophet(yearly_seasonality=True)
    model_prophet.fit(prophet_train)

    # Make predictions with Prophet
    future = prophet_data[['ds']]
    predictions_prophet = model_prophet.predict(future)

    # Create subplot for both models
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot Orbit results
    ax1.plot(orbit_data['date'], orbit_data['response'], label='Actual', alpha=0.5)
    ax1.plot(predictions_orbit['date'], predictions_orbit['prediction'], label='Predicted', color='red')
    ax1.fill_between(predictions_orbit['date'],
                    predictions_orbit['prediction_5'],
                    predictions_orbit['prediction_95'],
                    color='red',
                    alpha=0.1)
    ax1.axvline(x=orbit_train['date'].iloc[-1], color='black', linestyle='--', label='Train/Test Split')
    ax1.legend()
    ax1.set_title('Orbit Model Forecast')

    # Plot Prophet results
    ax2.plot(prophet_data['ds'], prophet_data['y'], label='Actual', alpha=0.5)
    ax2.plot(predictions_prophet['ds'], predictions_prophet['yhat'], label='Predicted', color='green')
    ax2.fill_between(predictions_prophet['ds'],
                    predictions_prophet['yhat_lower'],
                    predictions_prophet['yhat_upper'],
                    color='green',
                    alpha=0.1)
    ax2.axvline(x=prophet_train['ds'].iloc[-1], color='black', linestyle='--', label='Train/Test Split')
    ax2.legend()
    ax2.set_title('Prophet Model Forecast')

    plt.tight_layout()
    plt.savefig("timeseries_comparison.png")
    plt.show()

    # Calculate metrics for test set - Orbit
    test_predictions_orbit = predictions_orbit.iloc[train_size:]
    test_actual = orbit_test['response']

    print("\nOrbit Test Set Metrics:")
    print("SMAPE:", smape(test_actual, test_predictions_orbit['prediction']))
    print("RMSE:", rmse(test_actual, test_predictions_orbit['prediction']))

    # Calculate metrics for test set - Prophet
    test_predictions_prophet = predictions_prophet.iloc[train_size:]
    print("\nProphet Test Set Metrics:")
    print("SMAPE:", smape(test_actual, test_predictions_prophet['yhat']))
    print("RMSE:", rmse(test_actual, test_predictions_prophet['yhat']))

    # Show prediction intervals for both models
    print("\nOrbit Prediction Intervals:")
    print(predictions_orbit[["prediction", "prediction_5", "prediction_95"]].head())

    print("\nProphet Prediction Intervals:")
    print(predictions_prophet[["yhat", "yhat_lower", "yhat_upper"]].head())

This uses Damped Local Trend (DLT) which extends Local Linear Trend (LLT) by introducing damping for long-term trends. The seasonality is 12 since this is monthly data. Orbit provides credible intervals for each prediction, allowing you to assess the uncertainty of your forecasts.

    # Extract prediction intervals
    predictions[["prediction", "prediction_5", "prediction_95"]].head()

## Evaluation Metrics

Orbit and Prophet perform differently on the same data.

    Orbit Test Set Metrics:
    SMAPE: 1.2577330909302515
    RMSE: 99.90859640412387

    Prophet Test Set Metrics:
    SMAPE: 0.6695273556170346
    RMSE: 64.91185978677316

Clearly both models missed. Prophet did better. The cone of uncertainty for the Orbit model is huge. We should spend some more time tuning these before moving to production with these.

Orbit is a Bayesian forecasting library that is useful when understanding prediction uncertainty is important, such as demand forecasting, financial planning, and policy impact analysis. While Orbit requires more computational resources than simpler forecasting methods, its Bayesian framework and feature set make it a good choice when you are focused on quantifying uncertainty.

## Key Takeaways

- See the code examples above for a practical starting point.
