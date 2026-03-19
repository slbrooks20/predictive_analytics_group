import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve

# Set random seed
np.random.seed(42)

def log_message(message):
    """Log message to experiment log file"""
    with open('outputs/benchmark/experiment_log.txt', 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

def create_cyclical_features(df):
    """Create cyclical features for hour and month"""
    df = df.copy()
    df['sin_hour'] = np.sin(2 * np.pi * df['hr'] / 23)
    df['cos_hour'] = np.cos(2 * np.pi * df['hr'] / 23)
    df['sin_month'] = np.sin(2 * np.pi * df['mnth'] / 11)
    df['cos_month'] = np.cos(2 * np.pi * df['mnth'] / 11)
    return df

def main():
    # Create output directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/docs', exist_ok=True)
    os.makedirs('outputs/benchmark', exist_ok=True)
    
    log_message("Starting model improvement pipeline")
    
    # Load data splits
    log_message("Loading data splits")
    train_df = pd.read_csv('outputs/train.csv')
    val_df = pd.read_csv('outputs/val.csv')
    test_df = pd.read_csv('outputs/test.csv')
    
    # Define feature sets
    F0_features = ['hr', 'weekday', 'workingday', 'season', 'mnth', 'yr', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    # Create F1 features (F0 + cyclical)
    train_df = create_cyclical_features(train_df)
    val_df = create_cyclical_features(val_df)
    test_df = create_cyclical_features(test_df)
    
    F1_features = F0_features + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    
    log_message(f"F0 features: {len(F0_features)}, F1 features: {len(F1_features)}")
    print(f"F0 features: {len(F0_features)}, F1 features: {len(F1_features)}")
    
    # 1. Model comparison
    log_message("Running model comparison")
    
    all_results = []
    
    # Define models to compare
    models = {
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'MLP': MLPRegressor(random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1)
    }
    
    # Evaluate each model on both feature sets
    for model_name, model in models.items():
        for feature_set_name, features in [('F0', F0_features), ('F1', F1_features)]:
            log_message(f"Evaluating {model_name} with {feature_set_name}")
            
            # Fit scaler on train only
            scaler = StandardScaler()
            X_train = train_df[features]
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train_scaled, train_df['cnt'])
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate on validation
            X_val = val_df[features]
            X_val_scaled = scaler.transform(X_val)
            val_pred = model.predict(X_val_scaled)
            
            mae = mean_absolute_error(val_df['cnt'], val_pred)
            rmse = np.sqrt(mean_squared_error(val_df['cnt'], val_pred))
            
            all_results.append({
                'model': model_name,
                'feature_set': feature_set_name,
                'split': 'validation',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': training_time
            })
            
            log_message(f"{model_name} {feature_set_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Time: {training_time:.2f}s")
            print(f"{model_name} {feature_set_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Time: {training_time:.2f}s")
    
    # Save all results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv('outputs/metrics/all_results.csv', index=False)
    log_message("Saved all model results to outputs/metrics/all_results.csv")
    print("Saved all model results to outputs/metrics/all_results.csv")
    
    # 2. Tuning - simplified version
    log_message("Starting model tuning")
    
    tuning_results = []
    
    # Ridge tuning - just test a few alphas
    log_message("Tuning Ridge")
    alphas = [0.01, 0.1, 1, 10]
    for alpha in alphas:
        scaler = StandardScaler()
        X_train = train_df[F1_features]
        X_train_scaled = scaler.fit_transform(X_train)
        
        start_time = datetime.now()
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, train_df['cnt'])
        training_time = (datetime.now() - start_time).total_seconds()
        
        X_val = val_df[F1_features]
        X_val_scaled = scaler.transform(X_val)
        val_pred = model.predict(X_val_scaled)
        
        mae = mean_absolute_error(val_df['cnt'], val_pred)
        rmse = np.sqrt(mean_squared_error(val_df['cnt'], val_pred))
        
        tuning_results.append({
            'model': 'Ridge',
            'param': f'alpha={alpha}',
            'MAE': mae,
            'RMSE': rmse,
            'training_time_seconds': training_time
        })
    
    # Random Forest tuning - limited combinations
    log_message("Tuning Random Forest")
    n_estimators_list = [100, 200]
    max_depth_list = [None, 10]
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            scaler = StandardScaler()
            X_train = train_df[F1_features]
            X_train_scaled = scaler.fit_transform(X_train)
            
            start_time = datetime.now()
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42, 
                n_jobs=1
            )
            model.fit(X_train_scaled, train_df['cnt'])
            training_time = (datetime.now() - start_time).total_seconds()
            
            X_val = val_df[F1_features]
            X_val_scaled = scaler.transform(X_val)
            val_pred = model.predict(X_val_scaled)
            
            mae = mean_absolute_error(val_df['cnt'], val_pred)
            rmse = np.sqrt(mean_squared_error(val_df['cnt'], val_pred))
            
            tuning_results.append({
                'model': 'RandomForest',
                'param': f'n_estimators={n_estimators}, max_depth={max_depth}',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': training_time
            })
    
    # Gradient Boosting tuning - limited combinations
    log_message("Tuning Gradient Boosting")
    n_estimators_list = [100, 200]
    learning_rate_list = [0.05, 0.1]
    
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            scaler = StandardScaler()
            X_train = train_df[F1_features]
            X_train_scaled = scaler.fit_transform(X_train)
            
            start_time = datetime.now()
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X_train_scaled, train_df['cnt'])
            training_time = (datetime.now() - start_time).total_seconds()
            
            X_val = val_df[F1_features]
            X_val_scaled = scaler.transform(X_val)
            val_pred = model.predict(X_val_scaled)
            
            mae = mean_absolute_error(val_df['cnt'], val_pred)
            rmse = np.sqrt(mean_squared_error(val_df['cnt'], val_pred))
            
            tuning_results.append({
                'model': 'GradientBoosting',
                'param': f'n_estimators={n_estimators}, learning_rate={learning_rate}',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': training_time
            })
    
    # MLP tuning - limited combinations
    log_message("Tuning MLP")
    hidden_layer_sizes_list = [(50,), (100,)]
    alpha_list = [0.001, 0.01]
    
    for hidden_layer_sizes in hidden_layer_sizes_list:
        for alpha in alpha_list:
            scaler = StandardScaler()
            X_train = train_df[F1_features]
            X_train_scaled = scaler.fit_transform(X_train)
            
            start_time = datetime.now()
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                random_state=42,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            model.fit(X_train_scaled, train_df['cnt'])
            training_time = (datetime.now() - start_time).total_seconds()
            
            X_val = val_df[F1_features]
            X_val_scaled = scaler.transform(X_val)
            val_pred = model.predict(X_val_scaled)
            
            mae = mean_absolute_error(val_df['cnt'], val_pred)
            rmse = np.sqrt(mean_squared_error(val_df['cnt'], val_pred))
            
            tuning_results.append({
                'model': 'MLP',
                'param': f'hidden_layers={hidden_layer_sizes}, alpha={alpha}',
                'MAE': mae,
                'RMSE': rmse,
                'training_time_seconds': training_time
            })
    
    # Save tuning results
    tuning_results_df = pd.DataFrame(tuning_results)
    tuning_results_df.to_csv('outputs/metrics/tuning_results.csv', index=False)
    log_message("Saved tuning results to outputs/metrics/tuning_results.csv")
    print("Saved tuning results to outputs/metrics/tuning_results.csv")
    
    # Gradient Boosting validation curve for n_estimators
    log_message("Creating validation curve for Gradient Boosting")
    
    scaler = StandardScaler()
    X_train = train_df[F1_features]
    X_train_scaled = scaler.fit_transform(X_train)
    X_val = val_df[F1_features]
    X_val_scaled = scaler.transform(X_val)
    
    param_range = np.arange(20, 220, 20)
    train_scores, val_scores = validation_curve(
        GradientBoostingRegressor(learning_rate=0.1, random_state=42),
        X_train_scaled, train_df['cnt'],
        param_name='n_estimators',
        param_range=param_range,
        cv=3,
        scoring='neg_mean_absolute_error'
    )
    
    train_scores = -train_scores.mean(axis=1)
    val_scores = -val_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, label='Training MAE', linewidth=2)
    plt.plot(param_range, val_scores, label='Validation MAE', linewidth=2)
    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Validation Curve: Gradient Boosting MAE vs n_estimators', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/validation_curve_gb.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved validation_curve_gb.png")
    print("Saved validation_curve_gb.png")
    
    # 3. Select best model based on lowest validation MAE
    log_message("Selecting best model")
    
    # Combine all results and tuning results
    all_combined = pd.concat([all_results_df, tuning_results_df])
    best_idx = all_combined['MAE'].idxmin()
    best_model_name = str(all_combined.loc[best_idx, 'model'])
    best_feature_set = str(all_combined.loc[best_idx, 'feature_set'])
    best_mae = float(all_combined.loc[best_idx, 'MAE'])
    
    log_message(f"Best model: {best_model_name} with {best_feature_set} - MAE: {best_mae:.2f}")
    print(f"Best model: {best_model_name} with {best_feature_set} - MAE: {best_mae:.2f}")
    
    # Train best model on full training set
    log_message("Training best model on full training set")
    
    features = F1_features if best_feature_set == 'F1' else F0_features
    
    scaler = StandardScaler()
    X_train = train_df[features]
    X_train_scaled = scaler.fit_transform(X_train)
    
    if best_model_name == 'Ridge':
        # Use best alpha from tuning
        best_alpha = tuning_results_df[
            (tuning_results_df['model'] == 'Ridge')
        ]['MAE'].min()
        best_model = Ridge(alpha=best_alpha, random_state=42)
    elif best_model_name == 'RandomForest':
        # Use best params from tuning
        best_rf_params = tuning_results_df[
            (tuning_results_df['model'] == 'RandomForest')
        ].sort_values('MAE').iloc[0]
        best_model = RandomForestRegressor(
            n_estimators=int(best_rf_params['param'].split(',')[0].split('=')[1]),
            max_depth=eval(best_rf_params['param'].split(',')[1].split('=')[1]),
            random_state=42,
            n_jobs=1
        )
    elif best_model_name == 'GradientBoosting':
        # Use best params from tuning
        best_gb_params = tuning_results_df[
            (tuning_results_df['model'] == 'GradientBoosting')
        ].sort_values('MAE').iloc[0]
        params = best_gb_params['param'].split(', ')
        n_estimators = int(params[0].split('=')[1])
        learning_rate = float(params[1].split('=')[1])
        best_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    elif best_model_name == 'MLP':
        # Use best params from tuning
        best_mlp_params = tuning_results_df[
            (tuning_results_df['model'] == 'MLP')
        ].sort_values('MAE').iloc[0]
        params = best_mlp_params['param'].split(', ')
        hidden_layers = eval(params[0].split('=')[1])
        alpha = float(params[1].split('=')[1])
        best_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            alpha=alpha,
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    
    best_model.fit(X_train_scaled, train_df['cnt'])
    
    # Save best model
    import joblib
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'model_name': best_model_name,
        'feature_set': best_feature_set
    }, 'outputs/models/final_model.pkl')
    log_message("Saved final model to outputs/models/final_model.pkl")
    print("Saved final model to outputs/models/final_model.pkl")
    
    # 4. Final evaluation on TEST set
    log_message("Evaluating best model on test set")
    
    X_test = test_df[features]
    X_test_scaled = scaler.transform(X_test)
    test_pred = best_model.predict(X_test_scaled)
    
    test_mae = mean_absolute_error(test_df['cnt'], test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_df['cnt'], test_pred))
    
    final_results = pd.DataFrame([{
        'model': best_model_name,
        'feature_set': best_feature_set,
        'split': 'test',
        'MAE': test_mae,
        'RMSE': test_rmse
    }])
    final_results.to_csv('outputs/metrics/final_model_results.csv', index=False)
    log_message(f"Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    
    # Generate diagnostic plots
    log_message("Generating diagnostic plots")
    
    residuals = test_df['cnt'] - test_pred
    
    # Residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Distribution (Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved residual_distribution.png")
    print("Saved residual_distribution.png")
    
    # MAE by hour
    test_df['pred'] = test_pred
    test_df['mae'] = np.abs(residuals)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=test_df, x='hr', y='mae', ci=None, alpha=0.7)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE by Hour (Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/mae_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved mae_by_hour.png")
    print("Saved mae_by_hour.png")
    
    # MAE by weekday
    plt.figure(figsize=(10, 6))
    sns.barplot(data=test_df, x='weekday', y='mae', ci=None, alpha=0.7)
    plt.xlabel('Weekday (0=Sunday, 6=Saturday)', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE by Weekday (Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/mae_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved mae_by_weekday.png")
    print("Saved mae_by_weekday.png")
    
    # Residual vs temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=test_df, x='temp', y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Temperature (normalized)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs Temperature (Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/figures/residual_vs_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved residual_vs_temperature.png")
    print("Saved residual_vs_temperature.png")
    
    # Rolling MAE over time
    test_df['datetime'] = pd.to_datetime(test_df['dteday'])
    test_df = test_df.sort_values('datetime')
    test_df['rolling_mae'] = test_df['mae'].rolling(window=24, min_periods=1).mean()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=test_df, x='datetime', y='rolling_mae', ci=None, alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rolling MAE (24h window)', fontsize=12)
    plt.title('Rolling MAE Over Time (Test Set)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/figures/rolling_mae_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_message("Saved rolling_mae_over_time.png")
    print("Saved rolling_mae_over_time.png")
    
    log_message("Model improvement pipeline completed successfully")
    print("\nModel improvement pipeline completed successfully")

if __name__ == "__main__":
    main()
