import flask
import numpy as np
import pandas as pd
import joblib
import time
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

app = flask.Flask(__name__)

try:
    models = {
        'dt': {
            'standard': joblib.load('models/Decision_Tree_standard.pkl'),
            'tuned': joblib.load('models/Decision_Tree_tuned.pkl'),
            'pso': joblib.load('models/Decision_Tree_PSO.pkl')
        },
        'rf': {
            'standard': joblib.load('models/Random_Forest_standard.pkl'),
            'tuned': joblib.load('models/Random_Forest_tuned.pkl'),
            'pso': joblib.load('models/Random_Forest_PSO.pkl')
        },
        'xgb': {
            'standard': joblib.load('models/XGBoost_standard.pkl'),
            'tuned': joblib.load('models/XGBoost_tuned.pkl'),
            'pso': joblib.load('models/XGBoost_PSO.pkl')
        }
    }
    scaler = joblib.load('models/scaler.pkl')
    
    data = pd.read_excel('Concrete_Data.xls')
    data.columns = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer','CoarseAggregate', 'FineAggregate', 'Age', 'ConcreteStrength']
    data = data.dropna()
    X = data.drop('ConcreteStrength', axis = 1)
    y = data['ConcreteStrength']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explainers = {
        'dt': {
            'standard': shap.TreeExplainer(models['dt']['standard']),
            'tuned': shap.TreeExplainer(models['dt']['tuned']),
            'pso': shap.TreeExplainer(models['dt']['pso'])
        },
        'rf': {
            'standard': shap.TreeExplainer(models['rf']['standard']),
            'tuned': shap.TreeExplainer(models['rf']['tuned']),
            'pso': shap.TreeExplainer(models['rf']['pso'])
        },
        'xgb': {
            'standard': shap.TreeExplainer(models['xgb']['standard']),
            'tuned': shap.TreeExplainer(models['xgb']['tuned']),
            'pso': shap.TreeExplainer(models['xgb']['pso'])
        }
    }

except Exception as e:
    #print(f"FATAL ERROR during startup: {e}")
    exit()

VALIDATION_RANGES = {
    'Cement': (100, 550), 'Slag': (0, 360), 'FlyAsh': (0, 200),
    'Water': (120, 240), 'Superplasticizer': (0, 32),
    'CoarseAggregate': (800, 1150), 'FineAggregate': (590, 1000),
    'Age': (1, 365)
}

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = flask.request.form
        model_type = form_data.get('model_type')
        tuning_method = form_data.get('tuning_method')
        
        for key, (min_val, max_val) in VALIDATION_RANGES.items():
            value = float(form_data.get(key))
            if not (min_val <= value <= max_val):
                error_msg = f"Error: Input for '{key}' is out of range ({min_val} to {max_val})."
                return flask.render_template('index.html', error=error_msg, result={'form_data': form_data})
        
        input_features = [float(form_data.get(key)) for key in X.columns]
        features_np = np.array([input_features])
        features_scaled = scaler.transform(features_np)

        model_to_use = models[model_type][tuning_method]
        
        start_time = time.time()
        prediction = model_to_use.predict(features_scaled)
        end_time = time.time()
        runtime = (end_time - start_time) * 1000
        output_strength = prediction[0]

        test_predictions = model_to_use.predict(X_test_scaled)
        live_r2 = r2_score(y_test, test_predictions)
        live_mse = mean_squared_error(y_test, test_predictions)

        importances = model_to_use.feature_importances_
        feature_importance_dict = {name: float(imp) for name, imp in zip(X.columns, importances)}

        explainer_to_use = explainers[model_type][tuning_method]
        shap_values = explainer_to_use.shap_values(features_scaled)
        shap_values_dict = {name: float(val) for name, val in zip(X.columns, shap_values[0])}

        result_data = {
            "prediction_strength": f"{output_strength:.2f} MPa",
            "r2_score": f"{live_r2:.4f}",
            "mse": f"{live_mse:.2f}",
            "runtime": f"{runtime:.2f} ms",
            "form_data": form_data,
            "feature_importances": feature_importance_dict,
            "shap_values": shap_values_dict
        }
        
        return flask.render_template('index.html', result = result_data)

    except Exception as e:
        return flask.render_template("index.html", error = f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

