from flask import Flask, render_template, jsonify, request
import json, os, pickle
import numpy as np
import logging
import logging.handlers
from datetime import datetime
from functools import wraps

# === SILENT MODE CONFIGURATION ===
SILENT_MODE = True  # Set to False for console output

# === LOGGING CONFIGURATION ===
def setup_logging():
    """Configure comprehensive logging system"""
    
    # Create logs directory at project root level (outside web_app)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    log_dir = os.path.abspath(log_dir)  # Convert to absolute path
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default Flask logging if in silent mode
    if SILENT_MODE:
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('flask').setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Format for logs
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. ALL LOGS FILE (app_all.log) - All log levels
    all_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'app_all.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(all_handler)
    
    # 2. ERROR LOGS FILE (app_errors.log) - Error and Critical only
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'app_errors.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # 3. API ACCESS LOGS (api_access.log) - API request tracking
    api_logger = logging.getLogger('api_access')
    api_logger.setLevel(logging.INFO)
    api_logger.propagate = False  # Don't send to root logger
    
    api_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'api_access.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(simple_formatter)
    api_logger.addHandler(api_handler)
    
    # 4. ML MODEL LOGS (ml_model.log) - ML operations and predictions
    ml_logger = logging.getLogger('ml_model')
    ml_logger.setLevel(logging.DEBUG)
    ml_logger.propagate = False  # Don't send to root logger
    
    ml_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'ml_model.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    ml_handler.setLevel(logging.DEBUG)
    ml_handler.setFormatter(detailed_formatter)
    ml_logger.addHandler(ml_handler)
    
    # Console handler only if not in silent mode
    if not SILENT_MODE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # Log the initialization
    root_logger.info(f"Logging system initialized - Log directory: {log_dir}")
    
    return api_logger, ml_logger, log_dir

# Initialize logging
api_logger, ml_logger, LOG_DIR = setup_logging()

# Get main application logger
app_logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress Flask startup messages in silent mode
if SILENT_MODE:
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

# === API REQUEST LOGGING DECORATOR ===
def log_api_request(f):
    """Decorator to log API requests and responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = datetime.now()
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        
        # Log request
        api_logger.info(f"REQUEST | {request.method} {request.path} | IP: {client_ip} | Args: {dict(request.args)} | Data: {request.get_json() if request.is_json else 'N/A'}")
        
        try:
            # Execute the function
            result = f(*args, **kwargs)
            
            # Log successful response
            duration = (datetime.now() - start_time).total_seconds()
            status_code = getattr(result, 'status_code', 200)
            api_logger.info(f"RESPONSE | {request.method} {request.path} | Status: {status_code} | Duration: {duration:.3f}s | IP: {client_ip}")
            
            # Log performance metrics
            if 'log_performance_metrics' in globals():
                log_performance_metrics(f.__name__, duration, {'endpoint': request.path, 'method': request.method})
            
            return result
            
        except Exception as e:
            # Log error response
            duration = (datetime.now() - start_time).total_seconds()
            api_logger.error(f"ERROR | {request.method} {request.path} | Error: {str(e)} | Duration: {duration:.3f}s | IP: {client_ip}")
            app_logger.error(f"API endpoint {request.path} failed: {str(e)}", exc_info=True)
            raise
    
    return decorated_function

# === Global Variables ===
SYMPTOM_DATABASE = {}
SYMPTOM_BATCHES = []
MEDICAL_DATA = {}
CONDITION_PATTERNS = {}
ML_MODEL = None
ML_FEATURES = []
ML_ENCODER = None
ML_SCALER = None

# === Load JSON Data ===
def load_json_file(filename):
    # Try multiple paths for JSON files
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'app_data', filename),
        os.path.join(os.path.dirname(__file__), filename),
        filename
    ]
    
    for path in possible_paths:
        try:
            app_logger.debug(f"Trying to load {filename} from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                app_logger.info(f"Successfully loaded {filename} with {len(data)} items from {path}")
                return data
        except Exception as e:
            app_logger.debug(f"Failed to load from {path}: {e}")
            continue
    
    app_logger.error(f"Could not load {filename} from any location")
    return {} if filename.endswith('.json') else []

def load_medical_data():
    global SYMPTOM_DATABASE, SYMPTOM_BATCHES, MEDICAL_DATA, CONDITION_PATTERNS
    
    app_logger.info("Starting medical data loading process...")
    
    # Create app_data directory if it doesn't exist
    app_data_dir = os.path.join(os.path.dirname(__file__), 'app_data')
    os.makedirs(app_data_dir, exist_ok=True)
    app_logger.debug(f"Ensured app_data directory exists: {app_data_dir}")
    
    SYMPTOM_DATABASE = load_json_file('symptoms.json')
    app_logger.info(f"Loaded {len(SYMPTOM_DATABASE)} symptoms")
    
    batch_data = load_json_file('symptom_batches.json')
    SYMPTOM_BATCHES[:] = [b['symptoms'] for b in batch_data if 'symptoms' in b]
    app_logger.info(f"Loaded {len(SYMPTOM_BATCHES)} symptom batches")
    
    MEDICAL_DATA['treatments'] = load_json_file('treatments.json')
    app_logger.info(f"Loaded {len(MEDICAL_DATA['treatments'])} treatments")
    
    CONDITION_PATTERNS = load_json_file('condition_patterns.json')
    app_logger.info(f"Loaded {len(CONDITION_PATTERNS)} condition patterns")
    
    success = any([SYMPTOM_DATABASE, SYMPTOM_BATCHES, MEDICAL_DATA['treatments'], CONDITION_PATTERNS])
    
    if not success:
        app_logger.error("No JSON data loaded successfully!")
        if not SILENT_MODE:
            print("[ERROR] No JSON data loaded successfully!")
    else:
        app_logger.info("Medical data loading completed successfully")
    
    return success

# === Load ML Model Safely ===
def load_ml_model():
    global ML_MODEL, ML_FEATURES, ML_ENCODER, ML_SCALER
    
    ml_logger.info("Starting ML model loading process...")
    
    # Try multiple paths for PKL file - flexible path detection
    possible_paths = [
        # If running from web_app/ directory
        os.path.join('..', 'models', 'saved_models', 'disease_classification_model.pkl'),
        # If running from root directory  
        os.path.join('models', 'saved_models', 'disease_classification_model.pkl'),
        # Legacy paths
        os.path.join('models', 'disease_classification_model.pkl'),
        'disease_classification_model.pkl'
    ]
    
    for model_path in possible_paths:
        abs_path = os.path.abspath(model_path)
        ml_logger.debug(f"Trying ML model path: {abs_path}")
        
        if os.path.exists(abs_path):
            try:
                ml_logger.info(f"Found model file at: {abs_path}")
                with open(abs_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                ml_logger.debug(f"Model package keys: {list(model_package.keys())}")
                
                ML_MODEL = model_package.get('model')
                ML_FEATURES = model_package.get('feature_names', [])
                ML_SCALER = model_package.get('scaler')
                
                # Fixed label encoder handling
                if 'label_encoder' in model_package:
                    ML_ENCODER = model_package['label_encoder']
                    ml_logger.info("Found label_encoder in model package")
                elif 'label_encoders' in model_package and 'Disease' in model_package['label_encoders']:
                    ML_ENCODER = model_package['label_encoders']['Disease']
                    ml_logger.info("Found label_encoders.Disease in model package")
                else:
                    ml_logger.warning("No label encoder found in model package")
                    ML_ENCODER = None
                
                ml_logger.info("ML model loaded successfully!")
                ml_logger.info(f"Model type: {type(ML_MODEL).__name__ if ML_MODEL else 'None'}")
                ml_logger.info(f"Feature count: {len(ML_FEATURES)}")
                ml_logger.info(f"Encoder status: {'Available' if ML_ENCODER else 'Missing'}")
                ml_logger.info(f"Scaler status: {'Available' if ML_SCALER else 'Missing'}")
                
                # Log model metadata if available
                if 'model_name' in model_package:
                    ml_logger.info(f"Model name: {model_package['model_name']}")
                if 'accuracy' in model_package:
                    ml_logger.info(f"Model accuracy: {model_package['accuracy']:.4f}")
                if 'training_date' in model_package:
                    ml_logger.info(f"Training date: {model_package['training_date']}")
                
                if not SILENT_MODE:
                    print(f"[SUCCESS] ML model loaded from: {abs_path}")
                
                return True
                
            except Exception as e:
                ml_logger.error(f"Failed to load model from {abs_path}: {e}", exc_info=True)
                continue
        else:
            ml_logger.debug(f"Model file not found: {abs_path}")
    
    ml_logger.error("Could not load ML model from any path!")
    if not SILENT_MODE:
        print("[ERROR] Could not load ML model!")
    ML_MODEL = None
    return False

# Initialize data loading
app_logger.info("Application startup initiated")
json_loaded = load_medical_data()
model_loaded = load_ml_model()

app_logger.info(f"Initialization complete - JSON Data: {'Loaded' if json_loaded else 'Failed'}, ML Model: {'Loaded' if model_loaded else 'Failed'}")

if not SILENT_MODE:
    print(f"\n[SYSTEM STATUS]")
    print(f"JSON Data: {'Loaded' if json_loaded else 'Failed'}")
    print(f"ML Model: {'Loaded' if model_loaded else 'Failed'}")

# === Helper: Transform Symptoms to Input Vector ===
def symptoms_to_vector(symptom_ids):
    if not ML_FEATURES:
        ml_logger.error("No ML_FEATURES available for vector conversion")
        return None
    
    ml_logger.debug(f"Converting symptoms to vector: {symptom_ids}")
    
    x = np.zeros(len(ML_FEATURES))
    matches_found = 0
    
    for i, fname in enumerate(ML_FEATURES):
        if fname.endswith('_present'):
            symptom_name = fname.replace('_present', '').replace('Symptom_', '')
            if symptom_name in symptom_ids:
                x[i] = 1
                matches_found += 1
    
    ml_logger.debug(f"Vector conversion complete: {matches_found} symptom matches found out of {len(symptom_ids)} input symptoms")
    
    if matches_found == 0:
        ml_logger.warning("No symptom matches found during vector conversion")
    
    return x.reshape(1, -1)

# === Core Endpoint: Analyze Symptoms ===
@app.route('/api/analyze', methods=['POST'])
@log_api_request
def analyze():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])

    app_logger.info(f"Analyze request received with {len(selected_symptoms)} symptoms")
    ml_logger.info(f"Starting analysis for symptoms: {selected_symptoms}")

    if not selected_symptoms:
        app_logger.warning("Analyze request received with no symptoms")
        return jsonify({"success": False, "message": "No symptoms provided."}), 400

    conditions = []

    # === Try ML Model First ===
    try:
        if ML_MODEL and ML_FEATURES and ML_ENCODER:
            ml_logger.info("Attempting ML prediction...")
            x_input = symptoms_to_vector(selected_symptoms)
            
            if x_input is not None:
                # Apply scaling if available
                if ML_SCALER:
                    x_input = ML_SCALER.transform(x_input)
                    ml_logger.debug("Applied feature scaling to input vector")
                
                prediction_start = datetime.now()
                probs = ML_MODEL.predict_proba(x_input)[0]
                prediction_time = (datetime.now() - prediction_start).total_seconds()
                
                top_idx = np.argsort(probs)[::-1][:5]  # Top 5
                
                ml_logger.info(f"ML prediction completed in {prediction_time:.3f}s, found {len(top_idx)} results")
                
                for idx in top_idx:
                    condition = ML_ENCODER.inverse_transform([idx])[0]
                    confidence = round(float(probs[idx]) * 100, 1)
                    
                    ml_logger.debug(f"ML prediction: {condition} - {confidence}%")
                    
                    # Check if we have treatment info for this condition
                    if condition in MEDICAL_DATA.get('treatments', {}):
                        conditions.append({
                            "id": condition,
                            "name": condition.replace('_', ' ').title(),
                            "match_percentage": confidence,
                            "matched_symptoms": len(selected_symptoms),
                            "confidence": 'High' if confidence > 70 else 'Medium' if confidence > 40 else 'Low'
                        })
                
                ml_logger.info(f"ML model returned {len(conditions)} valid conditions with treatment data")
            else:
                ml_logger.error("Failed to create input vector from symptoms")
        else:
            missing_components = []
            if not ML_MODEL: missing_components.append("model")
            if not ML_FEATURES: missing_components.append("features")
            if not ML_ENCODER: missing_components.append("encoder")
            
            ml_logger.warning(f"ML model not fully available - missing: {', '.join(missing_components)}")
            
    except Exception as e:
        ml_logger.error(f"ML prediction failed: {e}", exc_info=True)
        app_logger.error(f"ML prediction error in analyze endpoint: {e}")

    # === Fallback to Rule-Based Matching ===
    if not conditions and CONDITION_PATTERNS:
        app_logger.info("Using rule-based fallback for symptom analysis")
        ml_logger.info("Falling back to rule-based pattern matching")
        
        rule_matches = 0
        for condition, symptoms in CONDITION_PATTERNS.items():
            match = len(set(selected_symptoms) & set(symptoms))
            if match > 0:
                pct = (match / len(symptoms)) * 100
                if condition in MEDICAL_DATA.get('treatments', {}):
                    conditions.append({
                        "id": condition,
                        "name": condition.replace('_', ' ').title(),
                        "match_percentage": round(pct, 1),
                        "matched_symptoms": match,
                        "confidence": 'High' if pct > 70 else 'Medium' if pct > 40 else 'Low'
                    })
                    ml_logger.debug(f"Rule-based match: {condition} - {pct}%")
                    rule_matches += 1
        
        conditions = sorted(conditions, key=lambda c: c['match_percentage'], reverse=True)[:5]
        ml_logger.info(f"Rule-based matching found {rule_matches} potential conditions, returning top 5")

    # === Resolve Symptom Names ===
    resolved = []
    for s in selected_symptoms:
        if s in SYMPTOM_DATABASE:
            resolved.append(SYMPTOM_DATABASE[s]['name'])
        else:
            resolved.append(s)  # Fallback to original text
            app_logger.debug(f"Unknown symptom ID: {s}, using as-is")

    app_logger.info(f"Analysis complete: {len(conditions)} conditions found for {len(selected_symptoms)} symptoms")
    ml_logger.info(f"Final analysis result: {len(conditions)} conditions returned")

    return jsonify({
        "success": True,
        "conditions": conditions,
        "analyzed_symptoms": resolved,
        "total_symptoms_analyzed": len(selected_symptoms)
    })

# === Enhanced Symptom Suggestions ===
@app.route('/api/suggest-symptoms/<query>')
@log_api_request
def suggest_symptoms(query):
    app_logger.debug(f"Symptom suggestion request for query: '{query}'")
    
    suggestions = []
    query_lower = query.lower()
    
    for symptom_id, symptom_data in SYMPTOM_DATABASE.items():
        # Check if query matches symptom name or keywords
        name = symptom_data['name'].lower()
        keywords = symptom_data.get('keywords', [])
        
        if (query_lower in name or 
            name.startswith(query_lower) or
            any(query_lower in keyword.lower() for keyword in keywords)):
            
            suggestions.append({
                'id': symptom_id,
                'name': symptom_data['name'],
                'category': symptom_data.get('category', 'general'),
                'icon': symptom_data.get('icon', 'medical'),
                'match_type': 'direct_match' if query_lower in name else 'keyword_match'
            })
    
    # Sort by relevance (direct matches first)
    suggestions.sort(key=lambda x: (x['match_type'] != 'direct_match', x['name']))
    
    app_logger.info(f"Symptom suggestion: '{query}' returned {len(suggestions)} suggestions (showing top 10)")
    
    return jsonify({
        "success": True,
        "suggestions": suggestions[:10],  # Limit to 10
        "total_count": len(suggestions)
    })

# === Enhanced Symptom Info ===
@app.route('/api/symptom-info/<symptom_id>')
@log_api_request
def get_symptom_info(symptom_id):
    app_logger.debug(f"Symptom info request for ID: {symptom_id}")
    
    if symptom_id in SYMPTOM_DATABASE:
        app_logger.debug(f"Found symptom info for: {symptom_id}")
        return jsonify({
            "success": True,
            "symptom": SYMPTOM_DATABASE[symptom_id] | {"id": symptom_id}
        })
    else:
        # Fallback for unknown symptoms
        app_logger.warning(f"Unknown symptom ID requested: {symptom_id}, providing fallback")
        return jsonify({
            "success": True,
            "symptom": {
                "id": symptom_id,
                "name": symptom_id.replace('_', ' ').title(),
                "category": "general",
                "icon": "medical",
                "severity": "unknown"
            }
        })

# === Existing Routes ===
@app.route('/')
def index():
    app_logger.info("Main page accessed")
    return render_template('index.html')

@app.route('/api/symptoms')
@log_api_request
def get_symptoms():
    app_logger.debug("All symptoms requested")
    return jsonify({
        "success": True,
        "symptoms": [
            {"id": sid, **sdata} for sid, sdata in SYMPTOM_DATABASE.items()
        ],
        "total_count": len(SYMPTOM_DATABASE)
    })

@app.route('/api/symptom-batches/<int:batch_number>')
@log_api_request
def get_symptom_batch(batch_number):
    app_logger.debug(f"Symptom batch {batch_number} requested")
    
    i = batch_number - 1
    if 0 <= i < len(SYMPTOM_BATCHES):
        batch = SYMPTOM_BATCHES[i]
        app_logger.debug(f"Returning symptom batch {batch_number} with {len(batch)} symptoms")
        return jsonify({
            "success": True,
            "batch_number": batch_number,
            "has_more_batches": batch_number < len(SYMPTOM_BATCHES),
            "batch_info": {
                "name": f"Symptom Set {batch_number}",
                "symptoms": [SYMPTOM_DATABASE[sid] | {"id": sid} for sid in batch if sid in SYMPTOM_DATABASE]
            }
        })
    
    app_logger.warning(f"Invalid symptom batch requested: {batch_number}")
    return jsonify({"success": False, "message": "Batch not found"}), 404

@app.route('/api/treatment/<condition>')
@log_api_request
def get_treatment(condition):
    app_logger.debug(f"Treatment info requested for condition: {condition}")
    
    t = MEDICAL_DATA.get('treatments', {}).get(condition)
    if t:
        app_logger.debug(f"Found treatment info for: {condition}")
        return jsonify({"success": True, "treatment": t})
    
    app_logger.warning(f"No treatment found for condition: {condition}")
    return jsonify({"success": False, "message": "Treatment not found"}), 404

# === Debug Endpoint ===
@app.route('/api/debug')
@log_api_request
def debug_status():
    app_logger.info("Debug status requested")
    
    debug_info = {
        "json_data_loaded": len(SYMPTOM_DATABASE) > 0,
        "symptom_count": len(SYMPTOM_DATABASE),
        "batch_count": len(SYMPTOM_BATCHES),
        "treatment_count": len(MEDICAL_DATA.get('treatments', {})),
        "condition_patterns": len(CONDITION_PATTERNS),
        "ml_model_loaded": ML_MODEL is not None,
        "ml_features": len(ML_FEATURES),
        "ml_encoder": ML_ENCODER is not None,
        "ml_scaler": ML_SCALER is not None,
        "silent_mode": SILENT_MODE,
        "server_time": datetime.now().isoformat()
    }
    
    app_logger.info(f"Debug info returned: {debug_info}")
    
    return jsonify(debug_info)

# === Error Handlers ===
@app.errorhandler(404)
def not_found(error):
    app_logger.warning(f"404 error: {request.url}")
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    app_logger.error(f"500 error: {str(error)}", exc_info=True)
    return jsonify({"success": False, "message": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app_logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"success": False, "message": "An unexpected error occurred"}), 500

# === Health Check Endpoint ===
@app.route('/api/health')
@log_api_request
def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "symptom_database": len(SYMPTOM_DATABASE) > 0,
            "ml_model": ML_MODEL is not None,
            "treatments": len(MEDICAL_DATA.get('treatments', {})) > 0
        },
        "version": "1.0.0",
        "silent_mode": SILENT_MODE
    }
    
    # Check if all critical components are working
    all_healthy = all(health_status["components"].values())
    
    if not all_healthy:
        health_status["status"] = "degraded"
        app_logger.warning("Health check failed - some components not available")
    
    return jsonify(health_status), 200 if all_healthy else 503

def log_startup_status():
    """Log detailed startup status"""
    app_logger.info("=== APPLICATION STARTUP STATUS ===")
    app_logger.info(f"Silent Mode: {SILENT_MODE}")
    app_logger.info(f"Log Directory: {LOG_DIR}")
    app_logger.info(f"Symptoms Database: {len(SYMPTOM_DATABASE)} items")
    app_logger.info(f"Symptom Batches: {len(SYMPTOM_BATCHES)} batches")
    app_logger.info(f"Treatments: {len(MEDICAL_DATA.get('treatments', {}))} treatments")
    app_logger.info(f"Condition Patterns: {len(CONDITION_PATTERNS)} patterns")
    app_logger.info(f"ML Model: {'Loaded' if ML_MODEL else 'Missing'}")
    app_logger.info(f"Working Directory: {os.getcwd()}")
    
    if not any([SYMPTOM_DATABASE, SYMPTOM_BATCHES, MEDICAL_DATA.get('treatments')]):
        app_logger.error("Critical: No JSON data loaded! Application may not function properly")
        if not SILENT_MODE:
            print("\n[CRITICAL ERROR] No JSON data loaded! Create 'app_data' folder with JSON files")
    
    ml_logger.info("=== ML MODEL STATUS ===")
    ml_logger.info(f"Model Loaded: {'Yes' if ML_MODEL else 'No'}")
    ml_logger.info(f"Features Available: {len(ML_FEATURES)}")
    ml_logger.info(f"Label Encoder: {'Available' if ML_ENCODER else 'Missing'}")
    ml_logger.info(f"Feature Scaler: {'Available' if ML_SCALER else 'Missing'}")

# === Request Middleware for Additional Logging ===
@app.before_request
def before_request():
    """Log request start and setup request context"""
    request.start_time = datetime.now()
    
    # Log non-static requests
    if not request.path.startswith('/static'):
        app_logger.debug(f"Request started: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Log request completion with timing"""
    if hasattr(request, 'start_time') and not request.path.startswith('/static'):
        duration = (datetime.now() - request.start_time).total_seconds()
        app_logger.debug(f"Request completed: {request.method} {request.path} - {response.status_code} in {duration:.3f}s")
    
    return response

if __name__ == '__main__':
    try:
        # Log startup status
        log_startup_status()
        
        if not SILENT_MODE:
            print(f"\n[STARTUP STATUS]")
            print(f"Symptoms Database: {len(SYMPTOM_DATABASE)} items")
            print(f"Symptom Batches: {len(SYMPTOM_BATCHES)} batches")
            print(f"Treatments: {len(MEDICAL_DATA.get('treatments', {}))} treatments")
            print(f"ML Model: {'Loaded' if ML_MODEL else 'Missing'}")
            print(f"\nStarting Flask server...")
            print(f"Silent Mode: {'ON' if SILENT_MODE else 'OFF'}")
            print(f"Logs Directory: {LOG_DIR}")
            print(f"Access at: http://localhost:5000")
        
        app_logger.info("Starting Flask development server...")
        
        # Configure Flask app for silent mode
        if SILENT_MODE:
            # Completely suppress Flask output
            import sys
            from io import StringIO
            
            # Capture Flask startup messages
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            try:
                app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
            finally:
                # Restore output streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            app.run(debug=True, host='0.0.0.0', port=5000)
            
    except KeyboardInterrupt:
        app_logger.info("Application shutdown requested by user (Ctrl+C)")
        if not SILENT_MODE:
            print("\n[INFO] Application shutdown requested by user")
            
    except Exception as e:
        app_logger.critical(f"Critical error during application startup: {str(e)}", exc_info=True)
        # Always show critical errors, even in silent mode
        print(f"\n[CRITICAL ERROR] Application failed to start: {str(e)}")
        print(f"Check {LOG_DIR}/app_errors.log for detailed error information")
        
    finally:
        app_logger.info("Application shutdown complete")
        if not SILENT_MODE:
            print("[INFO] Application shutdown complete")