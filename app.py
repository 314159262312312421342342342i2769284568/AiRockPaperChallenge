import os
import logging
import numpy as np
import cv2
import pickle
import base64
import json
from flask import Flask, render_template, request, jsonify, Response, session

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Training data storage - we'll save this to disk to persist between server restarts
TRAINING_DATA_FILE = 'training_data.json'

# Initialize training data and feature store
training_data = {
    'rock': [],
    'paper': [],
    'scissors': []
}

# Feature storage - persist features on disk to avoid inconsistencies
FEATURES_DIR = 'training_features'
os.makedirs(FEATURES_DIR, exist_ok=True)

# Helper function to save feature vectors to disk
def save_feature_vector(gesture, feature_vector, index):
    feature_path = os.path.join(FEATURES_DIR, f"{gesture}_{index}.npy")
    np.save(feature_path, feature_vector)
    return feature_path

# Helper function to load feature vectors from disk
def load_feature_vector(gesture, index):
    feature_path = os.path.join(FEATURES_DIR, f"{gesture}_{index}.npy")
    if os.path.exists(feature_path):
        return np.load(feature_path)
    return None

# We're changing our approach - instead of keeping features in memory,
# we'll just store counts and load features from disk when needed
try:
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'r') as f:
            counts = json.load(f)
            logger.debug(f"Loading training data counts: {counts}")
            
            # Validate and load actual feature files
            for gesture in ['rock', 'paper', 'scissors']:
                count = counts.get(gesture, 0)
                training_data[gesture] = []
                
                # Check for feature files
                for i in range(count):
                    feature_path = os.path.join(FEATURES_DIR, f"{gesture}_{i}.npy")
                    if os.path.exists(feature_path):
                        # Just add placeholder - we'll load when needed
                        training_data[gesture].append(i)  # Store index instead of feature
                    else:
                        logger.warning(f"Missing feature file: {feature_path}")
                
                # Update count based on actually found files
                counts[gesture] = len(training_data[gesture])
            
            # Save validated counts back
            with open(TRAINING_DATA_FILE, 'w') as f:
                json.dump(counts, f)
                
            logger.debug(f"Validated training data counts: {counts}")
except Exception as e:
    logger.error(f"Error loading training data counts: {str(e)}")

# Load the model if it exists
model = None
try:
    with open('hand_gesture_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
    # Check if it's the new model format (dictionary with model and scaler)
    if isinstance(model_data, dict) and 'model' in model_data:
        # Try to import the GestureClassifier for proper model loading
        try:
            from models.gesture_classifier import GestureClassifier
            classifier = GestureClassifier()
            classifier.model = model_data['model']
            classifier.scaler = model_data.get('scaler')
            classifier.class_labels = model_data.get('class_labels')
            classifier.feature_importances = model_data.get('feature_importances')
            classifier.is_trained = True
            model = classifier
            logger.debug("Enhanced gesture classifier model loaded successfully")
        except ImportError:
            # If import fails, just use the model directly
            model = model_data['model']
            logger.debug("Model loaded successfully (basic format)")
    else:
        # Old format - just the model directly
        model = model_data
        logger.debug("Legacy model format loaded successfully")
        
except FileNotFoundError:
    logger.debug("No model found, training will be required")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.debug("No model found or error loading model, training will be required")

# Game state
game_state = {
    'player_score': 0,
    'computer_score': 0,
    'rounds': 0,
    'result': ''
}

def reset_game_state():
    global game_state
    game_state = {
        'player_score': 0,
        'computer_score': 0,
        'rounds': 0,
        'result': ''
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/game')
def game():
    # Reset the game state when starting a new game
    reset_game_state()
    return render_template('game.html')

@app.route('/capture_training_image', methods=['POST'])
def capture_training_image():
    data = request.json
    image_data = data.get('image')
    gesture = data.get('gesture')
    
    if not image_data or not gesture:
        return jsonify({'success': False, 'error': 'Missing image or gesture data'})
    
    # Decode base64 image
    try:
        # Remove the base64 prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use enhanced image processing from utils module
        from utils.image_processing import preprocess_image, extract_features
        
        # Process the image with improved methods - now returns processed image and display image
        processed_img, _ = preprocess_image(img)
        
        # Extract enhanced features
        features = extract_features(processed_img)
        
        # Store feature dimensions for debugging
        logger.debug(f"Captured feature vector with shape: {features.shape if hasattr(features, 'shape') else len(features)}")
        
        # Save feature vector to a file
        index = len(training_data[gesture])
        feature_path = save_feature_vector(gesture, features, index)
        logger.debug(f"Saved feature vector to {feature_path}")
        
        # Add index to training data (not the actual feature vector)
        training_data[gesture].append(index)
        
        # Get current counts
        counts = {k: len(v) for k, v in training_data.items()}
        
        # Save counts to file for persistence
        try:
            with open(TRAINING_DATA_FILE, 'w') as f:
                json.dump(counts, f)
            logger.debug(f"Saved training data counts to {TRAINING_DATA_FILE}")
        except Exception as save_error:
            logger.error(f"Error saving training data counts: {str(save_error)}")
        
        # Log the training progress
        logger.debug(f"Added {gesture} image. Current counts: {counts}")
        
        return jsonify({
            'success': True, 
            'message': f'Captured {gesture} image with enhanced features', 
            'counts': counts
        })
    except Exception as e:
        logger.error(f"Error processing training image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    global model
    
    # Check if we have enough training data
    min_samples = min(len(training_data['rock']), 
                      len(training_data['paper']), 
                      len(training_data['scissors']))
    
    if min_samples < 10:
        return jsonify({
            'success': False, 
            'error': f'Not enough training data. Need at least 10 images per gesture. Currently have: Rock: {len(training_data["rock"])}, Paper: {len(training_data["paper"])}, Scissors: {len(training_data["scissors"])}'
        })
    
    try:
        from models.gesture_classifier import GestureClassifier
        import logging
        
        # Load feature vectors from disk for training
        rock_features = []
        for idx in training_data['rock']:
            feature = load_feature_vector('rock', idx)
            if feature is not None:
                rock_features.append(feature)
        
        paper_features = []
        for idx in training_data['paper']:
            feature = load_feature_vector('paper', idx)
            if feature is not None:
                paper_features.append(feature)
                
        scissors_features = []
        for idx in training_data['scissors']:
            feature = load_feature_vector('scissors', idx)
            if feature is not None:
                scissors_features.append(feature)
        
        # Report loaded feature counts
        logger.debug(f"Loaded feature vectors - Rock: {len(rock_features)}, Paper: {len(paper_features)}, Scissors: {len(scissors_features)}")
        
        # Check if we have enough loaded features
        if min(len(rock_features), len(paper_features), len(scissors_features)) < 10:
            return jsonify({
                'success': False, 
                'error': f'Not enough valid training data found on disk. Need at least 10 images per gesture. Currently have: Rock: {len(rock_features)}, Paper: {len(paper_features)}, Scissors: {len(scissors_features)}'
            })
        
        # Debug the feature vectors dimensions
        logger.debug(f"Feature dimensions - Rock: {[f.shape if hasattr(f, 'shape') else len(f) for f in rock_features[:3]]}")
        logger.debug(f"Feature dimensions - Paper: {[f.shape if hasattr(f, 'shape') else len(f) for f in paper_features[:3]]}")
        logger.debug(f"Feature dimensions - Scissors: {[f.shape if hasattr(f, 'shape') else len(f) for f in scissors_features[:3]]}")
        
        # Find the most common feature length to standardize
        all_lengths = []
        for features in [rock_features, paper_features, scissors_features]:
            for f in features:
                if hasattr(f, 'shape'):
                    all_lengths.append(f.shape[0])
                elif hasattr(f, '__len__'):
                    all_lengths.append(len(f))
        
        # Get mode (most common) length as our standard
        from collections import Counter
        length_counts = Counter(all_lengths)
        if length_counts:
            standard_length = length_counts.most_common(1)[0][0]
            logger.debug(f"Using standard feature length: {standard_length}")
        else:
            standard_length = 1000  # Fallback default
            logger.warning(f"No clear standard feature length, using default: {standard_length}")
        
        # Normalize all features to same length with padding/truncation
        def normalize_features(features_list, target_length):
            normalized = []
            for f in features_list:
                if hasattr(f, 'shape') and len(f.shape) > 0:
                    current_len = f.shape[0]
                    if current_len > target_length:
                        # Truncate if too long
                        normalized.append(f[:target_length])
                    elif current_len < target_length:
                        # Pad with zeros if too short
                        padded = np.zeros(target_length)
                        padded[:current_len] = f
                        normalized.append(padded)
                    else:
                        normalized.append(f)
                elif hasattr(f, '__len__'):
                    normalized.append(np.zeros(target_length))  # Use zeros for irregular data
                else:
                    normalized.append(np.zeros(target_length))  # Fallback
            return np.array(normalized)
        
        # Normalize feature vectors to same length
        rock_normalized = normalize_features(rock_features, standard_length)
        paper_normalized = normalize_features(paper_features, standard_length)
        scissors_normalized = normalize_features(scissors_features, standard_length)
        
        # Prepare training data with consistent dimensions
        X = np.vstack([
            rock_normalized,
            paper_normalized, 
            scissors_normalized
        ])
        
        y = np.hstack([
            np.full(len(rock_normalized), 'rock'),
            np.full(len(paper_normalized), 'paper'),
            np.full(len(scissors_normalized), 'scissors')
        ])
        
        # Create and train the gesture classifier
        classifier = GestureClassifier()
        training_success = classifier.train(X, y)
        
        if training_success:
            # Save the model
            classifier.save_model('hand_gesture_model.pkl')
            
            # Update global model
            model = classifier
            
            logger.info("Model trained and saved successfully")
            return jsonify({'success': True, 'message': 'Model trained successfully with improved accuracy!'})
        else:
            return jsonify({'success': False, 'error': 'Failed to train the model. Check the logs for details.'})
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    global model, game_state
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not trained yet'})
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'error': 'Missing image data'})
    
    try:
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use the enhanced image processing from utils module
        from utils.image_processing import preprocess_image, extract_features
        
        # Process the image with improved methods - now returns processed image and display image
        processed_img, _ = preprocess_image(img)
        
        # Extract enhanced features
        features = extract_features(processed_img)
        
        # Make prediction using our improved classifier
        try:
            if isinstance(model, dict) and 'model' in model:
                # Handle dictionary format
                model_obj = model.get('model')
                if model_obj is not None and hasattr(model_obj, 'predict'):
                    prediction = model_obj.predict([features])[0]
                else:
                    prediction = 'unknown'
                logger.debug(f"Predicted gesture from dict model: {prediction}")
            elif hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                # Handle GestureClassifier or direct model instance
                if hasattr(model, 'is_trained') and getattr(model, 'is_trained', False):
                    # It's our GestureClassifier
                    prediction = model.predict(features)
                else:
                    # It's a direct scikit-learn model
                    prediction = model.predict([features])[0]
                logger.debug(f"Predicted gesture: {prediction}")
            else:
                # Fallback if model isn't properly loaded
                logger.error(f"Model type not recognized for prediction: {type(model)}")
                prediction = 'unknown'
        except Exception as prediction_error:
            logger.error(f"Error during prediction: {str(prediction_error)}")
            prediction = 'unknown'  # Safe fallback
        
        # Generate computer choice
        computer_choice = np.random.choice(['rock', 'paper', 'scissors'])
        
        # Determine winner
        result = determine_winner(prediction, computer_choice)
        
        # Update game state
        if result == 'win':
            game_state['player_score'] += 1
        elif result == 'lose':
            game_state['computer_score'] += 1
        
        game_state['rounds'] += 1
        game_state['result'] = result
        
        # Log the results for debugging
        logger.debug(f"Player: {prediction}, Computer: {computer_choice}, Result: {result}")
        
        return jsonify({
            'success': True,
            'player_gesture': prediction,
            'computer_choice': computer_choice,
            'result': result,
            'game_state': game_state
        })
    except Exception as e:
        logger.error(f"Error predicting gesture: {str(e)}")
        # Log error details for debugging
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

# The preprocess_image and extract_features functions have been moved to utils/image_processing.py
# and are now imported when needed

def determine_winner(player, computer):
    """Determine winner of Rock Paper Scissors round"""
    if player == computer:
        return 'tie'
    elif (player == 'rock' and computer == 'scissors') or \
         (player == 'scissors' and computer == 'paper') or \
         (player == 'paper' and computer == 'rock'):
        return 'win'
    else:
        return 'lose'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
