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

# Initialize training data
training_data = {
    'rock': [],
    'paper': [],
    'scissors': []
}

# Load saved training data if it exists
try:
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'r') as f:
            counts = json.load(f)
            logger.debug(f"Loading training data counts: {counts}")
            if 'rock' in counts and counts['rock'] > 0:
                training_data['rock'] = [np.zeros(128) for _ in range(counts['rock'])]  # Placeholder
            if 'paper' in counts and counts['paper'] > 0:
                training_data['paper'] = [np.zeros(128) for _ in range(counts['paper'])]  # Placeholder
            if 'scissors' in counts and counts['scissors'] > 0:
                training_data['scissors'] = [np.zeros(128) for _ in range(counts['scissors'])]  # Placeholder
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
        
        # Process the image with improved methods
        processed_img = preprocess_image(img)
        
        # Extract enhanced features
        features = extract_features(processed_img)
        
        # Add to training data
        training_data[gesture].append(features)
        
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
        
        # Prepare training data
        X = np.vstack([
            np.array(training_data['rock']),
            np.array(training_data['paper']),
            np.array(training_data['scissors'])
        ])
        
        y = np.hstack([
            np.full(len(training_data['rock']), 'rock'),
            np.full(len(training_data['paper']), 'paper'),
            np.full(len(training_data['scissors']), 'scissors')
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
        
        # Process the image with improved methods
        processed_img = preprocess_image(img)
        
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
