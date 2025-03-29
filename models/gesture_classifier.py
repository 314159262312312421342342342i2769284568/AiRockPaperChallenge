import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging

logger = logging.getLogger(__name__)

class GestureClassifier:
    """
    Class for training and predicting hand gestures using Random Forest
    """
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train the classifier with feature vectors X and labels y
        
        Args:
            X: numpy array of feature vectors
            y: numpy array of labels ('rock', 'paper', 'scissors')
        
        Returns:
            bool: True if training was successful
        """
        try:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error training the model: {str(e)}")
            return False
    
    def predict(self, X):
        """
        Predict the gesture from a feature vector
        
        Args:
            X: feature vector
        
        Returns:
            str: predicted gesture ('rock', 'paper', 'scissors')
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        return self.model.predict([X])[0]
    
    def save_model(self, filepath='hand_gesture_model.pkl'):
        """
        Save the trained model to a file
        
        Args:
            filepath: path to save the model
        
        Returns:
            bool: True if saving was successful
        """
        if not self.is_trained:
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            return True
        except Exception as e:
            logger.error(f"Error saving the model: {str(e)}")
            return False
    
    def load_model(self, filepath='hand_gesture_model.pkl'):
        """
        Load a trained model from a file
        
        Args:
            filepath: path to the model file
        
        Returns:
            bool: True if loading was successful
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            return True
        except FileNotFoundError:
            logger.warning(f"Model file {filepath} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading the model: {str(e)}")
            return False
