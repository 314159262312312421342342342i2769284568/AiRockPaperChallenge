import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging

logger = logging.getLogger(__name__)

class GestureClassifier:
    """
    Enhanced class for training and predicting hand gestures
    using ensemble methods and better validation
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.class_labels = None  # Store the class labels
        self.feature_importances = None  # Store feature importances
    
    def train(self, X, y):
        """
        Train the classifier with feature vectors X and labels y
        with improved preprocessing and model selection
        
        Args:
            X: numpy array of feature vectors
            y: numpy array of labels ('rock', 'paper', 'scissors')
        
        Returns:
            bool: True if training was successful
        """
        try:
            # Store the class labels
            self.class_labels = np.unique(y)
            logger.info(f"Training with {len(X)} samples, class distribution: {np.bincount(np.searchsorted(self.class_labels, y))}")
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Create a scaler to standardize features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Try different models and select the best one
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    bootstrap=True,
                    class_weight='balanced',
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'svm': SVC(
                    kernel='rbf',
                    C=10,
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
            }
            
            best_score = 0
            best_model_name = None
            
            for name, model in models.items():
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val_scaled)
                score = accuracy_score(y_val, y_pred)
                
                logger.info(f"Model {name} validation accuracy: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    self.model = model
                    
            logger.info(f"Selected model: {best_model_name} with accuracy: {best_score:.4f}")
            
            # Log detailed performance metrics
            y_pred = self.model.predict(X_val_scaled)
            report = classification_report(y_val, y_pred, target_names=self.class_labels)
            logger.info(f"Classification report:\n{report}")
            
            # Store feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_
                # Log top important features
                if self.feature_importances is not None:
                    top_indices = np.argsort(self.feature_importances)[-10:]
                    logger.info(f"Top feature indices: {top_indices}")
                    logger.info(f"Top feature importances: {self.feature_importances[top_indices]}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training the model: {str(e)}")
            return False
    
    def predict(self, X):
        """
        Predict the gesture from a feature vector
        with confidence scores
        
        Args:
            X: feature vector
        
        Returns:
            str: predicted gesture ('rock', 'paper', 'scissors')
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        try:
            # Scale feature vector
            X_scaled = self.scaler.transform([X])
            
            # Get prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                max_proba = max(proba)
                logger.debug(f"Prediction: {prediction}, confidence: {max_proba:.4f}")
                
                # If confidence is too low, might be unreliable
                if max_proba < 0.6:
                    logger.warning(f"Low confidence prediction: {prediction} with confidence {max_proba:.4f}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fallback to a more robust prediction method if something went wrong
            try:
                # Try model's direct predict method as fallback
                return self.model.predict([X])[0]
            except:
                # Last resort: return most common class
                return self.class_labels[0] if self.class_labels is not None else "unknown"
    
    def save_model(self, filepath='hand_gesture_model.pkl'):
        """
        Save the trained model and scaler to a file
        
        Args:
            filepath: path to save the model
        
        Returns:
            bool: True if saving was successful
        """
        if not self.is_trained:
            return False
        
        try:
            # Save model, scaler and class labels together
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'class_labels': self.class_labels,
                'feature_importances': self.feature_importances
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving the model: {str(e)}")
            return False
    
    def load_model(self, filepath='hand_gesture_model.pkl'):
        """
        Load a trained model and scaler from a file
        
        Args:
            filepath: path to the model file
        
        Returns:
            bool: True if loading was successful
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if the file contains the new format (dict with model and scaler)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.class_labels = model_data.get('class_labels')
                self.feature_importances = model_data.get('feature_importances')
            else:
                # Old format: just the model
                self.model = model_data
                self.scaler = None
                
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Model file {filepath} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading the model: {str(e)}")
            return False
