import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EnsembleGestureClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom ensemble classifier that combines multiple models with confidence weighting
    and adds stability for gesture recognition
    """
    def __init__(self, models=None):
        self.models = models or []
        self.weights = None
        self.class_labels = None
    
    def fit(self, X, y):
        """Train all models and determine their weights based on validation performance"""
        # Store class labels
        self.class_labels = np.unique(y)
        
        # Split for internal validation to determine weights
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train each model
        accuracies = []
        for model_name, model in self.models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(max(acc, 0.1))  # Ensure minimum weight
        
        # Normalize to get weights that sum to 1
        self.weights = np.array(accuracies) / sum(accuracies)
        logger.info(f"Model weights: {[f'{n}: {w:.2f}' for (n, _), w in zip(self.models, self.weights)]}")
        
        # Retrain on full dataset
        for _, model in self.models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities as weighted average of individual models"""
        # Get predictions from each model
        all_probas = []
        
        for i, (name, model) in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba * self.weights[i])
            else:
                # For models without predict_proba, create one-hot encoded array
                preds = model.predict(X)
                proba = np.zeros((X.shape[0], len(self.class_labels)))
                for j, pred in enumerate(preds):
                    idx = np.where(self.class_labels == pred)[0][0]
                    proba[j, idx] = 1
                all_probas.append(proba * self.weights[i])
        
        # Weighted average
        avg_proba = sum(all_probas)
        return avg_proba
    
    def predict(self, X):
        """Predict class labels with weighted voting"""
        probas = self.predict_proba(X)
        return self.class_labels[np.argmax(probas, axis=1)]

class GestureClassifier:
    """
    Significantly enhanced class for training and predicting hand gestures
    using ensemble methods and improved validation techniques
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.class_labels = None  # Store the class labels
        self.feature_importances = None  # Store feature importances
        self.validation_score = 0  # Track validation score
        self.confidence_threshold = 0.5  # Reduced threshold to allow more predictions
        self.model_metrics = {}  # Store metrics per class
    
    def train(self, X, y):
        """
        Train the classifier with feature vectors X and labels y
        with significantly improved preprocessing and model selection
        
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
            
            # Split data for validation - stratify to ensure balanced classes
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            
            # Use a RobustScaler which is less influenced by outliers
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Enhanced models with optimized hyperparameters for better gesture discrimination
            rf = RandomForestClassifier(
                n_estimators=200,  # More trees for better ensemble effect
                max_depth=10,      # Reduced depth to avoid overfitting to rock
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=True,
                class_weight='balanced_subsample',  # Better handling of class imbalance
                criterion='entropy',  # Different split criterion
                n_jobs=-1,
                random_state=42
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,  # Lower learning rate for better generalization
                max_depth=4,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            svm = SVC(
                kernel='rbf',
                C=10,          # Lower C to reduce overfitting
                gamma='scale', # Better gamma strategy
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            
            # Create a custom ensemble classifier
            ensemble = EnsembleGestureClassifier(models=[
                ('random_forest', rf),
                ('gradient_boosting', gb),
                ('svm', svm)
            ])
            
            # Train the ensemble
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            y_pred = ensemble.predict(X_val_scaled)
            val_score = accuracy_score(y_val, y_pred)
            self.validation_score = val_score
            
            logger.info(f"Ensemble validation accuracy: {val_score:.4f}")
            
            # Log detailed performance metrics
            report = classification_report(y_val, y_pred, target_names=self.class_labels)
            logger.info(f"Classification report:\n{report}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            logger.info(f"Confusion matrix:\n{cm}")
            
            # Store feature importances from Random Forest
            if hasattr(rf, 'feature_importances_'):
                self.feature_importances = rf.feature_importances_
                # Log top important features
                if self.feature_importances is not None:
                    top_indices = np.argsort(self.feature_importances)[-10:]
                    logger.info(f"Top feature indices: {top_indices}")
                    logger.info(f"Top feature importances: {self.feature_importances[top_indices]}")
            
            # Set the model
            self.model = ensemble
            self.is_trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error training the model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, X):
        """
        Predict the gesture from a feature vector with robust error handling
        and confidence-based decision making
        
        Args:
            X: feature vector
        
        Returns:
            str: predicted gesture ('rock', 'paper', 'scissors')
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            # Convert single feature vector to 2D array if needed
            X_input = X.reshape(1, -1) if len(X.shape) == 1 else X
            
            # Scale feature vector
            X_scaled = self.scaler.transform(X_input)
            
            # Get predictions from all models in the ensemble to counteract overfitting
            individual_predictions = []
            all_probas = []
            
            if hasattr(self.model, 'models'):
                for name, model in self.model.models:
                    # Get individual model prediction
                    pred = model.predict(X_scaled)[0]
                    individual_predictions.append(pred)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        all_probas.append(proba)
            
                # If we have multiple models with different predictions, apply special logic
                if len(set(individual_predictions)) > 1:
                    # Not all models agree - implement a voting system
                    from collections import Counter
                    # Count occurrences of each prediction
                    pred_counts = Counter(individual_predictions)
                    # Get the most common prediction
                    prediction, count = pred_counts.most_common(1)[0]
                    
                    # If it's a real majority (more than half), use that prediction
                    if count > len(individual_predictions) / 2:
                        logger.debug(f"Majority vote prediction: {prediction} with {count}/{len(individual_predictions)} votes")
                        return prediction
            
            # Get ensemble probability scores
            probas = self.model.predict_proba(X_scaled)[0]
            max_proba_idx = np.argmax(probas)
            max_proba = probas[max_proba_idx]
            prediction = self.class_labels[max_proba_idx]
            
            # Check if probabilities are too close (within 20%)
            sorted_indices = np.argsort(probas)
            if len(sorted_indices) >= 2:
                top1_idx = sorted_indices[-1]
                top2_idx = sorted_indices[-2]
                
                top1_prob = probas[top1_idx]
                top2_prob = probas[top2_idx]
                
                # If the difference is small, look for additional evidence
                if (top1_prob - top2_prob) < 0.2:
                    logger.debug(f"Close prediction: {self.class_labels[top1_idx]} ({top1_prob:.2f}) vs {self.class_labels[top2_idx]} ({top2_prob:.2f})")
                    
                    # Use more aggressive tiebreaker by checking individual model predictions
                    if len(individual_predictions) > 0 and individual_predictions.count(self.class_labels[top2_idx]) > individual_predictions.count(self.class_labels[top1_idx]):
                        # The 2nd most likely class has more individual model votes
                        prediction = self.class_labels[top2_idx]
                        max_proba = top2_prob
                        logger.debug(f"Switched prediction to {prediction} based on individual model votes")
            
            logger.debug(f"Final prediction: {prediction}, confidence: {max_proba:.4f}")
            
            # If confidence is too low, it might be unreliable but we still return best guess
            if max_proba < self.confidence_threshold:
                logger.warning(f"Low confidence prediction: {prediction} with confidence {max_proba:.4f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback prediction with more robust error handling
            try:
                # Try direct prediction on unscaled data as last resort
                if hasattr(self.model, 'predict'):
                    return self.model.predict(X.reshape(1, -1))[0]
                elif hasattr(self.model, 'models') and self.model.models:
                    # Get individual model predictions for fallback
                    preds = []
                    for _, model in self.model.models:
                        if hasattr(model, 'predict'):
                            preds.append(model.predict(X.reshape(1, -1))[0])
                    
                    if preds:
                        # Return the most common prediction
                        from collections import Counter
                        most_common = Counter(preds).most_common(1)[0][0]
                        return most_common
            except Exception as fallback_error:
                logger.error(f"Fallback prediction error: {str(fallback_error)}")
                pass
                
            # As a very last resort, return a random choice to avoid biasing toward rock
            import random
            if self.class_labels is not None and len(self.class_labels) > 0:
                return random.choice(self.class_labels)
            else:
                return "unknown"
    
    def save_model(self, filepath='hand_gesture_model.pkl'):
        """
        Save the trained model and all necessary data to a file
        with backup file creation
        
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
                'feature_importances': self.feature_importances,
                'validation_score': self.validation_score,
                'version': 2,  # Increment version when making significant changes
                'timestamp': datetime.now().isoformat(),
            }
            
            # Create backup of existing model if it exists
            if os.path.exists(filepath):
                backup_path = f"{filepath}.bak"
                try:
                    os.rename(filepath, backup_path)
                    logger.info(f"Created backup of previous model: {backup_path}")
                except:
                    logger.warning(f"Could not create backup of previous model")
            
            # Save the new model
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving the model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_model(self, filepath='hand_gesture_model.pkl'):
        """
        Load a trained model and all necessary data from a file
        with robust error handling
        
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
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.class_labels = model_data.get('class_labels')
                self.feature_importances = model_data.get('feature_importances')
                self.validation_score = model_data.get('validation_score', 0)
                
                version = model_data.get('version', 1)
                logger.debug(f"Loaded model version {version} from {filepath}")
            else:
                # Very old format: just the model
                self.model = model_data
                self.scaler = None
                logger.debug(f"Loaded legacy model format from {filepath}")
                
            self.is_trained = True
            return True
            
        except FileNotFoundError:
            logger.warning(f"Model file {filepath} not found")
            return False
        except Exception as e:
            logger.error(f"Error loading the model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
