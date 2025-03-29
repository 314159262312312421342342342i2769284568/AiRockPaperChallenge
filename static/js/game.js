document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const playBtn = document.getElementById('play-btn');
    const resetBtn = document.getElementById('reset-game');
    const playerGestureDisplay = document.getElementById('player-gesture-display');
    const computerGestureDisplay = document.getElementById('computer-gesture-display');
    const playerGestureText = document.getElementById('player-gesture-text');
    const computerGestureText = document.getElementById('computer-gesture-text');
    const resultText = document.getElementById('result-text');
    const playerScore = document.getElementById('player-score');
    const computerScore = document.getElementById('computer-score');
    const roundsPlayed = document.getElementById('rounds-played');
    const detectionInfo = document.getElementById('detection-info');
    const modelNotTrained = document.getElementById('model-not-trained');
    const gameContainer = document.getElementById('game-container');
    
    // Game state
    let gameActive = false;
    
    // Initialize camera
    CameraHandler.init('webcam', 'canvas')
        .then(success => {
            if (!success) {
                showError('Failed to initialize camera. Please check your permissions.');
            } else {
                // Check if model is trained
                checkModelStatus();
            }
        })
        .catch(error => {
            console.error('Camera initialization error:', error);
            showError('Camera error: ' + error.message);
        });
    
    // Play button click handler
    playBtn.addEventListener('click', async () => {
        if (!CameraHandler.isActive()) {
            showError('Camera is not active. Please refresh the page.');
            return;
        }
        
        try {
            // Disable button while processing
            playBtn.disabled = true;
            
            // Show detection in progress
            detectionInfo.className = 'alert alert-info';
            detectionInfo.textContent = 'Detecting your gesture...';
            
            // Capture frame
            const imageData = CameraHandler.captureFrame();
            
            // Send to server for prediction
            const response = await fetch('/predict_gesture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update UI with results
                updateGameResults(data);
            } else {
                detectionInfo.className = 'alert alert-danger';
                detectionInfo.textContent = 'Error: ' + data.error;
            }
            
            // Re-enable button
            playBtn.disabled = false;
        } catch (error) {
            console.error('Error playing round:', error);
            detectionInfo.className = 'alert alert-danger';
            detectionInfo.textContent = 'Error playing round: ' + error.message;
            playBtn.disabled = false;
        }
    });
    
    // Reset button click handler
    resetBtn.addEventListener('click', () => {
        // Reset game state
        playerScore.textContent = '0';
        computerScore.textContent = '0';
        roundsPlayed.textContent = '0';
        
        // Reset displays
        playerGestureDisplay.innerHTML = '<i class="bi bi-question-circle"></i>';
        computerGestureDisplay.innerHTML = '<i class="bi bi-question-circle"></i>';
        playerGestureText.textContent = 'Ready';
        computerGestureText.textContent = 'Ready';
        resultText.textContent = 'Make a gesture and click "Play Round"';
        resultText.className = '';
        
        // Reset info
        detectionInfo.className = 'alert alert-secondary';
        detectionInfo.textContent = 'Game reset. Ready to play!';
    });
    
    // Check if model is trained
    async function checkModelStatus() {
        try {
            const response = await fetch('/predict_gesture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: CameraHandler.captureFrame()
                })
            });
            
            const data = await response.json();
            
            if (data.success || data.error !== 'Model not trained yet') {
                // Model is trained
                gameActive = true;
                modelNotTrained.style.display = 'none';
                gameContainer.style.display = 'flex';
                
                detectionInfo.className = 'alert alert-success';
                detectionInfo.textContent = 'Model loaded! Ready to play.';
            } else {
                // Model is not trained
                gameActive = false;
                modelNotTrained.style.display = 'block';
                gameContainer.style.display = 'none';
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            showError('Error checking model status: ' + error.message);
        }
    }
    
    // Update game results
    function updateGameResults(data) {
        // Update player gesture
        playerGestureText.textContent = capitalizeFirstLetter(data.player_gesture);
        playerGestureDisplay.innerHTML = getGestureIcon(data.player_gesture);
        
        // Update computer gesture
        computerGestureText.textContent = capitalizeFirstLetter(data.computer_choice);
        computerGestureDisplay.innerHTML = getGestureIcon(data.computer_choice);
        
        // Update result
        let resultMessage = '';
        let resultClass = '';
        
        switch (data.result) {
            case 'win':
                resultMessage = 'You Win!';
                resultClass = 'text-success';
                break;
            case 'lose':
                resultMessage = 'Computer Wins!';
                resultClass = 'text-danger';
                break;
            case 'tie':
                resultMessage = 'It\'s a Tie!';
                resultClass = 'text-warning';
                break;
        }
        
        resultText.textContent = resultMessage;
        resultText.className = resultClass;
        
        // Update scoreboard
        playerScore.textContent = data.game_state.player_score;
        computerScore.textContent = data.game_state.computer_score;
        roundsPlayed.textContent = data.game_state.rounds;
        
        // Update detection info
        detectionInfo.className = 'alert alert-success';
        detectionInfo.textContent = `Detected: ${capitalizeFirstLetter(data.player_gesture)}`;
    }
    
    // Get icon for gesture
    function getGestureIcon(gesture) {
        switch (gesture) {
            case 'rock':
                return '<i class="bi bi-circle-fill" style="font-size: 3rem;"></i>';
            case 'paper':
                return '<i class="bi bi-file-earmark" style="font-size: 3rem;"></i>';
            case 'scissors':
                return '<i class="bi bi-scissors" style="font-size: 3rem;"></i>';
            default:
                return '<i class="bi bi-question-circle"></i>';
        }
    }
    
    // Helper to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Show error message
    function showError(message) {
        detectionInfo.className = 'alert alert-danger';
        detectionInfo.textContent = message;
    }
});
