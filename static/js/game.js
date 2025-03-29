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
    const roundNumber = document.getElementById('round-number');
    const detectionInfo = document.getElementById('detection-info');
    const modelNotTrained = document.getElementById('model-not-trained');
    const gameContainer = document.getElementById('game-container');
    const cameraOverlay = document.getElementById('camera-overlay');
    const countdownElement = document.getElementById('countdown');
    
    // Game state
    let gameActive = false;
    let countdownActive = false;
    
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
    
    // Play button click handler with countdown
    playBtn.addEventListener('click', async () => {
        if (!CameraHandler.isActive()) {
            showError('Camera is not active. Please refresh the page.');
            return;
        }
        
        if (countdownActive) return; // Prevent multiple clicks during countdown
        
        try {
            // Disable button during countdown and processing
            playBtn.disabled = true;
            countdownActive = true;
            
            // Update detection info
            detectionInfo.className = 'alert alert-info';
            detectionInfo.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="me-3 fs-3"><i class="bi bi-hourglass-split"></i></div>
                    <div>Get ready! Hold your gesture steady...</div>
                </div>
            `;
            
            // Reset result display during countdown
            resultText.textContent = 'Get Ready!';
            resultText.className = 'fs-2';
            
            // Reset gesture displays
            playerGestureDisplay.innerHTML = '<i class="bi bi-hand-index"></i>';
            playerGestureText.textContent = 'Ready';
            computerGestureDisplay.innerHTML = '<i class="bi bi-cpu"></i>';
            computerGestureText.textContent = 'Ready';
            
            // Show countdown overlay
            cameraOverlay.style.display = 'flex';
            
            // Start countdown from 3
            countdownElement.textContent = '3';
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            countdownElement.textContent = '2';
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            countdownElement.textContent = '1';
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            countdownElement.textContent = 'GO!';
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Hide countdown overlay
            cameraOverlay.style.display = 'none';
            
            // Show processing state
            detectionInfo.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="me-3 fs-3"><i class="bi bi-camera"></i></div>
                    <div>Detecting your gesture...</div>
                </div>
            `;
            
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
                detectionInfo.innerHTML = `
                    <div class="d-flex align-items-center">
                        <div class="me-3 fs-3"><i class="bi bi-exclamation-triangle"></i></div>
                        <div>Error: ${data.error}</div>
                    </div>
                `;
            }
            
            // Re-enable button and reset countdown state
            playBtn.disabled = false;
            countdownActive = false;
        } catch (error) {
            console.error('Error playing round:', error);
            detectionInfo.className = 'alert alert-danger';
            detectionInfo.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="me-3 fs-3"><i class="bi bi-exclamation-triangle"></i></div>
                    <div>Error playing round: ${error.message}</div>
                </div>
            `;
            
            // Hide countdown overlay if error occurs
            cameraOverlay.style.display = 'none';
            
            // Re-enable button and reset countdown state
            playBtn.disabled = false;
            countdownActive = false;
        }
    });
    
    // Reset button click handler with enhanced feedback
    resetBtn.addEventListener('click', async () => {
        // Add reset animation
        const animateReset = async () => {
            // Apply fade-out effect
            gameContainer.style.opacity = '0.5';
            gameContainer.style.transition = 'opacity 0.3s ease';
            
            // Call the backend to reset game state
            await resetGameForNextRound();
            
            // Apply fade-in effect
            gameContainer.style.opacity = '1';
        };
        
        await animateReset();
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
                detectionInfo.innerHTML = `
                    <div class="d-flex align-items-center">
                        <div class="me-3 fs-3"><i class="bi bi-check-circle"></i></div>
                        <div>AI model loaded successfully! Click "Play Round" to start.</div>
                    </div>
                `;
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
    
    // Reset gameplay for a fresh round
    async function resetGameForNextRound() {
        try {
            const response = await fetch('/reset_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                console.log('Game state reset for next round');
                
                // Update UI with reset state
                playerScore.textContent = '0';
                computerScore.textContent = '0';
                roundsPlayed.textContent = '0';
                roundNumber.textContent = '0';
                
                // Reset UI
                playerGestureDisplay.innerHTML = '<i class="bi bi-hand-index"></i>';
                computerGestureDisplay.innerHTML = '<i class="bi bi-cpu"></i>';
                playerGestureText.textContent = 'Ready';
                computerGestureText.textContent = 'Ready';
                resultText.textContent = 'Make your move!';
                resultText.className = 'fs-2';
                
                // Update status
                detectionInfo.className = 'alert alert-success';
                detectionInfo.innerHTML = `
                    <div class="d-flex align-items-center">
                        <div class="me-3 fs-3"><i class="bi bi-check-circle"></i></div>
                        <div>Ready for a new game! Click "Play Round" to start.</div>
                    </div>
                `;
            } else {
                console.error('Failed to reset game state:', data.error);
            }
        } catch (error) {
            console.error('Error resetting game state:', error);
        }
    }
    
    // Update game results with enhanced feedback and animations
    function updateGameResults(data) {
        // Update round counter
        roundNumber.textContent = data.game_state.rounds;
        
        // Handle 'unknown' gesture
        const playerGesture = data.player_gesture === 'unknown' ? 'unrecognized gesture' : data.player_gesture;
        
        // Update player gesture with animation
        playerGestureDisplay.classList.add('gesture-animation');
        playerGestureText.textContent = capitalizeFirstLetter(playerGesture);
        playerGestureDisplay.innerHTML = getGestureIcon(data.player_gesture);
        
        // Anticipation pause before showing computer's choice
        resultText.textContent = 'Waiting for computer...';
        resultText.className = 'fs-2 text-info';
        
        // Delay computer's "choice" for dramatic effect
        setTimeout(() => {
            // Update computer gesture with animation
            computerGestureDisplay.classList.add('gesture-animation');
            computerGestureText.textContent = capitalizeFirstLetter(data.computer_choice);
            computerGestureDisplay.innerHTML = getGestureIcon(data.computer_choice);
            
            // Brief pause before showing the result
            setTimeout(() => {
                // Update result with more expressive messaging
                let resultMessage = '';
                let resultClass = '';
                let resultIcon = '';
                
                switch (data.result) {
                    case 'win':
                        resultMessage = 'You Win!';
                        resultClass = 'fs-2 text-success fw-bold';
                        resultIcon = '🏆';
                        // Add victory animation
                        playerGestureDisplay.classList.add('winner-pulse');
                        break;
                    case 'lose':
                        resultMessage = 'Computer Wins!';
                        resultClass = 'fs-2 text-danger fw-bold';
                        resultIcon = '💻';
                        // Add defeat animation
                        computerGestureDisplay.classList.add('winner-pulse');
                        break;
                    case 'tie':
                        resultMessage = 'It\'s a Tie!';
                        resultClass = 'fs-2 text-warning fw-bold';
                        resultIcon = '🤝';
                        // Add tie animation
                        playerGestureDisplay.classList.add('tie-pulse');
                        computerGestureDisplay.classList.add('tie-pulse');
                        break;
                    default:
                        resultMessage = 'Unexpected Result';
                        resultClass = 'fs-2 text-info';
                        resultIcon = '❓';
                }
                
                resultText.innerHTML = `${resultIcon} ${resultMessage} ${resultIcon}`;
                resultText.className = resultClass;
                
                // Update scoreboard with animation
                const oldPlayerScore = parseInt(playerScore.textContent) || 0;
                const oldComputerScore = parseInt(computerScore.textContent) || 0;
                const oldRoundsPlayed = parseInt(roundsPlayed.textContent) || 0;
                
                if (data.game_state.player_score > oldPlayerScore) {
                    playerScore.classList.add('score-change');
                }
                
                if (data.game_state.computer_score > oldComputerScore) {
                    computerScore.classList.add('score-change');
                }
                
                if (data.game_state.rounds > oldRoundsPlayed) {
                    roundsPlayed.classList.add('score-change');
                }
                
                playerScore.textContent = data.game_state.player_score;
                computerScore.textContent = data.game_state.computer_score;
                roundsPlayed.textContent = data.game_state.rounds;
                
                // Update detection info with enhanced feedback
                let confidenceIcon = '';
                let confidenceClass = '';
                let confidenceTitle = '';
                let confidenceMsg = '';
                
                if (data.player_gesture === 'unknown') {
                    confidenceClass = 'alert-warning';
                    confidenceIcon = 'bi-question-circle';
                    confidenceTitle = 'Gesture Not Recognized';
                    confidenceMsg = 'Try repositioning your hand or adjusting lighting';
                } else {
                    confidenceClass = 'alert-success';
                    confidenceIcon = 'bi-hand-thumbs-up';
                    confidenceTitle = 'Gesture Detected Successfully';
                    confidenceMsg = `${capitalizeFirstLetter(data.player_gesture)} was recognized with good confidence`;
                }
                
                detectionInfo.className = `alert ${confidenceClass} border-0 mb-3`;
                detectionInfo.innerHTML = `
                    <div class="d-flex align-items-center">
                        <div class="me-3 fs-3"><i class="bi ${confidenceIcon}"></i></div>
                        <div>
                            <strong>${confidenceTitle}</strong>
                            <div>${confidenceMsg}</div>
                        </div>
                    </div>
                `;
                
                // Remove animation classes after animation completes
                setTimeout(() => {
                    playerGestureDisplay.classList.remove('gesture-animation', 'winner-pulse', 'tie-pulse');
                    computerGestureDisplay.classList.remove('gesture-animation', 'winner-pulse', 'tie-pulse');
                    playerScore.classList.remove('score-change');
                    computerScore.classList.remove('score-change');
                    roundsPlayed.classList.remove('score-change');
                    
                    // Add a "Play Again" button after delay
                    const playAgainButton = document.createElement('button');
                    playAgainButton.className = 'btn btn-primary mt-3 w-100';
                    playAgainButton.innerHTML = '<i class="bi bi-play-circle me-2"></i> Play Again';
                    playAgainButton.onclick = () => {
                        // Re-enable the play button 
                        playBtn.disabled = false;
                        playBtn.click();
                        playAgainButton.remove();
                    };
                    
                    // Add the button to the result display
                    const resultDisplay = document.getElementById('result-display');
                    if (!document.getElementById('play-again-btn')) {
                        playAgainButton.id = 'play-again-btn';
                        resultDisplay.appendChild(playAgainButton);
                    }
                    
                    // Update the instructions message
                    detectionInfo.className = 'alert alert-info border-0 mb-3';
                    detectionInfo.innerHTML = `
                        <div class="d-flex align-items-center">
                            <div class="me-3 fs-3"><i class="bi bi-info-circle"></i></div>
                            <div>
                                <strong>Ready for next round!</strong>
                                <div>Click "Play Again" to show a new gesture.</div>
                            </div>
                        </div>
                    `;
                }, 1200);
                
            }, 500); // Delay showing result
            
        }, 800); // Delay computer choice reveal
    }
    
    // Get icon for gesture with enhanced styling
    function getGestureIcon(gesture) {
        switch (gesture) {
            case 'rock':
                return '<i class="bi bi-circle-fill" style="font-size: 3.5rem; color: var(--bs-gray-300);"></i>';
            case 'paper':
                return '<i class="bi bi-file-earmark" style="font-size: 3.5rem; color: var(--bs-info);"></i>';
            case 'scissors':
                return '<i class="bi bi-scissors" style="font-size: 3.5rem; color: var(--bs-warning);"></i>';
            case 'unknown':
                return '<i class="bi bi-question-circle" style="font-size: 3.5rem; color: var(--bs-danger);"></i>';
            default:
                return '<i class="bi bi-hand-index" style="font-size: 3.5rem;"></i>';
        }
    }
    
    // Helper to capitalize first letter
    function capitalizeFirstLetter(string) {
        if (typeof string !== 'string') return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Show error message with icon
    function showError(message) {
        detectionInfo.className = 'alert alert-danger';
        detectionInfo.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="me-3 fs-3"><i class="bi bi-exclamation-triangle"></i></div>
                <div>${message}</div>
            </div>
        `;
    }
});
