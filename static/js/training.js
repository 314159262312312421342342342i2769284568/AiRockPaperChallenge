document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const captureBtn = document.getElementById('capture-btn');
    const trainBtn = document.getElementById('train-btn');
    const goToGameBtn = document.getElementById('go-to-game');
    const statusMessage = document.getElementById('status-message');
    const rockProgress = document.getElementById('rock-progress');
    const paperProgress = document.getElementById('paper-progress');
    const scissorsProgress = document.getElementById('scissors-progress');
    const rockCount = document.getElementById('rock-count');
    const paperCount = document.getElementById('paper-count');
    const scissorsCount = document.getElementById('scissors-count');
    const cameraFeedback = document.getElementById('camera-feedback');
    const feedbackMessage = document.getElementById('feedback-message');
    const cameraContainer = document.getElementById('camera-container');
    
    // Current counts
    let counts = {
        rock: 0,
        paper: 0,
        scissors: 0
    };
    
    // Target sample count
    const targetCount = 30;
    
    // Button animations
    let captureAnimationTimeout = null;
    
    // Initialize camera
    CameraHandler.init('webcam', 'canvas')
        .then(success => {
            if (!success) {
                showErrorMessage('Failed to initialize camera. Please check your permissions.');
            } else {
                showWelcomeMessage();
            }
        })
        .catch(error => {
            console.error('Camera initialization error:', error);
            showErrorMessage('Camera error: ' + error.message);
        });
    
    // Welcome message function
    function showWelcomeMessage() {
        statusMessage.className = 'alert alert-info';
        statusMessage.innerHTML = `
            <div class="d-flex">
                <div class="me-3 fs-4">
                    <i class="bi bi-info-circle-fill"></i>
                </div>
                <div>
                    <strong>Welcome to Training Mode!</strong>
                    <p class="mb-0">Select a gesture type and capture at least 10 samples of each gesture to train the AI.</p>
                </div>
            </div>
        `;
    }
        
    // Capture image button click handler with enhanced feedback
    captureBtn.addEventListener('click', async () => {
        if (!CameraHandler.isActive()) {
            showErrorMessage('Camera is not active. Please refresh the page.');
            return;
        }
        
        // Add a visual feedback that image is being captured
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="bi bi-camera-fill me-2"></i> Capturing...';
        captureBtn.classList.add('btn-pulse');
        
        // Show feedback overlay
        cameraFeedback.style.display = 'block';
        feedbackMessage.textContent = 'Capturing image...';
        
        try {
            // Get selected gesture
            const selectedGesture = document.querySelector('input[name="gesture"]:checked').value;
            
            // Brief pause for visual effect
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Add flash effect to camera
            cameraContainer.classList.add('flash-effect');
            setTimeout(() => cameraContainer.classList.remove('flash-effect'), 300);
            
            // Capture frame
            const imageData = CameraHandler.captureFrame();
            
            // Update feedback
            feedbackMessage.textContent = 'Processing...';
            
            // Send to server
            const response = await fetch('/capture_training_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    gesture: selectedGesture
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update counts
                counts = data.counts;
                updateProgressBars();
                
                // Show capture success in camera feedback
                cameraFeedback.style.display = 'block';
                feedbackMessage.textContent = `${capitalizeFirstLetter(selectedGesture)} gesture captured successfully!`;
                
                // Hide feedback after 2 seconds
                setTimeout(() => {
                    cameraFeedback.style.display = 'none';
                }, 2000);
                
                // Show success message with icon and more details
                statusMessage.className = 'alert alert-success';
                statusMessage.innerHTML = `
                    <div class="d-flex">
                        <div class="me-3 fs-4">
                            <i class="bi bi-check-circle-fill"></i>
                        </div>
                        <div>
                            <strong>${capitalizeFirstLetter(selectedGesture)} gesture captured!</strong>
                            <p class="mb-0">${data.message} Keep adding variety in positions and angles.</p>
                        </div>
                    </div>
                `;
                
                // Enable train button if we have enough samples
                checkEnableTrainButton();
                
                // Highlight the progress bar that was just updated
                highlightProgressBar(selectedGesture);
            } else {
                // Show error feedback
                cameraFeedback.style.display = 'block';
                feedbackMessage.textContent = 'Error capturing image';
                
                // Hide feedback after 2 seconds
                setTimeout(() => {
                    cameraFeedback.style.display = 'none';
                }, 2000);
                
                showErrorMessage('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error capturing image:', error);
            showErrorMessage('Error capturing image: ' + error.message);
            
            // Show error feedback
            cameraFeedback.style.display = 'block';
            feedbackMessage.textContent = 'Error capturing image';
            
            // Hide feedback after 2 seconds
            setTimeout(() => {
                cameraFeedback.style.display = 'none';
            }, 2000);
        } finally {
            // Re-enable capture button
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="bi bi-camera-fill me-2"></i> Capture Image';
            captureBtn.classList.remove('btn-pulse');
        }
    });
    
    // Highlight a progress bar that was just updated
    function highlightProgressBar(gesture) {
        const progressElement = document.getElementById(`${gesture}-progress`);
        const countElement = document.getElementById(`${gesture}-count`);
        
        if (progressElement && countElement) {
            progressElement.classList.add('progress-highlight');
            countElement.classList.add('badge-pulse');
            
            setTimeout(() => {
                progressElement.classList.remove('progress-highlight');
                countElement.classList.remove('badge-pulse');
            }, 1500);
        }
    }
    
    // Train model button click handler with enhanced feedback
    trainBtn.addEventListener('click', async () => {
        try {
            // Update UI
            statusMessage.className = 'alert alert-info';
            statusMessage.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="me-3 fs-4">
                        <i class="bi bi-gear-fill fa-spin"></i>
                    </div>
                    <div>
                        <strong>Training AI Model...</strong>
                        <p class="mb-0">This may take a few seconds. Please wait while the system learns from your gestures.</p>
                    </div>
                </div>
            `;
            
            // Disable buttons and add visual feedback
            trainBtn.disabled = true;
            trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Training...';
            captureBtn.disabled = true;
            
            // Add pulsing effect to status message
            statusMessage.classList.add('pulse-subtle');
            
            const response = await fetch('/train_model', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            // Remove pulsing effect
            statusMessage.classList.remove('pulse-subtle');
            
            if (data.success) {
                // Success feedback
                statusMessage.className = 'alert alert-success';
                statusMessage.innerHTML = `
                    <div class="d-flex">
                        <div class="me-3 fs-4">
                            <i class="bi bi-check-circle-fill"></i>
                        </div>
                        <div>
                            <strong>Training Complete!</strong>
                            <p class="mb-0">${data.message} You can now play the game with your trained model.</p>
                        </div>
                    </div>
                `;
                
                // Enable and highlight go to game button
                goToGameBtn.classList.remove('disabled');
                goToGameBtn.classList.add('btn-glow');
                
                // Re-enable buttons
                captureBtn.disabled = false;
                trainBtn.innerHTML = '<i class="bi bi-robot me-2"></i> Train AI Model';
                
                // Show a celebration animation
                showTrainingCompleteCelebration();
            } else {
                // Error feedback
                statusMessage.className = 'alert alert-danger';
                statusMessage.innerHTML = `
                    <div class="d-flex">
                        <div class="me-3 fs-4">
                            <i class="bi bi-exclamation-triangle-fill"></i>
                        </div>
                        <div>
                            <strong>Training Error</strong>
                            <p class="mb-0">${data.error} Please try again or capture more sample images.</p>
                        </div>
                    </div>
                `;
                
                // Re-enable buttons
                trainBtn.disabled = false;
                trainBtn.innerHTML = '<i class="bi bi-robot me-2"></i> Train AI Model';
                captureBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error training model:', error);
            statusMessage.className = 'alert alert-danger';
            statusMessage.innerHTML = `
                <div class="d-flex">
                    <div class="me-3 fs-4">
                        <i class="bi bi-exclamation-triangle-fill"></i>
                    </div>
                    <div>
                        <strong>Error Training Model</strong>
                        <p class="mb-0">${error.message} Please try again or check your connection.</p>
                    </div>
                </div>
            `;
            
            // Re-enable buttons
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="bi bi-robot me-2"></i> Train AI Model';
            captureBtn.disabled = false;
            statusMessage.classList.remove('pulse-subtle');
        }
    });
    
    // Show a celebration animation when training is complete
    function showTrainingCompleteCelebration() {
        // Add CSS class for celebration animation
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.classList.add('celebration-pulse');
            
            // Remove after animation completes
            setTimeout(() => {
                bar.classList.remove('celebration-pulse');
            }, 3000);
        });
    }
    
    // Update progress bars and badge counters
    function updateProgressBars() {
        updateProgressBar(rockProgress, rockCount, counts.rock);
        updateProgressBar(paperProgress, paperCount, counts.paper);
        updateProgressBar(scissorsProgress, scissorsCount, counts.scissors);
    }
    
    // Update a single progress bar and its badge counter
    function updateProgressBar(progressElement, countElement, count) {
        const percentage = Math.min(Math.round((count / targetCount) * 100), 100);
        
        // Update progress bar
        progressElement.style.width = percentage + '%';
        progressElement.setAttribute('aria-valuenow', count);
        progressElement.textContent = count + '/' + targetCount;
        
        // Update badge counter
        if (countElement) {
            countElement.textContent = count + '/' + targetCount;
        }
        
        // Change color based on progress
        if (percentage < 33) {
            progressElement.className = 'progress-bar bg-danger';
        } else if (percentage < 66) {
            progressElement.className = 'progress-bar bg-warning';
        } else {
            progressElement.className = 'progress-bar bg-success';
        }
    }
    
    // Check if we should enable the train button
    function checkEnableTrainButton() {
        const minCount = Math.min(counts.rock, counts.paper, counts.scissors);
        const shouldEnable = minCount >= 10; // Enable if we have at least 10 samples of each
        
        trainBtn.disabled = !shouldEnable;
        
        // Add visual highlight if newly enabled
        if (shouldEnable && trainBtn.disabled) {
            trainBtn.classList.add('btn-attention');
            setTimeout(() => {
                trainBtn.classList.remove('btn-attention');
            }, 2000);
        }
    }
    
    // Helper to capitalize first letter
    function capitalizeFirstLetter(string) {
        if (typeof string !== 'string') return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Show error message with icon
    function showErrorMessage(message) {
        statusMessage.className = 'alert alert-danger';
        statusMessage.innerHTML = `
            <div class="d-flex">
                <div class="me-3 fs-4">
                    <i class="bi bi-exclamation-triangle-fill"></i>
                </div>
                <div>
                    <strong>Error</strong>
                    <p class="mb-0">${message}</p>
                </div>
            </div>
        `;
    }
    
    // Add CSS for flash effect and progress highlights
    const style = document.createElement('style');
    style.textContent = `
        .flash-effect {
            animation: flash 0.3s;
        }
        
        @keyframes flash {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .progress-highlight {
            animation: highlight-pulse 1.5s;
        }
        
        @keyframes highlight-pulse {
            0% { box-shadow: 0 0 0 0 rgba(var(--bs-success-rgb), 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(var(--bs-success-rgb), 0); }
            100% { box-shadow: 0 0 0 0 rgba(var(--bs-success-rgb), 0); }
        }
        
        .badge-pulse {
            animation: badge-highlight 1.5s;
        }
        
        @keyframes badge-highlight {
            0% { transform: scale(1); }
            50% { transform: scale(1.3); }
            100% { transform: scale(1); }
        }
        
        .celebration-pulse {
            animation: celebration 1.5s infinite;
        }
        
        @keyframes celebration {
            0% { background-color: var(--bs-success); }
            33% { background-color: var(--bs-primary); }
            66% { background-color: var(--bs-warning); }
            100% { background-color: var(--bs-success); }
        }
        
        .btn-pulse {
            animation: btn-pulse 1s infinite;
        }
        
        @keyframes btn-pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .pulse-subtle {
            animation: pulse-subtle 1.5s infinite;
        }
        
        @keyframes pulse-subtle {
            0% { opacity: 1; }
            50% { opacity: 0.8; }
            100% { opacity: 1; }
        }
        
        .btn-glow {
            animation: btn-glow 2s infinite;
        }
        
        @keyframes btn-glow {
            0% { box-shadow: 0 0 5px var(--bs-success); }
            50% { box-shadow: 0 0 20px var(--bs-success); }
            100% { box-shadow: 0 0 5px var(--bs-success); }
        }
    `;
    document.head.appendChild(style);
});
