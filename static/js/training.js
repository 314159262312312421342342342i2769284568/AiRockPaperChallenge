document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const captureBtn = document.getElementById('capture-btn');
    const trainBtn = document.getElementById('train-btn');
    const goToGameBtn = document.getElementById('go-to-game');
    const statusMessage = document.getElementById('status-message');
    const rockProgress = document.getElementById('rock-progress');
    const paperProgress = document.getElementById('paper-progress');
    const scissorsProgress = document.getElementById('scissors-progress');
    
    // Current counts
    let counts = {
        rock: 0,
        paper: 0,
        scissors: 0
    };
    
    // Target sample count
    const targetCount = 30;
    
    // Initialize camera
    CameraHandler.init('webcam', 'canvas')
        .then(success => {
            if (!success) {
                showErrorMessage('Failed to initialize camera. Please check your permissions.');
            }
        })
        .catch(error => {
            console.error('Camera initialization error:', error);
            showErrorMessage('Camera error: ' + error.message);
        });
    
    // Capture image button click handler
    captureBtn.addEventListener('click', async () => {
        if (!CameraHandler.isActive()) {
            showErrorMessage('Camera is not active. Please refresh the page.');
            return;
        }
        
        try {
            // Get selected gesture
            const selectedGesture = document.querySelector('input[name="gesture"]:checked').value;
            
            // Capture frame
            const imageData = CameraHandler.captureFrame();
            
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
                
                // Show success message
                statusMessage.className = 'alert alert-success';
                statusMessage.textContent = data.message;
                
                // Enable train button if we have enough samples
                checkEnableTrainButton();
            } else {
                showErrorMessage('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error capturing image:', error);
            showErrorMessage('Error capturing image: ' + error.message);
        }
    });
    
    // Train model button click handler
    trainBtn.addEventListener('click', async () => {
        try {
            statusMessage.className = 'alert alert-info';
            statusMessage.textContent = 'Training model... This may take a few seconds.';
            trainBtn.disabled = true;
            
            const response = await fetch('/train_model', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                statusMessage.className = 'alert alert-success';
                statusMessage.textContent = data.message + ' You can now play the game!';
                goToGameBtn.classList.remove('disabled');
            } else {
                statusMessage.className = 'alert alert-danger';
                statusMessage.textContent = 'Training error: ' + data.error;
                trainBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error training model:', error);
            statusMessage.className = 'alert alert-danger';
            statusMessage.textContent = 'Error training model: ' + error.message;
            trainBtn.disabled = false;
        }
    });
    
    // Update progress bars
    function updateProgressBars() {
        updateProgressBar(rockProgress, counts.rock);
        updateProgressBar(paperProgress, counts.paper);
        updateProgressBar(scissorsProgress, counts.scissors);
    }
    
    // Update a single progress bar
    function updateProgressBar(element, count) {
        const percentage = Math.min(Math.round((count / targetCount) * 100), 100);
        element.style.width = percentage + '%';
        element.setAttribute('aria-valuenow', count);
        element.textContent = count + '/' + targetCount;
        
        // Change color based on progress
        if (percentage < 33) {
            element.className = 'progress-bar bg-danger';
        } else if (percentage < 66) {
            element.className = 'progress-bar bg-warning';
        } else {
            element.className = 'progress-bar bg-success';
        }
    }
    
    // Check if we should enable the train button
    function checkEnableTrainButton() {
        const minCount = Math.min(counts.rock, counts.paper, counts.scissors);
        trainBtn.disabled = minCount < 10; // Enable if we have at least 10 samples of each
    }
    
    // Show error message
    function showErrorMessage(message) {
        statusMessage.className = 'alert alert-danger';
        statusMessage.textContent = message;
    }
});
