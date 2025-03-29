// Enhanced camera handling module
const CameraHandler = (() => {
    let videoElement = null;
    let canvasElement = null;
    let canvasContext = null;
    let streamActive = false;
    let cameraSettings = null;
    let cameraCheckInterval = null;
    
    // Initialize camera
    const init = async (videoId, canvasId) => {
        videoElement = document.getElementById(videoId);
        canvasElement = document.getElementById(canvasId);
        
        if (!videoElement || !canvasElement) {
            throw new Error('Video or canvas element not found');
        }
        
        canvasContext = canvasElement.getContext('2d');
        
        try {
            // Enhanced camera constraints for better hand detection
            const constraints = {
                video: {
                    width: { ideal: 1280 },  // Higher resolution for better detail
                    height: { ideal: 720 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 },  // Higher frame rate for smoother capture
                    advanced: [
                        { exposureMode: 'auto' },
                        { focusMode: 'continuous' },  // Auto-focus to keep hand in focus
                        { whiteBalanceMode: 'continuous' }  // Better color consistency
                    ]
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            
            // Store camera track for settings access
            const videoTrack = stream.getVideoTracks()[0];
            cameraSettings = videoTrack.getSettings();
            
            console.log('Camera initialized with settings:', cameraSettings);
            
            // Wait for video to be ready and properly sized
            await new Promise(resolve => {
                videoElement.onloadedmetadata = () => {
                    // Set canvas dimensions to match video
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    streamActive = true;
                    console.log(`Video size: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                    
                    // Start monitoring camera status
                    startCameraMonitoring();
                    
                    resolve();
                };
            });
            
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            return false;
        }
    };
    
    // Capture current frame from video with improved quality
    const captureFrame = () => {
        if (!streamActive) {
            throw new Error('Camera stream not active');
        }
        
        try {
            // Draw current video frame to canvas
            canvasContext.drawImage(
                videoElement, 
                0, 0, 
                canvasElement.width, 
                canvasElement.height
            );
            
            // Apply simple image enhancement
            enhanceImage();
            
            // Get image data as high-quality base64 encoded string
            return canvasElement.toDataURL('image/jpeg', 0.95);  // Higher quality JPEG
        } catch (error) {
            console.error('Error capturing frame:', error);
            // Return a basic frame if enhancement fails
            canvasContext.drawImage(
                videoElement, 
                0, 0, 
                canvasElement.width, 
                canvasElement.height
            );
            return canvasElement.toDataURL('image/jpeg', 0.9);
        }
    };
    
    // Basic image enhancement for better hand recognition
    const enhanceImage = () => {
        try {
            // Get image data
            const imageData = canvasContext.getImageData(
                0, 0, 
                canvasElement.width, 
                canvasElement.height
            );
            
            const data = imageData.data;
            
            // Simple contrast and brightness adjustment
            const contrast = 1.1;  // Slight contrast boost
            const brightness = 5;  // Slight brightness boost
            
            for (let i = 0; i < data.length; i += 4) {
                // Apply contrast and brightness
                data[i] = Math.max(0, Math.min(255, (data[i] - 128) * contrast + 128 + brightness));
                data[i+1] = Math.max(0, Math.min(255, (data[i+1] - 128) * contrast + 128 + brightness));
                data[i+2] = Math.max(0, Math.min(255, (data[i+2] - 128) * contrast + 128 + brightness));
                // Alpha channel remains unchanged
            }
            
            // Put the modified data back
            canvasContext.putImageData(imageData, 0, 0);
        } catch (error) {
            console.error('Image enhancement failed:', error);
            // Continue without enhancement
        }
    };
    
    // Start monitoring camera status in case it disconnects
    const startCameraMonitoring = () => {
        if (cameraCheckInterval) {
            clearInterval(cameraCheckInterval);
        }
        
        cameraCheckInterval = setInterval(() => {
            if (videoElement.srcObject && videoElement.srcObject.getVideoTracks()[0]) {
                const track = videoElement.srcObject.getVideoTracks()[0];
                if (!track.enabled || track.muted || !track.readyState || track.readyState === 'ended') {
                    console.warn('Camera track appears to be disabled or ended');
                    streamActive = false;
                } else {
                    streamActive = true;
                }
            } else {
                streamActive = false;
            }
        }, 2000);
    };
    
    // Stop camera stream and monitoring
    const stopStream = () => {
        if (cameraCheckInterval) {
            clearInterval(cameraCheckInterval);
            cameraCheckInterval = null;
        }
        
        if (videoElement && videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            streamActive = false;
        }
    };
    
    // Check if camera is active
    const isActive = () => streamActive;
    
    // Get camera information
    const getCameraInfo = () => cameraSettings;
    
    return {
        init,
        captureFrame,
        stopStream,
        isActive,
        getCameraInfo
    };
})();
