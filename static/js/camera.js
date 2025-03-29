// Camera handling module
const CameraHandler = (() => {
    let videoElement = null;
    let canvasElement = null;
    let canvasContext = null;
    let streamActive = false;
    
    // Initialize camera
    const init = async (videoId, canvasId) => {
        videoElement = document.getElementById(videoId);
        canvasElement = document.getElementById(canvasId);
        
        if (!videoElement || !canvasElement) {
            throw new Error('Video or canvas element not found');
        }
        
        canvasContext = canvasElement.getContext('2d');
        
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                videoElement.onloadedmetadata = () => {
                    // Set canvas dimensions to match video
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    streamActive = true;
                    resolve();
                };
            });
            
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            return false;
        }
    };
    
    // Capture current frame from video
    const captureFrame = () => {
        if (!streamActive) {
            throw new Error('Camera stream not active');
        }
        
        // Draw current video frame to canvas
        canvasContext.drawImage(
            videoElement, 
            0, 0, 
            canvasElement.width, 
            canvasElement.height
        );
        
        // Get image data as base64 encoded string
        return canvasElement.toDataURL('image/jpeg', 0.9);
    };
    
    // Stop camera stream
    const stopStream = () => {
        if (videoElement && videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            streamActive = false;
        }
    };
    
    // Check if camera is active
    const isActive = () => streamActive;
    
    return {
        init,
        captureFrame,
        stopStream,
        isActive
    };
})();
