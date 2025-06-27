let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let snapButton = document.getElementById('snap');

// Access the camera
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        });
}

// Snap button click
snapButton.addEventListener("click", function () {
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Blob and send via FormData
    canvas.toBlob(function (blob) {
        const formData = new FormData();
        formData.append('image', blob, 'snapshot.jpg'); // Key must match Flask: 'image'

        fetch('http://localhost:5002/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
               console.log("Prediction:", data.prediction);
                document.getElementById('result').textContent = "Prediction: " + data.prediction.toString();
                //alert("Predicted Digit: " + data.data);
            })
            .catch(error => {
                console.error("Upload error:", error);
                alert("Error uploading image.");
            });
    }, 'image/jpeg');
});
