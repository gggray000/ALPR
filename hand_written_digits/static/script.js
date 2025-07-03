let fileInput = document.getElementById('fileInput');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');

fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = function (event) {
        const img = new Image();
        img.onload = function () {
            context.drawImage(img, 0, 0, 640, 480);
        
            const formData = new FormData();
            formData.append('image', file); 

            fetch("/alpr/predict", {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log("Prediction:", data.prediction);
                document.getElementById('result').textContent = "Prediction: " + data.prediction;
            })
            .catch(error => {
                console.error("Upload error:", error);
                alert("Error: ", error);
            });
        };

        img.src = event.target.result;
    };

    reader.readAsDataURL(file);
});