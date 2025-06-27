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
            context.drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(function (blob) {
                const formData = new FormData();
                formData.append('image', blob, 'snapshot.jpg');

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
            }, 'image/jpeg');
        };

        img.src = event.target.result;
    };

    reader.readAsDataURL(file);
});