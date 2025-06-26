from flask import Blueprint, request, jsonify
from PIL import Image
import io

from services.inference import predict_digit

predict_bp = Blueprint('predict_bp', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict_route():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(io.BytesIO(file.read()))
    predicted_digit = predict_digit(image)

    return jsonify({'prediction': int(predicted_digit)})
