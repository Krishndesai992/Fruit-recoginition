"""
Fruit Recognition Web Application
==================================
Mumbai University Sem 7 Computer Engineering Project
Architecture: ResNet50 Deep Feature Embeddings (2048-D) + Linear SVM (C=0.01)
"""

import os
import io
import json
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

from src.inference_engine import FruitInferencePipeline

# Initialize Flask app
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Initialize Inference Pipeline
print("Initializing Fruit Recognition Inference Pipeline...")
pipeline = FruitInferencePipeline(model_path=str(BASE_DIR / 'models' / 'svm_model.pkl'))
print(f"Pipeline ready! Supported classes: {pipeline.class_names}")

# Nutritional & Botanical Intelligence Knowledge Base
FRUIT_METADATA = {
    'apple': {
        'scientific_name': 'Malus domestica',
        'family': 'Rosaceae',
        'emoji': '🍎',
        'color': '#ef4444',
        'accent_tint': 'rgba(239, 68, 68, 0.15)',
        'calories': 52,
        'water_pct': 86,
        'carbs_g': 14.0,
        'fiber_g': 2.4,
        'sugar_g': 10.3,
        'vitamin_c_mg': 4.6,
        'potassium_mg': 107,
        'key_benefits': [
            'Rich in quercetin and pectin soluble fiber',
            'Supports cardiovascular and metabolic health',
            'High antioxidant activity targeting cellular oxidation'
        ],
        'fun_fact': 'Apples float in water because 25% of their volume is actually air!'
    },
    'avocado': {
        'scientific_name': 'Persea americana',
        'family': 'Lauraceae',
        'emoji': '🥑',
        'color': '#84cc16',
        'accent_tint': 'rgba(132, 204, 22, 0.15)',
        'calories': 160,
        'water_pct': 73,
        'carbs_g': 8.5,
        'fiber_g': 6.7,
        'sugar_g': 0.7,
        'vitamin_c_mg': 10.0,
        'potassium_mg': 485,
        'key_benefits': [
            'Exceptional density of heart-healthy oleic monounsaturated fatty acids',
            'Contains more potassium per gram than a standard banana',
            'Enhances carotenoid absorption from accompanying vegetables'
        ],
        'fun_fact': 'Botanically, avocados are classified as single-seeded large berries!'
    },
    'banana': {
        'scientific_name': 'Musa acuminata',
        'family': 'Musaceae',
        'emoji': '🍌',
        'color': '#eab308',
        'accent_tint': 'rgba(234, 179, 8, 0.15)',
        'calories': 89,
        'water_pct': 75,
        'carbs_g': 22.8,
        'fiber_g': 2.6,
        'sugar_g': 12.2,
        'vitamin_c_mg': 8.7,
        'potassium_mg': 358,
        'key_benefits': [
            'Readily metabolizable carbohydrates for endurance performance',
            'High Vitamin B6 content regulating neurotransmitter synthesis',
            'Contains prebiotic resistant starch supporting gut microbiome'
        ],
        'fun_fact': 'Bananas are naturally slightly radioactive because of their potassium-40 isotope content!'
    },
    'cherry': {
        'scientific_name': 'Prunus avium',
        'family': 'Rosaceae',
        'emoji': '🍒',
        'color': '#f43f5e',
        'accent_tint': 'rgba(244, 63, 94, 0.15)',
        'calories': 63,
        'water_pct': 82,
        'carbs_g': 16.0,
        'fiber_g': 2.1,
        'sugar_g': 12.8,
        'vitamin_c_mg': 7.0,
        'potassium_mg': 222,
        'key_benefits': [
            'Packed with anthocyanins that dramatically diminish exercise-induced soreness',
            'One of the few natural dietary sources of bioavailable melatonin',
            'Demonstrated reduction in uric acid levels and gout symptom severity'
        ],
        'fun_fact': 'A single average cherry tree can yield up to 7,000 cherries per harvest season!'
    },
    'kiwi': {
        'scientific_name': 'Actinidia deliciosa',
        'family': 'Actinidiaceae',
        'emoji': '🥝',
        'color': '#10b981',
        'accent_tint': 'rgba(16, 185, 129, 0.15)',
        'calories': 61,
        'water_pct': 83,
        'carbs_g': 14.7,
        'fiber_g': 3.0,
        'sugar_g': 9.0,
        'vitamin_c_mg': 92.7,
        'potassium_mg': 312,
        'key_benefits': [
            'Exceeds citrus fruits in Vitamin C density (over 100% daily value)',
            'Contains actinidain, a proteolytic enzyme that accelerates protein digestion',
            'Improves sleep latency and sleep efficiency in clinical trials'
        ],
        'fun_fact': 'Kiwifruit was originally named "Chinese Gooseberry" before being branded in New Zealand!'
    },
    'mango': {
        'scientific_name': 'Mangifera indica',
        'family': 'Anacardiaceae',
        'emoji': '🥭',
        'color': '#f97316',
        'accent_tint': 'rgba(249, 115, 22, 0.15)',
        'calories': 60,
        'water_pct': 84,
        'carbs_g': 15.0,
        'fiber_g': 1.6,
        'sugar_g': 13.7,
        'vitamin_c_mg': 36.4,
        'potassium_mg': 168,
        'key_benefits': [
            'High concentration of beta-carotene and Vitamin A for ocular health',
            'Contains mangiferin, a potent antioxidant bioactive polyphenol',
            'Digestive amylase enzymes break down complex carbohydrates smoothly'
        ],
        'fun_fact': 'In India and across South Asia, the mango is reverently celebrated as the "King of Fruits"!'
    },
    'orange': {
        'scientific_name': 'Citrus sinensis',
        'family': 'Rutaceae',
        'emoji': '🍊',
        'color': '#ea580c',
        'accent_tint': 'rgba(234, 88, 12, 0.15)',
        'calories': 47,
        'water_pct': 87,
        'carbs_g': 11.8,
        'fiber_g': 2.4,
        'sugar_g': 9.4,
        'vitamin_c_mg': 53.2,
        'potassium_mg': 181,
        'key_benefits': [
            'Renowned for abundant bioflavonoids like hesperidin supporting vascular endothelium',
            'Supports collagen synthesis and dermal elasticity',
            'Citric acid content helps prevent calcium oxalate kidney stones'
        ],
        'fun_fact': 'Oranges are not a wild fruit; they are an ancient domestic hybrid of pomelo and mandarin!'
    },
    'pineapple': {
        'scientific_name': 'Ananas comosus',
        'family': 'Bromeliaceae',
        'emoji': '🍍',
        'color': '#d97706',
        'accent_tint': 'rgba(217, 119, 6, 0.15)',
        'calories': 50,
        'water_pct': 86,
        'carbs_g': 13.1,
        'fiber_g': 1.4,
        'sugar_g': 9.9,
        'vitamin_c_mg': 47.8,
        'potassium_mg': 109,
        'key_benefits': [
            'Sole commercial dietary source of bromelain enzyme complexes',
            'Powerful post-operative and post-traumatic anti-inflammatory support',
            'High manganese density indispensable for bone matrix mineralization'
        ],
        'fun_fact': 'A pineapple is not a single fruit, but a fusion of dozens of individual berries!'
    },
    'strawberry': {
        'scientific_name': 'Fragaria ananassa',
        'family': 'Rosaceae',
        'emoji': '🍓',
        'color': '#e11d48',
        'accent_tint': 'rgba(225, 29, 72, 0.15)',
        'calories': 32,
        'water_pct': 91,
        'carbs_g': 7.7,
        'fiber_g': 2.0,
        'sugar_g': 4.9,
        'vitamin_c_mg': 58.8,
        'potassium_mg': 153,
        'key_benefits': [
            'Exceptionally low glycemic load with rich polyphenolic profile',
            'Abundant ellagic acid suppressing cellular free-radical cascades',
            'Helps improve glycemic response when paired with carbohydrate meals'
        ],
        'fun_fact': 'Strawberries are the only fruit that wear their seeds on the outside—around 200 per berry!'
    },
    'watermelon': {
        'scientific_name': 'Citrullus lanatus',
        'family': 'Cucurbitaceae',
        'emoji': '🍉',
        'color': '#059669',
        'accent_tint': 'rgba(5, 150, 105, 0.15)',
        'calories': 30,
        'water_pct': 92,
        'carbs_g': 7.6,
        'fiber_g': 0.4,
        'sugar_g': 6.2,
        'vitamin_c_mg': 8.1,
        'potassium_mg': 112,
        'key_benefits': [
            'Supreme cellular hydration via 92% bio-filtered structured water',
            'Richest natural source of lycopene (even higher than ripe tomatoes)',
            'Supplies L-citrulline which boosts nitric oxide synthesis and arterial blood flow'
        ],
        'fun_fact': 'Every part of a watermelon is edible, including the rind and the seeds!'
    }
}

# Sample test image paths for 1-click live testing
SAMPLE_IMAGE_PATHS = {
    'apple': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'apple' / 'img_1001.jpeg',
    'avocado': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'avocado' / 'img_1001.jpeg',
    'banana': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'banana' / 'img_01.jpeg',
    'cherry': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'cherry' / 'img_1001.jpeg',
    'kiwi': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'kiwi' / 'img_1001.jpeg',
    'mango': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'mango' / 'img_101.jpeg',
    'orange': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'orange' / 'img_01.jpeg',
    'pineapple': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'pinenapple' / 'img_1001.jpeg',
    'strawberry': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'stawberries' / 'img_1001.jpeg',
    'watermelon': BASE_DIR / 'data' / 'raw' / 'MY_data' / 'test' / 'watermelon' / 'img_1011.jpeg'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    Accepts image file or base64 data and returns full inference diagnostics.
    """
    image = None
    filename = "uploaded_image"

    try:
        if 'image' in request.files:
            file = request.files['image']
            filename = file.filename or "uploaded_image"
            image = Image.open(file.stream).convert('RGB')
        elif request.is_json:
            data = request.get_json()
            if 'image_base64' in data:
                b64_str = data['image_base64']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                filename = data.get('filename', 'webcam_capture.jpg')

        if image is None:
            return jsonify({'error': 'No image provided. Please upload a file or base64 image.'}), 400

        # Run inference through pipeline
        results = pipeline.predict(image, top_k=5)
        pred_class = results['predicted_class']

        # Attach fruit intelligence metadata
        fruit_info = FRUIT_METADATA.get(pred_class, {
            'scientific_name': 'Unknown',
            'family': 'Unknown',
            'emoji': '🍎',
            'color': '#10b981',
            'calories': 0,
            'key_benefits': [],
            'fun_fact': ''
        })

        results['filename'] = filename
        results['fruit_info'] = fruit_info

        # Enrich top classes with emojis and colors
        for c in results['top_classes']:
            meta = FRUIT_METADATA.get(c['class'], {})
            c['emoji'] = meta.get('emoji', '🍎')
            c['color'] = meta.get('color', '#10b981')

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """
    Returns verified sample fruit catalog with metadata.
    """
    catalog = []
    for fruit_name in pipeline.class_names:
        meta = FRUIT_METADATA.get(fruit_name, {})
        has_file = fruit_name in SAMPLE_IMAGE_PATHS and os.path.exists(SAMPLE_IMAGE_PATHS[fruit_name])
        catalog.append({
            'fruit': fruit_name,
            'display_name': fruit_name.capitalize(),
            'emoji': meta.get('emoji', '🍎'),
            'color': meta.get('color', '#10b981'),
            'image_url': f'/api/sample-image/{fruit_name}' if has_file else None
        })
    return jsonify({'samples': catalog})


@app.route('/api/sample-image/<fruit_name>', methods=['GET'])
def serve_sample_image(fruit_name):
    """
    Serves test image for quick 1-click preview and testing.
    """
    clean_name = fruit_name.lower().strip()
    if clean_name in SAMPLE_IMAGE_PATHS:
        img_path = SAMPLE_IMAGE_PATHS[clean_name]
        if os.path.exists(img_path):
            return send_from_directory(str(img_path.parent), img_path.name)
    return jsonify({'error': 'Sample image not found'}), 404


@app.route('/api/metrics', methods=['GET'])
def get_project_metrics():
    """
    Returns Mumbai University Sem 7 project evaluation statistics.
    """
    metrics_report_path = BASE_DIR / 'results' / 'metrics_report.txt'
    csv_path = BASE_DIR / 'results' / 'backbone_kernel_comparison.csv'

    report_text = ""
    if os.path.exists(metrics_report_path):
        with open(metrics_report_path, 'r') as f:
            report_text = f.read()

    backbone_comparisons = [
        {
            'backbone': 'ResNet50',
            'feature_dim': 2048,
            'kernel': 'Linear',
            'C': 1.0,
            'test_accuracy': 93.79,
            'macro_f1': 0.9387,
            'is_selected': True
        },
        {
            'backbone': 'ResNet50',
            'feature_dim': 2048,
            'kernel': 'RBF',
            'C': 1.0,
            'test_accuracy': 87.98,
            'macro_f1': 0.8883,
            'is_selected': False
        },
        {
            'backbone': 'MobileNetV2',
            'feature_dim': 1280,
            'kernel': 'Linear',
            'C': 1.0,
            'test_accuracy': 94.79,
            'macro_f1': 0.9482,
            'is_selected': False
        },
        {
            'backbone': 'MobileNetV2',
            'feature_dim': 1280,
            'kernel': 'RBF',
            'C': 1.0,
            'test_accuracy': 91.58,
            'macro_f1': 0.9189,
            'is_selected': False
        }
    ]

    return jsonify({
        'project_title': 'Fruit Recognition using Deep Features & Support Vector Machines',
        'university': 'Mumbai University',
        'course': 'Bachelor of Engineering (Computer Engineering) - Semester VII',
        'test_accuracy': 93.79,
        'macro_precision': 0.9407,
        'macro_recall': 0.9381,
        'macro_f1': 0.9387,
        'classes_count': 10,
        'test_images_count': 499,
        'backbone': 'ResNet-50 (ImageNet Pretrained)',
        'feature_dim': 2048,
        'classifier': 'Linear Support Vector Classifier (C=0.01)',
        'comparisons': backbone_comparisons,
        'confusion_matrix_url': '/results/confusion_matrix.png',
        'backbone_plot_url': '/results/backbone_comparison.png',
        'raw_report': report_text
    })


@app.route('/results/<path:filename>')
def serve_results_asset(filename):
    return send_from_directory(str(BASE_DIR / 'results'), filename)


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    print("=========================================================")
    print("[*] Starting Fruit Recognition Web Server on http://127.0.0.1:5000")
    print("[*] Mumbai University Sem 7 Computer Engineering Project")
    print("=========================================================")
    app.run(host='127.0.0.1', port=5000, debug=False)

