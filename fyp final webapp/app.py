import os
import logging
import traceback
import socket
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
CORS(app)

# Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join('static', 'uploads'),
    RESULTS_FOLDER=os.path.join('static', 'results'),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'default-secret-key'),
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'webp'},
    TEMPLATES_AUTO_RELOAD=True
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Damage severity classes
CLASS_NAMES = ['minor', 'moderate', 'severe']

def configure_detectron():
    """Configure Detectron2 model for damage detection"""
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.abspath('model_final.pth')
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        MetadataCatalog.get("damage_dataset").set(
            thing_classes=["damage"],
            thing_colors=[(0, 255, 0)]
        )
        
        logger.info(f"Detectron2 configured to use {cfg.MODEL.DEVICE}")
        return cfg, MetadataCatalog.get("damage_dataset")
    except Exception as e:
        logger.error(f"Detectron2 config error: {str(e)}\n{traceback.format_exc()}")
        raise

try:
    cfg, metadata = configure_detectron()
    detection_predictor = DefaultPredictor(cfg)
    classification_model = tf.keras.models.load_model(
        os.path.abspath('final_model.keras'),
        compile=False
    )
    logger.info("Computer vision models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}\n{traceback.format_exc()}")
    raise

class DamageAnalyzer:
    """Handles damage analysis using lightweight Zephyr-7B model"""
    def __init__(self):
        try:
            # Initialize BLIP model for image understanding
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Configure 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Initialize Zephyr-7B-alpha (openly available)
            model_name = "HuggingFaceH4/zephyr-7b-alpha"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            # Create optimized text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            if torch.cuda.is_available():
                self.image_caption_model = self.image_caption_model.to('cuda')
            
            logger.info(f"AI models loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            logger.error(f"Failed to load AI models: {str(e)}\n{traceback.format_exc()}")
            raise

    def generate_damage_report(self, original_image_path, detection_image_path, damage_class, confidence):
        """Generate comprehensive damage report"""
        try:
            # Calculate basic metrics
            damage_area = self._calculate_damage_area(detection_image_path)
            damage_locations = self._identify_damage_locations(detection_image_path)
            
            # Get image description
            img_desc = self._get_image_description(detection_image_path)
            
            # Generate detailed report
            report = self._generate_llm_report(
                damage_class=damage_class,
                confidence=confidence,
                damage_area=damage_area,
                damage_locations=damage_locations,
                image_description=img_desc
            )
            
            return self._format_final_report(report, damage_class, confidence)
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}\n{traceback.format_exc()}")
            return self._generate_fallback_report(damage_class, confidence)

    def _calculate_damage_area(self, detection_image_path):
        """Calculate percentage of damaged area"""
        img = cv2.imread(detection_image_path, 0)
        if img is None:
            return 0
        
        _, thresholded = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        damage_pixels = cv2.countNonZero(thresholded)
        total_pixels = img.size
        return (damage_pixels / total_pixels) * 100

    def _identify_damage_locations(self, detection_image_path):
        """Identify general locations of damage"""
        try:
            img = Image.open(detection_image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt").to(self.image_caption_model.device)
            
            prompt = "Identify the exact vehicle parts with damage from these options: " \
                    "front bumper, hood, fender, door, roof, quarter panel, rear bumper, trunk. " \
                    "Respond with only the damaged parts."
            
            outputs = self.image_caption_model.generate(**inputs,
                                              max_length=100,
                                              num_beams=4,
                                              early_stopping=True)
            
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Damage location identification failed: {str(e)}")
            return "multiple areas"

    def _get_image_description(self, image_path):
        """Get basic description of the damage image"""
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt").to(self.image_caption_model.device)
            outputs = self.image_caption_model.generate(**inputs, max_length=150)
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Image description failed: {str(e)}")
            return "vehicle damage"

    def _generate_llm_report(self, damage_class, confidence, damage_area, damage_locations, image_description):
        """Generate detailed report using LLM"""
        prompt = f"""
        Generate a professional vehicle damage assessment report with these specifications:

        VEHICLE DAMAGE DETAILS:
        - Severity Classification: {damage_class}
        - Detection Confidence: {confidence:.2f}%
        - Damage Area Percentage: {damage_area:.2f}%
        - Affected Locations: {damage_locations}
        - Visual Description: {image_description}

        REPORT STRUCTURE:
        1. DAMAGE SUMMARY:
           - Overall severity assessment
           - Primary affected components
           - Visible damage characteristics

        2. DAMAGE LOCATION ANALYSIS:
           - Detailed breakdown by vehicle area
           - Specific damage types per location
           - Estimated repair complexity for each

        3. REPAIR RECOMMENDATIONS:
           - Required repair procedures
           - Estimated repair time
           - Special considerations

        4. COST ESTIMATION:
           - Approximate cost range
           - Cost factors
           - Potential additional costs

        Use technical automotive terminology.
        Be specific and quantitative where possible.
        Format with clear section headers and bullet points.
        """
        
        response = self.text_generator(
            prompt,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        return response[0]['generated_text']

    def _format_final_report(self, raw_report, damage_class, confidence):
        """Format the final report with headers"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""
VEHICLE DAMAGE ASSESSMENT REPORT
================================
Generated on: {timestamp}
Severity: {damage_class.title()} (Confidence: {confidence:.2f}%)

{raw_report}

IMPORTANT NOTE:
This is an AI-generated preliminary assessment. For accurate evaluation,
please consult a certified automotive professional.
"""

    def _generate_fallback_report(self, damage_class, confidence):
        """Generate basic report when detailed analysis fails"""
        return f"""
BASIC DAMAGE ASSESSMENT
-----------------------
Damage Severity: {damage_class.title()}
Confidence: {confidence:.2f}%

Note: Detailed analysis unavailable. Please consult a professional
for comprehensive assessment.
"""

# Initialize analyzer
try:
    analyzer = DamageAnalyzer()
    logger.info("Damage analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize damage analyzer: {str(e)}")
    raise

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Process uploaded image and generate damage analysis"""
    try:
        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image file")
        
        # Convert color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run damage detection
        outputs = detection_predictor(img_rgb)
        
        # Visualize results
        visualizer = Visualizer(
            img_rgb,
            metadata=metadata,
            scale=1.2,
            instance_mode=ColorMode.IMAGE_BW
        )
        instances = outputs["instances"].to("cpu")
        vis_output = visualizer.draw_instance_predictions(instances)
        detection_img = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)
        
        # Save detection result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_filename = f"detected_{timestamp}_{os.path.basename(image_path)}"
        detection_path = os.path.join(app.config['RESULTS_FOLDER'], detection_filename)
        cv2.imwrite(detection_path, detection_img)
        
        # Run damage classification
        classification_img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(classification_img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        predictions = classification_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        
        if predictions.size == 0 or len(predictions[0]) != len(CLASS_NAMES):
            raise ValueError("Invalid model predictions")
        
        confidence = np.max(predictions[0])
        class_index = np.argmax(predictions[0])
        damage_class = CLASS_NAMES[class_index]
        
        # Generate enhanced report
        report_text = analyzer.generate_damage_report(
            original_image_path=image_path,
            detection_image_path=detection_path,
            damage_class=damage_class,
            confidence=float(confidence) * 100
        )
        
        return detection_path, {
            'class': damage_class,
            'confidence': confidence,
            'report': report_text
        }
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}\n{traceback.format_exc()}")
        raise

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render contact page"""
    return render_template('contact.html')

@app.route('/results')
def results():
    """Render results page"""
    try:
        original = request.args.get('original')
        detection_image = request.args.get('detection_image')
        damage_type = request.args.get('damage_type')
        confidence = request.args.get('confidence')
        severity = request.args.get('severity')
        report = request.args.get('report')
        
        if not all([original, detection_image, damage_type, confidence, severity, report]):
            logger.warning("Missing parameters in results request")
            return redirect(url_for('index'))
        
        return render_template('results.html',
                       original=original,
                       detection_image=detection_image,
                       damage_type=damage_type,
                       confidence=confidence,
                       severity=severity,
                       report=report,
                       vehicle_type="car")
    except Exception as e:
        logger.error(f"Error showing results: {str(e)}")
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Handle image upload and analysis"""
    if 'image' not in request.files:
        logger.warning("No image in upload request")
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        logger.warning("Empty filename in upload")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process image
        detection_path, damage_info = process_image(upload_path)
        
        return jsonify({
            'success': True,
            'original': upload_path.replace('\\', '/'),
            'detection_image': detection_path.replace('\\', '/'),
            'damage_type': damage_info['class'],
            'confidence': float(damage_info['confidence']),
            'severity': damage_info['class'],
            'report': damage_info['report']
        })
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    try:
        # Get network information
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("\n" + "="*50)
        print("Starting CarDamagePro Server")
        print("="*50)
        print(f"- Local URL:    http://127.0.0.1:5000")
        print(f"- Network URL:  http://{local_ip}:5000")
        print(f"- Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
        print(f"- Results folder: {os.path.abspath(app.config['RESULTS_FOLDER'])}")
        print("="*50 + "\n")
        
        # Run the server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}\n{traceback.format_exc()}")