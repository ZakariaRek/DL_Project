# ========================================
# 🍊 FASTAPI BACKEND - ORANGE DISEASE DETECTION
# Updated with Video & Image Export
# ========================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os
import tempfile
import uuid
from datetime import datetime
from collections import defaultdict
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIGURATION ====================
class Config:
    MODEL_PATH = 'shufflenet_v2_orange_diseases.pth'
    YOLO_MODEL = 'yolov8n.pt'
    IMG_SIZE = 224
    CLASS_NAMES = ['blackspot', 'canker', 'fresh', 'grenning']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIDENCE_THRESHOLD = 0.25
    CLASSIFICATION_THRESHOLD = 0.60
    FRAME_SKIP = 3
    UPLOAD_DIR = 'uploads'
    RESULTS_DIR = 'results'

# Create directories
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="🍊 Orange Disease Detection API",
    description="AI-powered system for detecting diseases in oranges using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving results
app.mount("/files", StaticFiles(directory=Config.RESULTS_DIR), name="files")

# ==================== MODELS ====================
class DiseaseClassifier:
    def __init__(self):
        self.model = None
        self.yolo_model = None
        self.transform = None
        self.load_models()
    
    def load_models(self):
        """Load ShuffleNet V2 and YOLO models"""
        print("🔧 Loading models...")
        
        # Load classification model
        self.model = models.shufflenet_v2_x1_0(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, len(Config.CLASS_NAMES))
        )
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        )
        self.model.to(Config.DEVICE)
        self.model.eval()
        
        # Load YOLO model
        self.yolo_model = YOLO(Config.YOLO_MODEL)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Models loaded successfully!")
    
    def classify_image(self, image: Image.Image) -> Dict:
        """Classify disease on a single orange image"""
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = Config.CLASS_NAMES[predicted_idx.item()]
            confidence_pct = confidence.item() * 100
            all_probs = probabilities.cpu().numpy()[0] * 100
            
            return {
                'disease': predicted_class,
                'confidence': float(confidence_pct),
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(Config.CLASS_NAMES, all_probs)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
    
    def create_visualization(self, results: Dict, output_path: str):
        """Create analysis visualization plots"""
        if results['total_oranges'] == 0:
            return None
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        colors = ['#FF4444', '#FF9944', '#44FF44', '#44FFFF']
        
        # 1. Pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = {d: len(c) for d, c in results['classifications'].items()}
        ax1.pie(class_counts.values(), labels=class_counts.keys(),
               autopct='%1.1f%%', colors=colors[:len(class_counts)], startangle=90)
        ax1.set_title('Disease Distribution', fontsize=14, fontweight='bold')
        
        # 2. Bar chart
        ax2 = fig.add_subplot(gs[0, 1:])
        bars = ax2.bar(class_counts.keys(), class_counts.values(), 
                      color=colors[:len(class_counts)], alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Detections per Disease Class', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, class_counts.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 3. Box plot
        ax3 = fig.add_subplot(gs[1, :])
        data = [results['classifications'][d] for d in Config.CLASS_NAMES 
                if d in results['classifications']]
        labels = [d for d in Config.CLASS_NAMES 
                 if d in results['classifications']]
        
        if data:
            bp = ax3.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_linewidth(2)
        
        ax3.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Confidence Distribution by Disease', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Timeline
        ax4 = fig.add_subplot(gs[2, :])
        
        for i, disease in enumerate(Config.CLASS_NAMES):
            disease_detections = [d for d in results['detections'] if d['disease'] == disease]
            if disease_detections:
                times = [d['time_sec'] for d in disease_detections]
                ax4.scatter(times, [i]*len(times),
                          c=colors[i], s=100, alpha=0.6, label=disease, edgecolors='black')
        
        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_yticks(range(len(Config.CLASS_NAMES)))
        ax4.set_yticklabels(Config.CLASS_NAMES)
        ax4.set_title('Detection Timeline', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def detect_and_classify_video(self, video_path: str, file_id: str) -> Dict:
        """Process video and detect/classify oranges with annotated output"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = {
            'detections': [],
            'classifications': defaultdict(list),
            'total_oranges': 0,
            'processed_frames': 0,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps if fps > 0 else 0
            }
        }
        
        # Setup video writer for annotated output
        output_video_path = os.path.join(Config.RESULTS_DIR, f"{file_id}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Color map for diseases
        color_map = {
            'blackspot': (0, 0, 255),      # Red
            'canker': (0, 140, 255),       # Orange
            'fresh': (0, 255, 0),          # Green
            'grenning': (255, 255, 0)      # Cyan
        }
        
        frame_idx = 0
        detection_id = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % Config.FRAME_SKIP == 0:
                    # Detect fruits with YOLO
                    yolo_results = self.yolo_model(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
                    
                    for result in yolo_results:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            
                            # Filter for fruit classes
                            if class_id in [46, 47, 49] or 'orange' in class_name.lower():
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Filter small detections
                                if (x2 - x1) > 30 and (y2 - y1) > 30:
                                    # Crop and classify
                                    fruit_crop = frame[y1:y2, x1:x2]
                                    
                                    if fruit_crop.size > 0:
                                        # Convert to PIL Image
                                        fruit_rgb = cv2.cvtColor(fruit_crop, cv2.COLOR_BGR2RGB)
                                        pil_image = Image.fromarray(fruit_rgb)
                                        
                                        # Classify disease
                                        classification = self.classify_image(pil_image)
                                        
                                        if classification['confidence'] >= Config.CLASSIFICATION_THRESHOLD:
                                            results['detections'].append({
                                                'id': detection_id,
                                                'frame': frame_idx,
                                                'time_sec': float(frame_idx / fps),
                                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                                'disease': classification['disease'],
                                                'confidence': classification['confidence']
                                            })
                                            
                                            results['classifications'][classification['disease']].append(
                                                classification['confidence']
                                            )
                                            results['total_oranges'] += 1
                                            detection_id += 1
                                            
                                            # Draw on frame
                                            disease = classification['disease']
                                            color = color_map.get(disease, (255, 255, 255))
                                            
                                            # Draw bounding box
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                            
                                            # Draw label
                                            label = f"{disease}: {classification['confidence']:.0f}%"
                                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                                        (x1 + label_size[0], y1), color, -1)
                                            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    results['processed_frames'] += 1
                
                # Write annotated frame
                out.write(frame)
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
        
        # Calculate statistics
        statistics = {}
        for disease, confidences in results['classifications'].items():
            statistics[disease] = {
                'count': len(confidences),
                'percentage': (len(confidences) / results['total_oranges'] * 100) if results['total_oranges'] > 0 else 0,
                'avg_confidence': float(np.mean(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences))
            }
        
        results['statistics'] = statistics
        
        # Calculate health score
        fresh_count = len(results['classifications'].get('fresh', []))
        results['health_score'] = (fresh_count / results['total_oranges'] * 100) if results['total_oranges'] > 0 else 0
        
        # Convert defaultdict to dict
        results['classifications'] = dict(results['classifications'])
        
        # Create visualization
        viz_path = os.path.join(Config.RESULTS_DIR, f"{file_id}_analysis.png")
        self.create_visualization(results, viz_path)
        
        # Add file paths to results
        results['annotated_video_url'] = f"/files/{file_id}_annotated.mp4"
        results['analysis_image_url'] = f"/files/{file_id}_analysis.png"
        
        return results

# Initialize classifier
classifier = DiseaseClassifier()

# ==================== RESPONSE MODELS ====================
class ImageClassificationResponse(BaseModel):
    disease: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    status: str

class VideoAnalysisResponse(BaseModel):
    total_oranges: int
    processed_frames: int
    health_score: float
    statistics: Dict[str, Dict]
    detections: List[Dict]
    video_info: Dict
    annotated_video_url: str
    analysis_image_url: str
    timestamp: str
    status: str

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🍊 Orange Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "image": "/api/classify-image",
            "video": "/api/analyze-video",
            "health": "/health"
        },
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(Config.DEVICE),
        "models_loaded": classifier.model is not None and classifier.yolo_model is not None
    }

@app.post("/api/classify-image", response_model=ImageClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Classify disease on a single orange image
    
    - **file**: Image file (JPG, PNG, JPEG)
    - Returns: Disease classification with confidence scores
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = classifier.classify_image(image)
        result['timestamp'] = datetime.now().isoformat()
        result['status'] = 'success'
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze video and detect oranges with disease classification
    
    - **file**: Video file (MP4, AVI, MOV)
    - Returns: Comprehensive analysis report with annotated video and visualization
    """
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(Config.UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        # Save uploaded video
        contents = await file.read()
        with open(video_path, 'wb') as f:
            f.write(contents)
        
        # Process video
        result = classifier.detect_and_classify_video(video_path, file_id)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['status'] = 'success'
        result['file_id'] = file_id
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)

@app.get("/api/download/{file_id}/{file_type}")
async def download_file(file_id: str, file_type: str):
    """
    Download generated files (annotated video or analysis image)
    
    - **file_id**: Unique file identifier from video analysis
    - **file_type**: 'video' or 'image'
    """
    if file_type == 'video':
        file_path = os.path.join(Config.RESULTS_DIR, f"{file_id}_annotated.mp4")
        media_type = "video/mp4"
    elif file_type == 'image':
        file_path = os.path.join(Config.RESULTS_DIR, f"{file_id}_analysis.png")
        media_type = "image/png"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type=media_type)

@app.post("/api/batch-classify")
async def batch_classify_images(files: List[UploadFile] = File(...)):
    """
    Classify multiple orange images at once
    
    - **files**: List of image files
    - Returns: List of classifications for each image
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    
    for idx, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            classification = classifier.classify_image(image)
            
            results.append({
                'filename': file.filename,
                'index': idx,
                **classification
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'index': idx,
                'error': str(e)
            })
    
    return {
        'total_images': len(files),
        'processed': len(results),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Orange Disease Detection API...")
    print(f"📍 Device: {Config.DEVICE}")
    print(f"🔧 Models loaded from: {Config.MODEL_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)