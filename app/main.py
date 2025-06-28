from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import cv2
from PIL import Image
import io
import base64
import numpy as np
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load face detector and recognition model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

# Configuration
MAX_IMAGE_SIZE = 1024  # Maximum dimension for image preprocessing

# ---------- Models ----------
class FaceDetectionMetrics(BaseModel):
    face_detected: bool
    detection_time_ms: float
    face_coordinates: Optional[List[float]] = None
    confidence_score: Optional[float] = None

class EmbeddingMetrics(BaseModel):
    embedding_time_ms: float
    embedding_dimension: int
    embedding_norm: float
    embedding_stats: Dict[str, float]  # min, max, mean, std

class ComparisonMetrics(BaseModel):
    similarity_computation_time_ms: float
    cosine_similarity: float
    euclidean_distance: float
    threshold_used: float

class CompareRequest(BaseModel):
    image1_base64: str
    image2_base64: str

class CompareResponse(BaseModel):
    similarity: float
    match: bool
    total_processing_time_ms: float
    image1_metrics: FaceDetectionMetrics
    image2_metrics: FaceDetectionMetrics
    embedding1_metrics: EmbeddingMetrics
    embedding2_metrics: EmbeddingMetrics
    comparison_metrics: ComparisonMetrics

# ---------- Utils ----------
def resize_image_if_needed(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image if it's larger than max_size while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def decode_image(base64_string: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize image for better performance
        return resize_image_if_needed(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

async def get_face_with_metrics_async(image: Image.Image) -> tuple[torch.Tensor, FaceDetectionMetrics]:
    """Async wrapper for face detection"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_face_with_metrics, image)

def get_face_with_metrics(image: Image.Image) -> tuple[torch.Tensor, FaceDetectionMetrics]:
    start_time = time.time()
    
    # Get face detection with additional information
    face, prob = mtcnn(image, return_prob=True)
    
    detection_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    if face is None:
        return None, FaceDetectionMetrics(
            face_detected=False,
            detection_time_ms=detection_time,
            face_coordinates=None,
            confidence_score=None
        )
    
    # Get bounding box information
    boxes, _ = mtcnn.detect(image)
    face_coords = boxes[0].tolist() if boxes is not None and len(boxes) > 0 else None
    
    metrics = FaceDetectionMetrics(
        face_detected=True,
        detection_time_ms=detection_time,
        face_coordinates=face_coords,
        confidence_score=float(prob) if prob is not None else None
    )
    
    return face, metrics

def get_embeddings_batch_with_metrics(faces: List[torch.Tensor]) -> tuple[List[torch.Tensor], List[EmbeddingMetrics]]:
    """Process multiple faces in a single batch for better GPU utilization"""
    if not faces:
        return [], []
    
    start_time = time.time()
    
    # Stack faces into a batch
    batch_faces = torch.stack(faces).to(device)
    
    with torch.no_grad():
        batch_embeddings = resnet(batch_faces)
    
    embedding_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Split batch results and calculate individual metrics
    embeddings = []
    metrics_list = []
    
    for i, embedding in enumerate(batch_embeddings):
        embedding_np = embedding.cpu().numpy().flatten()
        embedding_norm = float(torch.norm(embedding).item())
        
        embedding_stats = {
            "min": float(np.min(embedding_np)),
            "max": float(np.max(embedding_np)),
            "mean": float(np.mean(embedding_np)),
            "std": float(np.std(embedding_np))
        }
        
        metrics = EmbeddingMetrics(
            embedding_time_ms=embedding_time / len(faces),  # Distribute batch time
            embedding_dimension=embedding.shape[0],
            embedding_norm=embedding_norm,
            embedding_stats=embedding_stats
        )
        
        embeddings.append(embedding.unsqueeze(0))
        metrics_list.append(metrics)
    
    return embeddings, metrics_list

async def get_embedding_with_metrics_async(face: torch.Tensor) -> tuple[torch.Tensor, EmbeddingMetrics]:
    """Async wrapper for single embedding processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_embedding_with_metrics_single, face)

def get_embedding_with_metrics_single(face: torch.Tensor) -> tuple[torch.Tensor, EmbeddingMetrics]:
    """Single face embedding processing"""
    embeddings, metrics = get_embeddings_batch_with_metrics([face])
    return embeddings[0], metrics[0]

def get_embedding_with_metrics(face: torch.Tensor) -> tuple[torch.Tensor, EmbeddingMetrics]:
    """Backwards compatibility wrapper"""
    return get_embedding_with_metrics_single(face)

async def compute_similarity_with_metrics_async(emb1: torch.Tensor, emb2: torch.Tensor, threshold: float = 0.6) -> ComparisonMetrics:
    """Async wrapper for similarity computation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, compute_similarity_with_metrics, emb1, emb2, threshold)

def compute_similarity_with_metrics(emb1: torch.Tensor, emb2: torch.Tensor, threshold: float = 0.6) -> ComparisonMetrics:
    start_time = time.time()
    
    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    # Euclidean distance
    euclidean_dist = torch.norm(emb1 - emb2).item()
    
    computation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return ComparisonMetrics(
        similarity_computation_time_ms=computation_time,
        cosine_similarity=cosine_sim,
        euclidean_distance=euclidean_dist,
        threshold_used=threshold
    )

# ---------- MediaPipe Utils ----------
def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

async def get_face_with_mediapipe_async(image: Image.Image) -> tuple[torch.Tensor, FaceDetectionMetrics]:
    """Async wrapper for MediaPipe face detection"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_face_with_mediapipe, image)

def get_face_with_mediapipe(image: Image.Image) -> tuple[torch.Tensor, FaceDetectionMetrics]:
    start_time = time.time()
    
    # Convert PIL to OpenCV format
    cv2_image = pil_to_cv2(image)
    
    # Make sure we're working with RGB format for MediaPipe
    # MediaPipe expects RGB format, so we convert directly without another conversion
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe needs valid images with correct shape and dtype
    if rgb_image.shape[0] == 0 or rgb_image.shape[1] == 0:
        # Return None if image dimensions are invalid
        return None, FaceDetectionMetrics(
            face_detected=False,
            detection_time_ms=(time.time() - start_time) * 1000,
            face_coordinates=None,
            confidence_score=None
        )
    
    # Ensure the image is in the right format for MediaPipe (uint8)
    rgb_image = rgb_image.astype(np.uint8)
    
    # Detect faces with MediaPipe
    results = face_detection.process(rgb_image)
    
    detection_time = (time.time() - start_time) * 1000
    
    if not results.detections:
        return None, FaceDetectionMetrics(
            face_detected=False,
            detection_time_ms=detection_time,
            face_coordinates=None,
            confidence_score=None
        )
    
    # Get the first (most confident) detection
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    # Convert relative coordinates to absolute
    h, w = rgb_image.shape[:2]
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)
    
    # Add some margin and ensure coordinates are within image bounds
    margin = 20
    x = max(0, x - margin)
    y = max(0, y - margin)
    x2 = min(w, x + width + 2 * margin)
    y2 = min(h, y + height + 2 * margin)
    
    # Extract face region
    face_region = rgb_image[y:y2, x:x2]
    
    if face_region.size == 0:
        return None, FaceDetectionMetrics(
            face_detected=False,
            detection_time_ms=detection_time,
            face_coordinates=None,
            confidence_score=None
        )
    
    # Convert to PIL and resize to 160x160 for compatibility with InceptionResnetV1
    face_pil = Image.fromarray(face_region)
    face_resized = face_pil.resize((160, 160), Image.Resampling.LANCZOS)
    
    # Convert to tensor format expected by the model
    face_tensor = torch.tensor(np.array(face_resized)).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
    
    face_coords = [float(x), float(y), float(x2), float(y2)]
    confidence = float(detection.score[0]) if detection.score else None
    
    metrics = FaceDetectionMetrics(
        face_detected=True,
        detection_time_ms=detection_time,
        face_coordinates=face_coords,
        confidence_score=confidence
    )
    
    return face_tensor, metrics

def get_embeddings_mediapipe_batch_with_metrics(faces: List[torch.Tensor]) -> tuple[List[torch.Tensor], List[EmbeddingMetrics]]:
    """Process multiple MediaPipe faces in a single batch"""
    if not faces:
        return [], []
    
    start_time = time.time()
    
    # Stack faces into a batch
    batch_faces = torch.stack(faces).to(device)
    
    with torch.no_grad():
        batch_embeddings = resnet(batch_faces)
    
    embedding_time = (time.time() - start_time) * 1000
    
    # Split batch results and calculate individual metrics
    embeddings = []
    metrics_list = []
    
    for i, embedding in enumerate(batch_embeddings):
        embedding_np = embedding.cpu().numpy().flatten()
        embedding_norm = float(torch.norm(embedding).item())
        
        embedding_stats = {
            "min": float(np.min(embedding_np)),
            "max": float(np.max(embedding_np)),
            "mean": float(np.mean(embedding_np)),
            "std": float(np.std(embedding_np))
        }
        
        metrics = EmbeddingMetrics(
            embedding_time_ms=embedding_time / len(faces),
            embedding_dimension=embedding.shape[0],
            embedding_norm=embedding_norm,
            embedding_stats=embedding_stats
        )
        
        embeddings.append(embedding.unsqueeze(0))
        metrics_list.append(metrics)
    
    return embeddings, metrics_list

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "Face comparison service is running."}

@app.post("/compare", response_model=CompareResponse)
async def compare_faces(payload: CompareRequest):
    total_start_time = time.time()
    
    # Decode images asynchronously
    loop = asyncio.get_event_loop()
    img1, img2 = await asyncio.gather(
        loop.run_in_executor(executor, decode_image, payload.image1_base64),
        loop.run_in_executor(executor, decode_image, payload.image2_base64)
    )

    # Face detection in parallel
    face_detection_tasks = await asyncio.gather(
        get_face_with_metrics_async(img1),
        get_face_with_metrics_async(img2)
    )
    
    face1, face1_metrics = face_detection_tasks[0]
    face2, face2_metrics = face_detection_tasks[1]
    
    # Check if faces were detected
    if face1 is None:
        raise HTTPException(status_code=422, detail="No face detected in image 1.")
    if face2 is None:
        raise HTTPException(status_code=422, detail="No face detected in image 2.")

    # Process embeddings in batch for better GPU utilization
    embeddings, embedding_metrics = await loop.run_in_executor(
        executor, get_embeddings_batch_with_metrics, [face1, face2]
    )
    
    emb1, emb1_metrics = embeddings[0], embedding_metrics[0]
    emb2, emb2_metrics = embeddings[1], embedding_metrics[1]

    # Compute similarity
    threshold = 0.6  # Adjustable
    comparison_metrics = await compute_similarity_with_metrics_async(emb1, emb2, threshold)
    
    # Final similarity and match decision
    similarity = comparison_metrics.cosine_similarity
    match = similarity > (1 - threshold)
    
    total_processing_time = (time.time() - total_start_time) * 1000  # Convert to milliseconds

    return CompareResponse(
        similarity=similarity,
        match=match,
        total_processing_time_ms=total_processing_time,
        image1_metrics=face1_metrics,
        image2_metrics=face2_metrics,
        embedding1_metrics=emb1_metrics,
        embedding2_metrics=emb2_metrics,
        comparison_metrics=comparison_metrics
    )

@app.post("/mediapipe", response_model=CompareResponse)
async def compare_faces_mediapipe(payload: CompareRequest):
    total_start_time = time.time()
    
    try:
        # Decode images asynchronously
        loop = asyncio.get_event_loop()
        img1, img2 = await asyncio.gather(
            loop.run_in_executor(executor, decode_image, payload.image1_base64),
            loop.run_in_executor(executor, decode_image, payload.image2_base64)
        )

        # Validate image dimensions
        for i, img in enumerate([img1, img2], 1):
            if img.size[0] < 20 or img.size[1] < 20:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Image {i} is too small ({img.size[0]}x{img.size[1]}). Minimum size is 20x20 pixels."
                )

        # Face detection with MediaPipe in parallel
        face_detection_tasks = await asyncio.gather(
            get_face_with_mediapipe_async(img1),
            get_face_with_mediapipe_async(img2),
            return_exceptions=True
        )
        
        # Handle potential exceptions from face detection
        for i, result in enumerate(face_detection_tasks, 1):
            if isinstance(result, Exception):
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing image {i}: {str(result)}"
                )
        
        face1, face1_metrics = face_detection_tasks[0]
        face2, face2_metrics = face_detection_tasks[1]
        
        # Check if faces were detected
        if face1 is None:
            raise HTTPException(status_code=422, detail="No face detected in image 1.")
        if face2 is None:
            raise HTTPException(status_code=422, detail="No face detected in image 2.")
            
        # Process embeddings in batch for better GPU utilization
        embeddings, embedding_metrics = await loop.run_in_executor(
            executor, get_embeddings_mediapipe_batch_with_metrics, [face1, face2]
        )
        
        emb1, emb1_metrics = embeddings[0], embedding_metrics[0]
        emb2, emb2_metrics = embeddings[1], embedding_metrics[1]

        # Compute similarity
        threshold = 0.6  # Adjustable
        comparison_metrics = await compute_similarity_with_metrics_async(emb1, emb2, threshold)
        
        # Final similarity and match decision
        similarity = comparison_metrics.cosine_similarity
        match = similarity > (1 - threshold)
        
        total_processing_time = (time.time() - total_start_time) * 1000  # Convert to milliseconds

        return CompareResponse(
            similarity=similarity,
            match=match,
            total_processing_time_ms=total_processing_time,
            image1_metrics=face1_metrics,
            image2_metrics=face2_metrics,
            embedding1_metrics=emb1_metrics,
            embedding2_metrics=emb2_metrics,
            comparison_metrics=comparison_metrics
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and convert other exceptions to HTTP exceptions
        print(f"Error in /mediapipe endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    # Process embeddings in batch for better GPU utilization
    embeddings, embedding_metrics = await loop.run_in_executor(
        executor, get_embeddings_mediapipe_batch_with_metrics, [face1, face2]
    )
    
    emb1, emb1_metrics = embeddings[0], embedding_metrics[0]
    emb2, emb2_metrics = embeddings[1], embedding_metrics[1]

    # Compute similarity
    threshold = 0.6  # Adjustable
    comparison_metrics = await compute_similarity_with_metrics_async(emb1, emb2, threshold)
    
    # Final similarity and match decision
    similarity = comparison_metrics.cosine_similarity
    match = similarity > (1 - threshold)
    
    total_processing_time = (time.time() - total_start_time) * 1000  # Convert to milliseconds

    return CompareResponse(
        similarity=similarity,
        match=match,
        total_processing_time_ms=total_processing_time,
        image1_metrics=face1_metrics,
        image2_metrics=face2_metrics,
        embedding1_metrics=emb1_metrics,
        embedding2_metrics=emb2_metrics,
        comparison_metrics=comparison_metrics
    )
