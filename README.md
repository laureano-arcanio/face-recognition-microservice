# Face Recognition API Service

This is a high-performance face recognition service that compares two face images and determines if they match. It provides two main endpoints:
- `/compare` - Uses MTCNN for face detection
- `/mediapipe` - Uses MediaPipe for face detection

Both endpoints use InceptionResnetV1 (pretrained on VGGFace2) for face embeddings and similarity computation.

## Features

- Fast face detection using either MTCNN or MediaPipe
- High-quality face embedding generation with InceptionResnetV1
- GPU acceleration support via CUDA
- Asynchronous processing for optimal performance
- Detailed metrics for each stage of the face comparison process
- Configurable similarity threshold
- Performance testing tool

## Requirements

- Docker
- NVIDIA GPU with CUDA support (recommended) or CPU
- For performance testing: Python 3.8+ with dependencies in `requirements.test.txt`

## Docker Setup

### Building the Docker Image

To build the Docker image, run:

```bash
docker build -t face-recognition-api .
```

For GPU support, use:

```bash
docker build -t face-recognition-api . --gpus all
```

### Running the Docker Container

To run the container with CPU:

```bash
docker run -p 8000:8000 face-recognition-api
```

To run with GPU support:

```bash
docker run --gpus all -p 8000:8000 face-recognition-api
```

You can adjust the number of workers in the Dockerfile or override the command:

```bash
docker run -p 8000:8000 face-recognition-api uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Usage

### Endpoints

1. `/` - Health check endpoint
2. `/compare` - Compare faces using MTCNN
3. `/mediapipe` - Compare faces using MediaPipe (recommended for better performance)

### Request Format

```json
{
  "image1_base64": "base64_encoded_image_string",
  "image2_base64": "base64_encoded_image_string"
}
```

### Response Format

```json
{
  "similarity": 0.85,
  "match": true,
  "total_processing_time_ms": 120.5,
  "image1_metrics": {
    "face_detected": true,
    "detection_time_ms": 45.2,
    "face_coordinates": [10, 20, 100, 110],
    "confidence_score": 0.98
  },
  "image2_metrics": {
    "face_detected": true,
    "detection_time_ms": 43.8,
    "face_coordinates": [15, 25, 105, 115],
    "confidence_score": 0.97
  },
  "embedding1_metrics": {
    "embedding_time_ms": 12.3,
    "embedding_dimension": 512,
    "embedding_norm": 1.0,
    "embedding_stats": {
      "min": -0.5,
      "max": 0.5,
      "mean": 0.0,
      "std": 0.2
    }
  },
  "embedding2_metrics": {
    "embedding_time_ms": 12.1,
    "embedding_dimension": 512,
    "embedding_norm": 1.0,
    "embedding_stats": {
      "min": -0.5,
      "max": 0.5,
      "mean": 0.0,
      "std": 0.2
    }
  },
  "comparison_metrics": {
    "similarity_computation_time_ms": 0.1,
    "cosine_similarity": 0.85,
    "euclidean_distance": 0.3,
    "threshold_used": 0.6
  }
}
```

## Performance Testing

### Setup

Install the required dependencies:

```bash
pip install -r requirements.test.txt
```

### Running Performance Tests

The `performance_test.py` script allows you to stress test the API with concurrent requests:

```bash
python performance_test.py --num-requests 100 --url http://localhost:8000/mediapipe
```

### Command Line Options

- `--num-requests`: Number of concurrent requests to make (default: 10)
- `--url`: API endpoint URL (default: http://localhost:8000/mediapipe)
- `--image1`: Path to first image (default: man-a.png)
- `--image2`: Path to second image (default: man-aa.png)

### Sample Test Run

```bash
python performance_test.py --num-requests 50
```

This will:
1. Load the two test images (`man-a.png` and `man-aa.png`)
2. Convert them to base64
3. Send 50 concurrent requests to the API
4. Report detailed performance metrics including:
   - Success rate
   - Requests per second
   - Response time statistics (avg, min, max, median, p95, p99)

## Performance Optimization Tips

1. **Use the MediaPipe endpoint** (`/mediapipe`) for better performance compared to MTCNN
2. Adjust the number of workers in the Docker run command based on your CPU cores
3. Use GPU acceleration when available
4. Resize large images before sending to reduce processing time and bandwidth
5. Tune the number of concurrent requests based on server capacity

## Architecture

The service uses FastAPI for the API layer and leverages asynchronous processing for handling concurrent requests. Face detection is performed using either MTCNN or MediaPipe, and face embeddings are generated using the InceptionResnetV1 model. The comparison is based on cosine similarity between embeddings.

## License

This project is licensed under the MIT License.
