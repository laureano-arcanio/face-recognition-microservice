#!/usr/bin/env python3
"""
Performance testing script for face recognition API
Performs asynchronous requests to /mediapipe endpoint with performance metrics
"""

import asyncio
import aiohttp
import argparse
import base64
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import json


class PerformanceMetrics:
    def __init__(self):
        self.response_times = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = None
        self.end_time = None
        
    def add_response_time(self, response_time: float):
        self.response_times.append(response_time)
        
    def add_success(self):
        self.successful_requests += 1
        
    def add_failure(self):
        self.failed_requests += 1
        
    def start_timing(self):
        self.start_time = time.time()
        
    def end_timing(self):
        self.end_time = time.time()
        
    def get_summary(self) -> Dict[str, Any]:
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        total_requests = self.successful_requests + self.failed_requests
        
        summary = {
            "total_requests": total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
        }
        
        if self.response_times:
            summary.update({
                "avg_response_time_ms": statistics.mean(self.response_times),
                "min_response_time_ms": min(self.response_times),
                "max_response_time_ms": max(self.response_times),
                "median_response_time_ms": statistics.median(self.response_times),
                "p95_response_time_ms": self._percentile(self.response_times, 95),
                "p99_response_time_ms": self._percentile(self.response_times, 99),
            })
        
        return summary
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of numbers"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


def load_image_as_base64(image_path: Path) -> str:
    """Load image file and convert to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")


async def make_request(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], metrics: PerformanceMetrics) -> None:
    """Make a single async request to the API"""
    request_start = time.time()
    
    try:
        async with session.post(url, json=payload) as response:
            response_time = (time.time() - request_start) * 1000  # Convert to milliseconds
            metrics.add_response_time(response_time)
            
            if response.status == 200:
                await response.json()  # Read response body
                metrics.add_success()
                print(f"✓ Request completed in {response_time:.2f}ms")
            else:
                metrics.add_failure()
                error_text = await response.text()
                print(f"✗ Request failed with status {response.status}: {error_text}")
                
    except Exception as e:
        response_time = (time.time() - request_start) * 1000
        metrics.add_response_time(response_time)
        metrics.add_failure()
        print(f"✗ Request failed with exception: {e}")


async def run_performance_test(api_url: str, num_requests: int, image1_b64: str, image2_b64: str) -> PerformanceMetrics:
    """Run the performance test with specified number of concurrent requests"""
    metrics = PerformanceMetrics()
    
    # Prepare payload
    payload = {
        "image1_base64": image1_b64,
        "image2_base64": image2_b64
    }
    
    print(f"Starting performance test with {num_requests} requests...")
    print(f"Target URL: {api_url}")
    print(f"Payload size: ~{len(json.dumps(payload))} bytes")
    print("-" * 50)
    
    # Configure session with timeout and connection limits
    timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        metrics.start_timing()
        
        # Create tasks for all requests
        tasks = [
            make_request(session, api_url, payload, metrics)
            for _ in range(num_requests)
        ]
        
        # Execute all requests concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_timing()
    
    return metrics


def print_results(metrics: PerformanceMetrics):
    """Print formatted performance results"""
    summary = metrics.get_summary()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 60)
    
    print(f"Total Requests:        {summary['total_requests']}")
    print(f"Successful Requests:   {summary['successful_requests']}")
    print(f"Failed Requests:       {summary['failed_requests']}")
    print(f"Success Rate:          {summary['success_rate']:.2f}%")
    print(f"Total Time:            {summary['total_time_seconds']:.2f} seconds")
    print(f"Requests per Second:   {summary['requests_per_second']:.2f}")
    
    if 'avg_response_time_ms' in summary:
        print("\nResponse Time Statistics:")
        print(f"Average:               {summary['avg_response_time_ms']:.2f}ms")
        print(f"Median:                {summary['median_response_time_ms']:.2f}ms")
        print(f"Min:                   {summary['min_response_time_ms']:.2f}ms")
        print(f"Max:                   {summary['max_response_time_ms']:.2f}ms")
        print(f"95th Percentile:       {summary['p95_response_time_ms']:.2f}ms")
        print(f"99th Percentile:       {summary['p99_response_time_ms']:.2f}ms")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Performance test for face recognition API")
    parser.add_argument(
        "--num-requests", 
        type=int, 
        default=10,
        help="Number of concurrent requests to make (default: 10)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/mediapipe",
        help="API endpoint URL (default: http://localhost:8000/mediapipe)"
    )
    parser.add_argument(
        "--image1",
        type=str,
        default="man-a.png",
        help="Path to first image (default: man-a.png)"
    )
    parser.add_argument(
        "--image2", 
        type=str,
        default="man-aa.png",
        help="Path to second image (default: man-aa.png)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.num_requests <= 0:
        print("Error: --num-requests must be a positive integer")
        return 1
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    image1_path = script_dir / args.image1
    image2_path = script_dir / args.image2
    
    # Check if image files exist
    if not image1_path.exists():
        print(f"Error: Image file not found: {image1_path}")
        return 1
    
    if not image2_path.exists():
        print(f"Error: Image file not found: {image2_path}")
        return 1
    
    try:
        # Load images
        print("Loading images...")
        image1_b64 = load_image_as_base64(image1_path)
        image2_b64 = load_image_as_base64(image2_path)
        print(f"Loaded {image1_path.name} ({len(image1_b64)} chars)")
        print(f"Loaded {image2_path.name} ({len(image2_b64)} chars)")
        
        # Run performance test
        metrics = asyncio.run(run_performance_test(
            args.url, 
            args.num_requests, 
            image1_b64, 
            image2_b64
        ))
        
        # Print results
        print_results(metrics)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
