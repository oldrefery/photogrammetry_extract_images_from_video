# video_processor.py
# pip install -r requirements.txt
# to run
# python video_processor.py IMG_0553.MOV

import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time

class VideoProcessor:
    SUPPORTED_FORMATS = {'.mov', '.mp4', '.avi', '.m4v'}

    def __init__(self, video_path, output_dir="images"):
        """
        Initialize the video processor

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save extracted frames
        """
        self.video_path = Path(video_path)

        # Check file format
        if self.video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_frames = set()  # To track frames with debug output
        self.debug_all = False  # Flag to output debugging for all frames

    def check_frame_quality(self, frame):
        """
        Check various quality metrics for photogrammetry suitability with thresholds adjusted for your video
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)

        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        feature_count = len(keypoints)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

        # Adjusted thresholds based on your video metrics
        blur_ok = blur_score > 5  # Lowered from 10
        brightness_ok = 10 < brightness < 245  # Keep wide range
        contrast_ok = contrast > 10  # Keep as is
        features_ok = feature_count > 20  # Keep as is
        edges_ok = edge_density > 0.005  # Lowered from 0.01

        is_good = blur_ok and brightness_ok and contrast_ok and features_ok and edges_ok

        # Print debug info only for the first few frames or when explicitly requested
        if not is_good and (len(self.debug_frames) < 5 or self.debug_all):
            self.debug_frames.add(len(self.debug_frames))
            print(f"\nFrame quality metrics:")
            print(f"Blur score: {blur_score:.1f} ({'PASS' if blur_ok else 'FAIL'})")
            print(f"Brightness: {brightness:.1f} ({'PASS' if brightness_ok else 'FAIL'})")
            print(f"Contrast: {contrast:.1f} ({'PASS' if contrast_ok else 'FAIL'})")
            print(f"Feature count: {feature_count} ({'PASS' if features_ok else 'FAIL'})")
            print(f"Edge density: {edge_density:.3f} ({'PASS' if edges_ok else 'FAIL'})")

        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'feature_count': feature_count,
            'edge_density': edge_density,
            'is_good': is_good
        }

    def extract_frames(self, frame_interval=15, blur_threshold=50, min_frames=30, quality=95):
        """
        Extract frames from video with quality check and progress tracking
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_count = 0
        saved_count = 0
        frames_info = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps

        quality_summary = {
            'total_frames_checked': 0,
            'good_quality_frames': 0,
            'poor_quality_frames': 0,
            'quality_metrics': []
        }

        print(f"\nProcessing video: {self.video_path.name}")
        print(f"Total frames: {total_frames}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print("\nProgress:")

        last_update_time = time.time()
        update_interval = 1.0  # Update progress every second

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                # Calculate progress
                current_duration = frame_count / fps
                progress = (frame_count / total_frames) * 100

                # Create progress bar
                bar_length = 40
                filled_length = int(bar_length * frame_count // total_frames)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)

                # Clear previous line and print progress
                print(f"\r[{bar}] {progress:.1f}% "
                      f"| Time: {current_duration:.1f}/{total_duration:.1f}s "
                      f"| Frames: {frame_count}/{total_frames} "
                      f"| Saved: {saved_count}", end='')

                last_update_time = current_time

            if frame_count % frame_interval == 0:
                # Check frame quality
                quality_metrics = self.check_frame_quality(frame)
                quality_summary['total_frames_checked'] += 1

                if quality_metrics['is_good']:
                    quality_summary['good_quality_frames'] += 1

                    # Save frame if it's good quality
                    frame_path = self.output_dir / f"frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

                    # Save quality information
                    frame_info = {
                        'frame_number': frame_count,
                        'quality_metrics': quality_metrics,
                        'file_name': frame_path.name
                    }
                    frames_info.append(frame_info)
                    saved_count += 1

                else:
                    quality_summary['poor_quality_frames'] += 1

                quality_summary['quality_metrics'].append(quality_metrics)

            frame_count += 1

        # Print final newline after progress bar
        print("\n")

        cap.release()

        # Save quality information
        self._save_quality_report(frames_info, quality_summary)

        return saved_count, quality_summary

    def _save_quality_report(self, frames_info, quality_summary):
        """Save detailed quality report"""
        report_path = self.output_dir / "quality_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Photogrammetry Quality Report ===\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total frames checked: {quality_summary['total_frames_checked']}\n")
            f.write(f"Good quality frames: {quality_summary['good_quality_frames']}\n")
            f.write(f"Poor quality frames: {quality_summary['poor_quality_frames']}\n")
            f.write(
                f"Quality ratio: {quality_summary['good_quality_frames'] / quality_summary['total_frames_checked'] * 100:.1f}%\n\n")

            # Per-frame details
            f.write("Per-Frame Quality Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write("Frame | Blur Score | Brightness | Contrast | Features | Edge Density | Status\n")
            f.write("-" * 80 + "\n")

            for info in frames_info:
                metrics = info['quality_metrics']
                f.write(f"{info['frame_number']:5d} | "
                        f"{metrics['blur_score']:10.1f} | "
                        f"{metrics['brightness']:9.1f} | "
                        f"{metrics['contrast']:8.1f} | "
                        f"{metrics['feature_count']:8d} | "
                        f"{metrics['edge_density']:11.3f} | "
                        f"{'GOOD' if metrics['is_good'] else 'POOR'}\n")

    def analyze_video(self):
        """Analyze video properties"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        return {
            "resolution": (width, height),
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "codec": codec,
            "file_size_mb": os.path.getsize(self.video_path) / (1024 * 1024)
        }


def main():
    import sys
    import time
    from datetime import datetime, timedelta

    start_time = time.time()
    start_datetime = datetime.now()

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Please specify the path to the video file")
        print("Usage: python video_processor.py path/to/video.mov")
        sys.exit(1)

    processor = VideoProcessor(video_path)

    # Video analysis
    video_info = processor.analyze_video()
    print("\nVideo Information:")
    print(f"Resolution: {video_info['resolution']}")
    print(f"FPS: {video_info['fps']:.2f} FPS")
    print(f"Duration: {video_info['duration']:.2f} seconds")
    print(f"Codec: {video_info['codec']}")
    print(f"File size: {video_info['file_size_mb']:.2f} MB")

    print(f"\nStart time: {start_datetime.strftime('%H:%M:%S')}")

    # Frame extraction
    frames_saved, quality_summary = processor.extract_frames(
        frame_interval=30,
        blur_threshold=5,
        min_frames=20,
        quality=95
    )

    # Time calculations
    end_time = time.time()
    execution_time = end_time - start_time
    end_datetime = datetime.now()

    frames_per_second = quality_summary['total_frames_checked'] / execution_time
    processing_time_per_frame = execution_time / quality_summary['total_frames_checked'] if quality_summary[
                                                                                                'total_frames_checked'] > 0 else 0

    print(f"\nPerformance Summary:")
    print(f"Start time: {start_datetime.strftime('%H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%H:%M:%S')}")
    print(f"Total execution time: {int(execution_time // 60)} minutes {execution_time % 60:.1f} seconds")
    print(f"Average processing speed: {frames_per_second:.1f} frames/second")
    print(f"Average time per frame: {processing_time_per_frame:.2f} seconds")

    print(f"\nExtraction Results:")
    print(f"Total frames checked: {quality_summary['total_frames_checked']}")
    print(f"Good quality frames: {quality_summary['good_quality_frames']}")
    print(f"Poor quality frames: {quality_summary['poor_quality_frames']}")
    print(
        f"Quality ratio: {quality_summary['good_quality_frames'] / quality_summary['total_frames_checked'] * 100:.1f}%")
    print(f"Successfully saved {frames_saved} frames")
    print(f"Frames saved in directory: {processor.output_dir}")


if __name__ == "__main__":
    main()