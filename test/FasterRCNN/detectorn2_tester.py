import time
import psutil
from Oculus.FasterRCNN.detectron2 import Detectron2Detector
class TestDetector:
    def setup(self):
        self.model_optimatization = Detectron2Detector(
            onnx_model_path="assets/models/detectron2_fasterrcnn.onnx",
            coco_json_path="assets/labels/coco.json",
            conf_thresh=0.5
        )
        self.model_Unoptimatization = Detectron2Detector(
            onnx_model_path="assets/models/detectron2_fasterrcnn.onnx",
            coco_json_path="assets/labels/coco.json",
            conf_thresh=0.5,
            optimization=False
        )
    
    def testing_timebenmark(self,file_path:str,warmup=3,iterations:int=10):
        """Running benchmark on both optimized and unoptimized models"""

        def benchmark(detector,file_path:str,warmup,iterations):
            # warmup runs
            for _ in range(warmup):
                detector.detect(file_path)
            
            # times runs
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                detector.detect(file_path)
                end = time.perf_counter()
                times.append((end-start)*1000)

            # calculate metrics
            avg_time = sum(times)/len(times)
            fps = 1000 / avg_time if avg_time > 0 else 0

            # memory usage
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024**2)

            output = {
                "avg_time_ms": round(avg_time, 2),
                "min_time_ms": round(min(times), 2),
                "max_time_ms": round(max(times), 2),
                "fps": round(fps, 2),
                "memory_mb": round(mem_mb, 2),
            }
            return output
        print("=" * 50)
        print("BENCHMARK: Detectron2 Inference")
        print("=" * 50)
        
        # Optimized
        print("\n[1] OPTIMIZED (graph optimization ON)")
        result_opt = benchmark(
            self.model_optimatization, 
            file_path, 
            warmup, 
            iterations
        )
        print(f"  Avg Time    : {result_opt['avg_time_ms']:.2f} ms")
        print(f"  Min Time   : {result_opt['min_time_ms']:.2f} ms")
        print(f"  Max Time   : {result_opt['max_time_ms']:.2f} ms")
        print(f"  FPS        : {result_opt['fps']:.2f}")
        print(f"  Memory     : {result_opt['memory_mb']:.2f} MB")
        
        # Unoptimized
        print("\n[2] UNOPTIMIZED (graph optimization OFF)")
        result_unopt = benchmark(
            self.model_Unoptimatization, 
            file_path, 
            warmup, 
            iterations
        )
        print(f"  Avg Time    : {result_unopt['avg_time_ms']:.2f} ms")
        print(f"  Min Time   : {result_unopt['min_time_ms']:.2f} ms")
        print(f"  Max Time   : {result_unopt['max_time_ms']:.2f} ms")
        print(f"  FPS        : {result_unopt['fps']:.2f}")
        print(f"  Memory     : {result_unopt['memory_mb']:.2f} MB")
        
        # Comparison
        print("\n[3] COMPARISON")
        speedup = result_unopt['avg_time_ms'] / result_opt['avg_time_ms']
        time_saved = result_unopt['avg_time_ms'] - result_opt['avg_time_ms']
        print(f"  Speedup    : {speedup:.2f}x")
        print(f"  Time Saved : {time_saved:.2f} ms")
        print("=" * 50)
        
        return result_opt, result_unopt
