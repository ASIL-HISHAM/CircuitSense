#!/usr/bin/env python3
"""
CircuitSense — Inference Script
Run predictions on waveforms from command line
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import CircuitSensePredictor
from src.core import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    """Run inference on a single waveform or batch."""
    parser = argparse.ArgumentParser(
        description="CircuitSense Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/infer.py --input waveforms.json
  python scripts/infer.py --waveform "[0.1, 0.2, ...]" --checkpoint checkpoints/best.pt
        """
    )
    parser.add_argument("--checkpoint", default="data/checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--input", help="Path to JSON file with waveforms")
    parser.add_argument("--waveform", help="Single waveform as JSON array string")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|cpu|mps")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level, module_name="circuitsense.infer")
    
    # Load predictor
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = CircuitSensePredictor(args.checkpoint, device=args.device)
    logger.info(f"Model loaded on device: {predictor.device}")
    
    results = []
    
    # Process input
    if args.waveform:
        try:
            waveform = json.loads(args.waveform)
            waveform = np.array(waveform, dtype=np.float32)
            result = predictor.predict(waveform)
            results.append(result)
            logger.info(f"Predicted: {result['fault_name']} ({result['confidence']:.1%})")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON waveform: {e}")
            sys.exit(1)
    
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        with open(input_path) as f:
            data = json.load(f)
        
        # Handle both single waveform and list
        waveforms = data if isinstance(data, list) else [data]
        logger.info(f"Processing {len(waveforms)} waveforms...")
        
        for i, w in enumerate(waveforms, 1):
            try:
                waveform = np.array(w, dtype=np.float32)
                result = predictor.predict(waveform)
                results.append(result)
                logger.info(f"[{i}/{len(waveforms)}] {result['fault_name']} ({result['confidence']:.1%})")
            except Exception as e:
                logger.error(f"[{i}/{len(waveforms)}] Prediction failed: {e}")
                results.append({"error": str(e)})
    
    else:
        parser.print_help()
        logger.warning("Provide either --waveform or --input")
        sys.exit(1)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
