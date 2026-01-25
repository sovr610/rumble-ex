#!/usr/bin/env python3
"""
Brain-Inspired AI Command Line Interface

Usage:
    # Interactive mode
    python -m brain_ai.cli interactive

    # Single inference
    python -m brain_ai.cli classify --image path/to/image.jpg
    python -m brain_ai.cli classify --text "Some text to classify"
    python -m brain_ai.cli generate --prompt "Once upon a time"
    
    # Batch processing
    python -m brain_ai.cli batch --input images/*.jpg --output results.json
    
    # Start server
    python -m brain_ai.cli serve --port 8000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Brain-Inspired AI Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device for inference',
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Start interactive inference session',
    )
    
    # Classify
    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify single input',
    )
    classify_parser.add_argument('--image', type=str, help='Image path')
    classify_parser.add_argument('--text', type=str, help='Text to classify')
    classify_parser.add_argument('--audio', type=str, help='Audio path')
    classify_parser.add_argument('--top-k', type=int, default=5, help='Top K predictions')
    
    # Generate
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate text',
    )
    generate_parser.add_argument('--prompt', '-p', type=str, required=True, help='Input prompt')
    generate_parser.add_argument('--max-length', type=int, default=100, help='Max generation length')
    generate_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    # Batch processing
    batch_parser = subparsers.add_parser(
        'batch',
        help='Process multiple inputs',
    )
    batch_parser.add_argument('--input', '-i', type=str, required=True, help='Input pattern (glob)')
    batch_parser.add_argument('--output', '-o', type=str, default='results.json', help='Output file')
    batch_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Server
    serve_parser = subparsers.add_parser(
        'serve',
        help='Start inference server',
    )
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port number')
    
    # Info
    info_parser = subparsers.add_parser(
        'info',
        help='Show model information',
    )
    
    # Demo
    demo_parser = subparsers.add_parser(
        'demo',
        help='Run demo with sample inputs',
    )
    demo_parser.add_argument('--modality', type=str, choices=['vision', 'text', 'audio', 'all'], default='all')
    
    return parser


def load_model(args):
    """Load model from checkpoint or create default."""
    from .inference import BrainInference
    from .system import create_brain_ai
    
    if args.checkpoint:
        return BrainInference.load(args.checkpoint, device=args.device)
    else:
        # Create default model for demo
        logger.info("No checkpoint provided, creating default model...")
        model = create_brain_ai(
            modalities=['vision', 'text', 'audio'],
            output_type='classify',
            num_classes=10,
            device=args.device,
        )
        return BrainInference(model=model, device=args.device)


def cmd_interactive(args):
    """Run interactive session."""
    brain = load_model(args)
    brain.interactive()


def cmd_classify(args):
    """Classify single input."""
    brain = load_model(args)
    
    if args.image:
        result = brain.classify_image(args.image, top_k=args.top_k)
    elif args.text:
        result = brain.classify_text(args.text, top_k=args.top_k)
    elif args.audio:
        result = brain.classify_audio(args.audio, top_k=args.top_k)
    else:
        print("Error: Provide --image, --text, or --audio")
        sys.exit(1)
    
    print(json.dumps(result.to_dict(), indent=2))


def cmd_generate(args):
    """Generate text."""
    brain = load_model(args)
    
    output = brain.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated:\n{output}")


def cmd_batch(args):
    """Batch processing."""
    import glob
    
    brain = load_model(args)
    
    # Find input files
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"No files found matching: {args.input}")
        sys.exit(1)
    
    print(f"Processing {len(input_files)} files...")
    
    # Process in batches
    results = brain.batch_classify(input_files, batch_size=args.batch_size)
    
    # Save results
    output_data = {
        'results': [
            {
                'file': str(f),
                **r.to_dict(),
            }
            for f, r in zip(input_files, results)
        ],
        'total': len(results),
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")


def cmd_serve(args):
    """Start inference server."""
    try:
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("Server requires: pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    
    brain = load_model(args)
    
    app = FastAPI(
        title="Brain-Inspired AI API",
        description="Multi-modal inference API",
        version="1.0.0",
    )
    
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    @app.post("/classify/image")
    async def classify_image(file: UploadFile = File(...)):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = brain.classify_image(tmp_path)
        Path(tmp_path).unlink()
        
        return JSONResponse(result.to_dict())
    
    @app.post("/classify/text")
    async def classify_text(text: str = Form(...)):
        result = brain.classify_text(text)
        return JSONResponse(result.to_dict())
    
    @app.post("/generate")
    async def generate(
        prompt: str = Form(...),
        max_length: int = Form(100),
        temperature: float = Form(1.0),
    ):
        output = brain.generate(prompt, max_length=max_length, temperature=temperature)
        return {"prompt": prompt, "generated": output}
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("API docs available at /docs")
    
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_info(args):
    """Show model information."""
    brain = load_model(args)
    brain._print_info()


def cmd_demo(args):
    """Run demo."""
    import torch
    import numpy as np
    
    brain = load_model(args)
    
    print("\n" + "="*60)
    print("Brain-Inspired AI Demo")
    print("="*60)
    
    if args.modality in ['vision', 'all']:
        print("\n📷 Vision Demo:")
        print("-" * 40)
        # Create random image tensor
        dummy_image = torch.randn(1, 3, 224, 224)
        try:
            result = brain.infer({'vision': dummy_image})
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Inference time: {result.inference_time_ms:.1f}ms")
        except Exception as e:
            print(f"  Error: {e}")
    
    if args.modality in ['text', 'all']:
        print("\n📝 Text Demo:")
        print("-" * 40)
        sample_text = "The brain-inspired AI system processes information efficiently."
        try:
            result = brain.classify_text(sample_text)
            print(f"  Input: '{sample_text[:50]}...'")
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.2%}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if args.modality in ['audio', 'all']:
        print("\n🔊 Audio Demo:")
        print("-" * 40)
        # Create random audio tensor (1 second at 16kHz)
        dummy_audio = torch.randn(1, 1, 16000)
        try:
            result = brain.infer({'audio': dummy_audio})
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.2%}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    commands = {
        'interactive': cmd_interactive,
        'classify': cmd_classify,
        'generate': cmd_generate,
        'batch': cmd_batch,
        'serve': cmd_serve,
        'info': cmd_info,
        'demo': cmd_demo,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
