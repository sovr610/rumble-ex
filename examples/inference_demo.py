#!/usr/bin/env python3
"""
Brain-Inspired AI - Inference Demo

This script demonstrates how to:
1. Load a trained model
2. Perform single-modality inference
3. Perform multi-modal inference
4. Use different output types (classify, generate, control)
5. Access internal brain states

Run: python examples/inference_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def demo_model_creation():
    """Demo: Creating models with different configurations."""
    print("\n" + "=" * 70)
    print("1. MODEL CREATION DEMO")
    print("=" * 70)
    
    from brain_ai.system import (
        create_brain_ai,
        create_vision_classifier,
        create_multimodal_system,
        create_control_agent,
    )
    
    # Vision classifier
    print("\n📸 Creating vision classifier...")
    vision_model = create_vision_classifier(
        num_classes=10,
        device='cpu',
    )
    print(f"   Created: {type(vision_model).__name__}")
    
    # Multimodal system
    print("\n🔀 Creating multimodal system...")
    multimodal = create_multimodal_system(
        num_classes=100,
        modalities=['vision', 'text'],
        device='cpu',
    )
    print(f"   Created: {type(multimodal).__name__}")
    
    # Control agent
    print("\n🎮 Creating control agent...")
    agent = create_control_agent(
        state_dim=32,
        action_dim=4,
        device='cpu',
    )
    print(f"   Created: {type(agent).__name__}")
    
    # Custom configuration
    print("\n⚙️ Creating custom brain...")
    custom = create_brain_ai(
        modalities=['vision', 'text', 'audio', 'sensors'],
        output_type='classify',
        num_classes=1000,
        hidden_dim=256,
        snn_steps=8,
        htm_columns=1024,
        use_reasoning=True,
        device='cpu',
    )
    print(f"   Created with all modalities and reasoning")
    
    return vision_model


def demo_direct_inference(model):
    """Demo: Direct inference without inference wrapper."""
    print("\n" + "=" * 70)
    print("2. DIRECT MODEL INFERENCE")
    print("=" * 70)
    
    # Create dummy vision input
    batch_size = 2
    vision_input = torch.randn(batch_size, 3, 224, 224)
    
    print("\n🔮 Running forward pass...")
    with torch.no_grad():
        output = model({'vision': vision_input}, return_details=True)
    
    print(f"   Output shape: {output.output.shape}")
    print(f"   Workspace shape: {output.workspace.shape}")
    print(f"   Confidence: {output.confidence.numpy()}")
    
    # Classification
    print("\n📊 Classification...")
    logits = model.classify({'vision': vision_input})
    predictions = torch.argmax(logits, dim=-1)
    print(f"   Predictions: {predictions.numpy()}")


def demo_inference_wrapper():
    """Demo: Using BrainInference wrapper for easy inference."""
    print("\n" + "=" * 70)
    print("3. INFERENCE WRAPPER DEMO")
    print("=" * 70)
    
    from brain_ai.inference import BrainInference
    from brain_ai.system import create_brain_ai
    
    # Create model
    model = create_brain_ai(
        modalities=['vision', 'text'],
        output_type='classify',
        num_classes=10,
        device='cpu',
    )
    
    # Wrap for inference
    brain = BrainInference(model=model, device='cpu')
    
    # Vision inference
    print("\n📷 Vision inference...")
    vision_input = torch.randn(1, 3, 224, 224)
    result = brain.infer({'vision': vision_input})
    
    print(f"   Prediction: {result.prediction}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Top-5 classes: {result.top_k_classes}")
    print(f"   Inference time: {result.inference_time_ms:.1f}ms")
    
    # Text inference
    print("\n📝 Text inference...")
    text_input = "The brain processes information hierarchically."
    result = brain.classify_text(text_input)
    
    print(f"   Input: '{text_input}'")
    print(f"   Prediction: {result.prediction}")
    print(f"   Confidence: {result.confidence:.2%}")
    
    # Multimodal inference
    print("\n🔀 Multimodal inference...")
    result = brain.infer({
        'vision': vision_input,
        'text': text_input,
    })
    
    print(f"   Modalities used: {result.modalities_used}")
    print(f"   Prediction: {result.prediction}")
    print(f"   Confidence: {result.confidence:.2%}")
    
    return brain


def demo_generation():
    """Demo: Text generation mode."""
    print("\n" + "=" * 70)
    print("4. TEXT GENERATION DEMO")
    print("=" * 70)
    
    from brain_ai.inference import BrainInference
    from brain_ai.system import create_brain_ai
    
    # Create generator model
    model = create_brain_ai(
        modalities=['text'],
        output_type='generate',
        vocab_size=10000,
        device='cpu',
    )
    
    brain = BrainInference(model=model, device='cpu')
    
    print("\n✍️ Generating text...")
    prompt = "Once upon a time"
    
    try:
        output = brain.generate(
            prompt=prompt,
            max_length=50,
            temperature=0.8,
        )
        print(f"   Prompt: {prompt}")
        print(f"   Generated: {output}")
    except NotImplementedError as e:
        print(f"   Note: {e}")
        print("   (Generation requires trained model)")


def demo_control_agent():
    """Demo: Control/RL agent mode."""
    print("\n" + "=" * 70)
    print("5. CONTROL AGENT DEMO")
    print("=" * 70)
    
    from brain_ai.inference import BrainInference
    from brain_ai.system import create_control_agent
    
    state_dim = 32
    action_dim = 4
    
    # Create agent
    agent = create_control_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu',
    )
    
    brain = BrainInference(model=agent, device='cpu')
    
    print("\n🎮 Running control agent...")
    
    # Simulate environment interaction
    for step in range(5):
        # Random observation
        observation = torch.randn(state_dim)
        
        # Get action
        action = brain.get_action(observation, deterministic=True)
        
        print(f"   Step {step + 1}: obs shape={observation.shape}, action={action}")
    
    # Reset agent state
    brain.reset()
    print("\n   Agent state reset for new episode")


def demo_internal_states(brain):
    """Demo: Accessing internal brain states."""
    print("\n" + "=" * 70)
    print("6. INTERNAL STATES DEMO")
    print("=" * 70)
    
    # Create input
    vision_input = torch.randn(1, 3, 224, 224)
    
    print("\n🧠 Examining internal states...")
    result = brain.infer({'vision': vision_input})
    
    # Workspace state
    print(f"\n   Workspace state shape: {result.workspace_state.shape if result.workspace_state is not None else 'None'}")
    
    # Attention weights (can be a dict of tensors or a single tensor)
    if result.attention_weights is not None:
        if isinstance(result.attention_weights, dict):
            print(f"   Attention weights: {len(result.attention_weights)} modalities")
            for modality, weights in result.attention_weights.items():
                if hasattr(weights, 'shape'):
                    print(f"      {modality}: shape={weights.shape}, max={weights.max().item():.4f}")
                else:
                    print(f"      {modality}: {type(weights).__name__}")
        elif hasattr(result.attention_weights, 'shape'):
            print(f"   Attention weights shape: {result.attention_weights.shape}")
            print(f"   Max attention: {result.attention_weights.max().item():.4f}")
    else:
        print("   Attention weights: None")
    
    # Anomaly score (from HTM)
    if result.anomaly_score is not None:
        print(f"   Anomaly score: {result.anomaly_score:.4f}")
    else:
        print("   Anomaly score: None")
    
    # Reasoning trace
    if result.reasoning_used:
        print("   Reasoning was used for this inference")
    else:
        print("   No reasoning applied")


def demo_batch_processing(brain):
    """Demo: Batch processing for efficiency."""
    print("\n" + "=" * 70)
    print("7. BATCH PROCESSING DEMO")
    print("=" * 70)
    
    import time
    
    # Create batch of inputs
    batch_size = 16
    images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    
    print(f"\n📦 Processing {batch_size} images...")
    
    # Reset state to ensure clean batch processing
    brain.reset()
    
    # Time single image processing
    start = time.time()
    for img in images[:4]:
        _ = brain.infer({'vision': img.unsqueeze(0)})
    single_time = (time.time() - start) / 4
    
    # Reset state before batch processing to avoid dimension mismatch
    brain.reset()
    
    # Time batch processing
    start = time.time()
    results = brain.batch_classify(images, batch_size=8)
    batch_time = (time.time() - start) / batch_size
    
    print(f"   Single image time: {single_time * 1000:.1f}ms")
    print(f"   Batch per-image time: {batch_time * 1000:.1f}ms")
    print(f"   Speedup: {single_time / batch_time:.2f}x")
    print(f"   Processed {len(results)} images")


def demo_save_load():
    """Demo: Saving and loading models."""
    print("\n" + "=" * 70)
    print("8. SAVE/LOAD DEMO")
    print("=" * 70)
    
    import tempfile
    from brain_ai.system import create_brain_ai
    from brain_ai.inference import BrainInference
    
    # Create model
    model = create_brain_ai(
        modalities=['vision'],
        output_type='classify',
        num_classes=10,
        device='cpu',
    )
    
    # Save checkpoint
    checkpoint_path = Path(tempfile.gettempdir()) / 'brain_demo.pth'
    
    print(f"\n💾 Saving model to {checkpoint_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'modalities': ['vision'],
            'output_type': 'classify',
            'num_classes': 10,
        },
        'epoch': 100,
        'best_accuracy': 0.95,
    }
    torch.save(checkpoint, checkpoint_path)
    print("   Model saved!")
    
    # Load checkpoint
    print("\n📂 Loading model...")
    brain = BrainInference.load(str(checkpoint_path), device='cpu')
    print("   Model loaded!")
    
    # Verify
    test_input = torch.randn(1, 3, 224, 224)
    result = brain.infer({'vision': test_input})
    print(f"   Verification: prediction={result.prediction}, confidence={result.confidence:.2%}")
    
    # Cleanup
    checkpoint_path.unlink()


def demo_with_class_names():
    """Demo: Using class names for readable predictions."""
    print("\n" + "=" * 70)
    print("9. CLASS NAMES DEMO")
    print("=" * 70)
    
    from brain_ai.inference import BrainInference
    from brain_ai.system import create_brain_ai
    
    # Define class names (e.g., CIFAR-10)
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    
    # Create model
    model = create_brain_ai(
        modalities=['vision'],
        output_type='classify',
        num_classes=len(class_names),
        device='cpu',
    )
    
    # Create inference wrapper with class names
    brain = BrainInference(model=model, device='cpu', class_names=class_names)
    
    # Run inference
    print("\n🏷️ Classification with class names...")
    image = torch.randn(1, 3, 224, 224)
    result = brain.infer({'vision': image})
    
    print(f"   Prediction: {result.prediction}")  # Now shows class name!
    print(f"   Top-5 classes: {result.top_k_classes}")
    print(f"   Confidence: {result.confidence:.2%}")


def main():
    """Run all demos."""
    print("\n" + "🧠" * 35)
    print("\n   BRAIN-INSPIRED AI - INFERENCE DEMO")
    print("\n" + "🧠" * 35)
    
    try:
        # 1. Model creation
        model = demo_model_creation()
        
        # 2. Direct inference
        demo_direct_inference(model)
        
        # 3. Inference wrapper
        brain = demo_inference_wrapper()
        
        # 4. Text generation
        demo_generation()
        
        # 5. Control agent
        demo_control_agent()
        
        # 6. Internal states
        demo_internal_states(brain)
        
        # 7. Batch processing
        demo_batch_processing(brain)
        
        # 8. Save/Load
        demo_save_load()
        
        # 9. Class names
        demo_with_class_names()
        
        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
