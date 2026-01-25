"""
Brain-Inspired AI Inference Interface

Provides a unified interface for loading trained models and running inference
across all supported modalities and tasks.

Usage:
    from brain_ai.inference import BrainInference
    
    # Load trained model
    brain = BrainInference.load('checkpoints/brain_ai_v1.pth')
    
    # Single modality inference
    result = brain.classify_image('path/to/image.jpg')
    result = brain.classify_text("Some input text")
    result = brain.classify_audio('path/to/audio.wav')
    
    # Multi-modal inference
    result = brain.infer({
        'vision': image_tensor,
        'text': "What is in this image?",
    })
    
    # Interactive session
    brain.interactive()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from model inference."""
    # Core outputs
    output: torch.Tensor
    prediction: Union[int, str, np.ndarray]
    confidence: float
    
    # Optional details
    probabilities: Optional[torch.Tensor] = None
    top_k_classes: Optional[List[Tuple[int, float]]] = None
    workspace_state: Optional[torch.Tensor] = None
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    reasoning_used: bool = False
    anomaly_score: Optional[float] = None
    
    # Metadata
    modalities_used: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prediction': self.prediction if not isinstance(self.prediction, np.ndarray) 
                         else self.prediction.tolist(),
            'confidence': self.confidence,
            'top_k_classes': self.top_k_classes,
            'reasoning_used': self.reasoning_used,
            'anomaly_score': self.anomaly_score,
            'modalities_used': self.modalities_used,
            'inference_time_ms': self.inference_time_ms,
        }
    
    def __repr__(self) -> str:
        return (
            f"InferenceResult(\n"
            f"  prediction={self.prediction},\n"
            f"  confidence={self.confidence:.2%},\n"
            f"  modalities={self.modalities_used},\n"
            f"  reasoning_used={self.reasoning_used}\n"
            f")"
        )


class BrainInference:
    """
    High-level inference interface for Brain-Inspired AI.
    
    Handles:
    - Model loading from checkpoints
    - Input preprocessing for all modalities
    - Batched and single-sample inference
    - Result interpretation and formatting
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None,
        device: str = 'auto',
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize inference interface.
        
        Args:
            model: Trained BrainAI model
            config: Model configuration
            device: Device for inference
            class_names: Optional class name mapping
        """
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        
        self.config = config or {}
        self.class_names = class_names
        
        # Lazy-load preprocessors
        self._vision_transform = None
        self._text_tokenizer = None
        self._audio_processor = None
    
    def reset(self):
        """Reset all stateful components in the model."""
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: str = 'auto',
        **kwargs,
    ) -> 'BrainInference':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            device: Device for inference
            **kwargs: Additional arguments
            
        Returns:
            BrainInference instance
        """
        from .system import BrainAI, create_brain_ai
        from .config import BrainAIConfig
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config
        config = checkpoint.get('config', {})
        if not isinstance(config, dict):
            config = {}
        
        # Separate BrainAI-level params from BrainAIConfig params
        modalities = config.pop('modalities', ['vision'])
        output_type = config.pop('output_type', 'classify')
        num_classes = config.pop('num_classes', 10)
        
        # Build BrainAIConfig only with valid fields
        # For simplicity, use defaults - the weights will override
        model_config = BrainAIConfig()
        model_config.modalities = modalities
        if num_classes:
            model_config.decision.num_classes = num_classes
            
        # Create model
        model = BrainAI(
            config=model_config,
            modalities=modalities,
            output_type=output_type,
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        # Get class names if available
        class_names = checkpoint.get('class_names', kwargs.get('class_names'))
        
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return cls(
            model=model,
            config={'modalities': modalities, 'output_type': output_type, 'num_classes': num_classes},
            device=device,
            class_names=class_names,
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = 'auto',
    ) -> 'BrainInference':
        """
        Load a pretrained model by name.
        
        Args:
            model_name: Name of pretrained model
            device: Device for inference
            
        Returns:
            BrainInference instance
        """
        from .system import create_brain_ai
        
        # Predefined configurations
        presets = {
            'vision-classifier': {
                'modalities': ['vision'],
                'output_type': 'classify',
                'num_classes': 1000,  # ImageNet
            },
            'multimodal-reasoning': {
                'modalities': ['vision', 'text'],
                'output_type': 'classify',
                'use_symbolic': True,
            },
            'text-classifier': {
                'modalities': ['text'],
                'output_type': 'classify',
                'num_classes': 10,
            },
            'control-agent': {
                'modalities': ['sensors'],
                'output_type': 'control',
                'control_dim': 6,
            },
        }
        
        if model_name not in presets:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(presets.keys())}")
        
        model = create_brain_ai(**presets[model_name], device=device)
        
        return cls(model=model, config=presets[model_name], device=device)
    
    # ==================== Preprocessing ====================
    
    @property
    def vision_transform(self):
        """Get or create vision preprocessing transform."""
        if self._vision_transform is None:
            try:
                from torchvision import transforms
                self._vision_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
            except ImportError:
                logger.warning("torchvision not available for image preprocessing")
                self._vision_transform = lambda x: x
        return self._vision_transform
    
    @property
    def text_tokenizer(self):
        """Get or create text tokenizer."""
        if self._text_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            except ImportError:
                logger.warning("transformers not available, using simple tokenizer")
                self._text_tokenizer = SimpleTokenizer()
        return self._text_tokenizer
    
    def preprocess_image(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor, 'PIL.Image.Image'],
    ) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path, numpy array, tensor, or PIL Image
            
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        if not isinstance(image, torch.Tensor):
            image = self.vision_transform(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        return image.to(self.device)
    
    def preprocess_text(
        self,
        text: Union[str, List[str]],
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Preprocess text for inference.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            
        Returns:
            Token tensor [batch, seq_len]
        """
        if isinstance(text, str):
            text = [text]
        
        if hasattr(self.text_tokenizer, '__call__'):
            # HuggingFace tokenizer
            encoded = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            return encoded['input_ids'].to(self.device)
        else:
            # Simple tokenizer fallback
            return self.text_tokenizer.encode(text).to(self.device)
    
    def preprocess_audio(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Preprocess audio for inference.
        
        Args:
            audio: Audio path or waveform
            sample_rate: Expected sample rate
            
        Returns:
            Preprocessed audio tensor
        """
        if isinstance(audio, (str, Path)):
            try:
                import torchaudio
                waveform, sr = torchaudio.load(str(audio))
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            except ImportError:
                import scipy.io.wavfile as wav
                sr, waveform = wav.read(str(audio))
                waveform = torch.from_numpy(waveform.astype(np.float32))
                if sr != sample_rate:
                    # Simple resampling
                    ratio = sample_rate / sr
                    new_length = int(len(waveform) * ratio)
                    waveform = F.interpolate(
                        waveform.unsqueeze(0).unsqueeze(0),
                        size=new_length,
                        mode='linear',
                    ).squeeze()
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio.astype(np.float32))
        else:
            waveform = audio
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            
        return waveform.to(self.device)
    
    # ==================== Inference Methods ====================
    
    @torch.no_grad()
    def infer(
        self,
        inputs: Dict[str, Any],
        task: Optional[str] = None,
        top_k: int = 5,
        return_workspace: bool = False,
    ) -> InferenceResult:
        """
        Run inference on multi-modal inputs.
        
        Args:
            inputs: Dict mapping modality names to inputs
            task: Override task type
            top_k: Number of top predictions to return
            return_workspace: Include workspace state in result
            
        Returns:
            InferenceResult with predictions and details
        """
        import time
        start_time = time.perf_counter()
        
        # Preprocess inputs
        processed = {}
        modalities_used = []
        
        for name, data in inputs.items():
            if name == 'vision' or name == 'image':
                processed['vision'] = self.preprocess_image(data)
                modalities_used.append('vision')
            elif name == 'text':
                processed['text'] = self.preprocess_text(data)
                modalities_used.append('text')
            elif name == 'audio':
                processed['audio'] = self.preprocess_audio(data)
                modalities_used.append('audio')
            elif name == 'sensors':
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                processed['sensors'] = data.to(self.device)
                modalities_used.append('sensors')
            else:
                # Pass through unknown modalities
                if isinstance(data, torch.Tensor):
                    processed[name] = data.to(self.device)
                    modalities_used.append(name)
        
        # Run model
        output = self.model.forward(
            processed,
            task=task,
            return_details=True,
        )
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Process results
        if hasattr(output, 'output'):
            # SystemOutput
            logits = output.output
            confidence_tensor = output.confidence
            workspace_state = output.workspace if return_workspace else None
            attention = output.attention
            reasoning_used = output.reasoning_trace is not None
        else:
            logits = output
            confidence_tensor = None
            workspace_state = None
            attention = None
            reasoning_used = False
        
        # Get predictions
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities.max(dim=-1)[0].item()
        prediction = probabilities.argmax(dim=-1).item()
        
        # Top-k
        top_k_probs, top_k_indices = probabilities.topk(min(top_k, probabilities.shape[-1]))
        top_k_classes = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_k_indices[0], top_k_probs[0])
        ]
        
        # Map to class names if available
        if self.class_names:
            prediction = self.class_names[prediction]
            top_k_classes = [
                (self.class_names[idx], prob)
                for idx, prob in top_k_classes
            ]
        
        return InferenceResult(
            output=logits,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            top_k_classes=top_k_classes,
            workspace_state=workspace_state,
            attention_weights=attention,
            reasoning_used=reasoning_used,
            modalities_used=modalities_used,
            inference_time_ms=inference_time,
        )
    
    def classify_image(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> InferenceResult:
        """
        Classify a single image.
        
        Args:
            image: Image to classify
            top_k: Number of top predictions
            
        Returns:
            InferenceResult
        """
        return self.infer({'vision': image}, task='classify', top_k=top_k)
    
    def classify_text(
        self,
        text: Union[str, List[str]],
        top_k: int = 5,
    ) -> InferenceResult:
        """
        Classify text.
        
        Args:
            text: Text to classify
            top_k: Number of top predictions
            
        Returns:
            InferenceResult
        """
        return self.infer({'text': text}, task='classify', top_k=top_k)
    
    def classify_audio(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> InferenceResult:
        """
        Classify audio.
        
        Args:
            audio: Audio to classify
            top_k: Number of top predictions
            
        Returns:
            InferenceResult
        """
        return self.infer({'audio': audio}, task='classify', top_k=top_k)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        inputs = {'text': prompt}
        processed = {'text': self.preprocess_text(prompt)}
        
        with torch.no_grad():
            output = self.model.generate(
                processed,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
        
        # Decode output
        if hasattr(self.text_tokenizer, 'decode'):
            return self.text_tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            return str(output.tolist())
    
    def get_action(
        self,
        observation: Union[Dict[str, Any], np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get control action from observation.
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic policy
            
        Returns:
            Action array
        """
        if isinstance(observation, (np.ndarray, torch.Tensor)):
            inputs = {'sensors': observation}
        else:
            inputs = observation
            
        with torch.no_grad():
            action = self.model.forward(
                {k: self.preprocess_sensors(v) if k == 'sensors' else v 
                 for k, v in inputs.items()},
                task='control',
                deterministic=deterministic,
            )
        
        return action.cpu().numpy()
    
    def preprocess_sensors(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess sensor data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return data.to(self.device)
    
    # ==================== Batch Inference ====================
    
    @torch.no_grad()
    def batch_classify(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 32,
    ) -> List[InferenceResult]:
        """
        Classify multiple images in batches.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            List of InferenceResults
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Preprocess batch
            tensors = [self.preprocess_image(img) for img in batch]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Run inference
            output = self.model({'vision': batch_tensor}, task='classify', return_details=True)
            
            # Process each result
            probs = F.softmax(output.output, dim=-1)
            for j in range(len(batch)):
                pred = probs[j].argmax().item()
                conf = probs[j].max().item()
                
                results.append(InferenceResult(
                    output=output.output[j:j+1],
                    prediction=self.class_names[pred] if self.class_names else pred,
                    confidence=conf,
                    probabilities=probs[j:j+1],
                    modalities_used=['vision'],
                ))
        
        return results
    
    # ==================== Interactive Mode ====================
    
    def interactive(self):
        """
        Start interactive inference session.
        
        Provides a command-line interface for testing the model.
        """
        print("\n" + "="*60)
        print("Brain-Inspired AI Interactive Session")
        print("="*60)
        print("\nCommands:")
        print("  image <path>     - Classify an image")
        print("  text <text>      - Classify text")
        print("  audio <path>     - Classify audio")
        print("  generate <prompt> - Generate text")
        print("  info             - Show model info")
        print("  quit             - Exit session")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'info':
                    self._print_info()
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ''
                
                if command == 'image':
                    if not arg:
                        print("Usage: image <path>")
                        continue
                    result = self.classify_image(arg)
                    self._print_result(result)
                
                elif command == 'text':
                    if not arg:
                        print("Usage: text <text>")
                        continue
                    result = self.classify_text(arg)
                    self._print_result(result)
                
                elif command == 'audio':
                    if not arg:
                        print("Usage: audio <path>")
                        continue
                    result = self.classify_audio(arg)
                    self._print_result(result)
                
                elif command == 'generate':
                    if not arg:
                        print("Usage: generate <prompt>")
                        continue
                    output = self.generate(arg)
                    print(f"\nGenerated:\n{output}\n")
                
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_result(self, result: InferenceResult):
        """Print formatted inference result."""
        print(f"\n{'─'*40}")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Inference time: {result.inference_time_ms:.1f}ms")
        
        if result.top_k_classes:
            print(f"\nTop predictions:")
            for i, (cls, prob) in enumerate(result.top_k_classes[:5], 1):
                print(f"  {i}. {cls}: {prob:.2%}")
        
        if result.reasoning_used:
            print(f"\n⚡ System 2 reasoning was engaged")
        
        print(f"{'─'*40}\n")
    
    def _print_info(self):
        """Print model information."""
        print(f"\n{'─'*40}")
        print(f"Model Configuration")
        print(f"{'─'*40}")
        print(f"Device: {self.device}")
        print(f"Modalities: {self.config.get('modalities', 'unknown')}")
        print(f"Output type: {self.config.get('output_type', 'unknown')}")
        if self.class_names:
            print(f"Classes: {len(self.class_names)}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'─'*40}\n")


class SimpleTokenizer:
    """Fallback simple tokenizer when transformers not available."""
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        self._build_basic_vocab()
    
    def _build_basic_vocab(self):
        """Build basic vocabulary."""
        special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        for i, token in enumerate(special):
            self.vocab[token] = i
            self.inv_vocab[i] = token
    
    def encode(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode texts to token IDs."""
        batch = []
        for text in texts:
            tokens = text.lower().split()[:max_length-2]
            ids = [self.vocab.get('[CLS]', 1)]
            for token in tokens:
                # Simple hash-based ID
                token_id = hash(token) % (self.vocab_size - 4) + 4
                ids.append(token_id)
            ids.append(self.vocab.get('[SEP]', 2))
            
            # Pad
            while len(ids) < max_length:
                ids.append(0)
            batch.append(ids[:max_length])
        
        return torch.tensor(batch, dtype=torch.long)
    
    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if ids.dim() > 1:
            ids = ids[0]
        tokens = []
        for id in ids.tolist():
            if skip_special_tokens and id < 4:
                continue
            tokens.append(self.inv_vocab.get(id, f'[{id}]'))
        return ' '.join(tokens)
