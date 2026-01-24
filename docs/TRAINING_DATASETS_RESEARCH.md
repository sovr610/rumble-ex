# Brain-Inspired AI Training Dataset Research

> **Purpose**: This document provides comprehensive research on production-level datasets for training each phase of the brain-inspired AI architecture. Designed for Claude deep research to find and validate actual datasets.

**Generated**: January 23, 2026  
**Status**: In Progress

---

## Table of Contents

1. [Phase 1: SNN Core](#phase-1-snn-core)
2. [Phase 2: Event-Driven Sensory Processing](#phase-2-event-driven-sensory-processing)
3. [Phase 3: Hierarchical Temporal Memory (HTM)](#phase-3-hierarchical-temporal-memory-htm)
4. [Phase 4: Global Workspace & Working Memory](#phase-4-global-workspace--working-memory)
5. [Phase 5: Active Inference Decision System](#phase-5-active-inference-decision-system)
6. [Phase 6: Neuro-Symbolic Reasoning](#phase-6-neuro-symbolic-reasoning)
7. [Phase 7: Meta-Learning & Plasticity](#phase-7-meta-learning--plasticity)

---

## Phase 1: SNN Core

### Architecture Summary

The SNN Core implements Leaky Integrate-and-Fire (LIF) neurons with surrogate gradients for backpropagation through spikes.

**Key Components:**
- **Neuron Model**: LIF with membrane decay (β=0.9)
- **Surrogate Gradient**: Arctangent (α=2.0) for gradient flow through discrete spikes
- **Encoding**: Rate coding over 25 timesteps
- **Architecture**: Feedforward SNN with layers [784 → 800 → 400 → 10] for MNIST
- **Training Method**: Supervised with CrossEntropyLoss on rate-coded output (spike sum)

**Current Training**: MNIST (28×28 grayscale handwritten digits)
- 60,000 training / 10,000 test samples
- 10 classes (digits 0-9)
- **Target Accuracy**: 98%

### Production Dataset Requirements

For production-level SNN training, datasets should:
1. Have temporal/spike-based representations (neuromorphic format)
2. Test spatio-temporal processing capabilities
3. Include both static-to-spike converted data AND native event data
4. Span multiple modalities (vision, audio)

### Recommended Production Datasets

#### Vision (Static-to-Spike Conversion)

| Dataset | Classes | Samples | Resolution | Description |
|---------|---------|---------|------------|-------------|
| **MNIST** | 10 | 70,000 | 28×28 | Baseline validation (current) |
| **Fashion-MNIST** | 10 | 70,000 | 28×28 | Harder clothing classification |
| **CIFAR-10** | 10 | 60,000 | 32×32×3 | Natural images, RGB |
| **CIFAR-100** | 100 | 60,000 | 32×32×3 | Fine-grained classification |
| **ImageNet-1K** | 1000 | 1.2M+ | 224×224×3 | Large-scale benchmark |

#### Neuromorphic Vision (Native Spike Data)

| Dataset | Classes | Samples | Sensor | Description |
|---------|---------|---------|--------|-------------|
| **N-MNIST** | 10 | 70,000 | DVS128 | Neuromorphic MNIST via saccades |
| **DVS-CIFAR10** | 10 | 10,000 | DVS128 | DVS recordings of CIFAR images |
| **N-Caltech101** | 101 | 8,242 | ATIS | Event-based Caltech-101 |
| **DVS-Gesture** | 11 | 1,342 | DVS128 | Hand gestures for action recognition |
| **N-Cars** | 2 | 24,029 | ATIS | Car vs background classification |

#### Neuromorphic Audio (Native Spike Data)

| Dataset | Classes | Samples | Input Channels | Description |
|---------|---------|---------|----------------|-------------|
| **SHD** (Spiking Heidelberg Digits) | 20 | 10,420 | 700 (cochlea) | Spoken digits as spike trains |
| **SSC** (Spiking Speech Commands) | 35 | 105,829 | 700 (cochlea) | Speech commands as spike trains |
| **TIMIT** | phonemes | 6,300 | variable | Phoneme recognition benchmark |

### Dataset Sources & Downloads

1. **N-MNIST**: https://www.garrickorchard.com/datasets/n-mnist
2. **DVS-CIFAR10**: SpikingJelly loader - `spikingjelly.datasets.dvs128_gesture`
3. **Heidelberg Spiking Datasets**: https://ieee-dataport.org/open-access/heidelberg-spiking-datasets
   - DOI: 10.21227/51gn-m114
   - Direct: https://compneuro.net/datasets/
4. **DVS-Gesture**: Included in SpikingJelly and snnTorch
5. **N-Caltech101**: https://www.garrickorchard.com/datasets/n-caltech101

### Training Methodology

1. **Rate Coding**: Convert static images to spike trains via Poisson encoding
2. **Direct Spike Input**: Use neuromorphic data directly (N-MNIST, DVS-*)
3. **Surrogate Gradient**: Use arctangent or fast sigmoid for backprop
4. **Loss Function**: CrossEntropyLoss on summed output spikes
5. **Optimizer**: Adam with lr=1e-3, weight decay for regularization

### Expected Benchmarks

| Dataset | Architecture | Target Accuracy |
|---------|--------------|-----------------|
| MNIST | FC-SNN | 98%+ |
| MNIST | Conv-SNN | 99%+ |
| N-MNIST | SNN | 98%+ |
| DVS-CIFAR10 | SNN | 70-75% |
| SHD | Recurrent SNN | 85-92% |
| SSC | Recurrent SNN | 70-80% |

---

## Phase 2: Event-Driven Sensory Processing

### Architecture Summary

The Event-Driven Vision Encoder processes neuromorphic event streams from DVS (Dynamic Vision Sensor) cameras.

**Key Components:**
- **Input Format**: Event streams (x, y, t, polarity) with ON/OFF channels
- **Representation**: Voxel grid or frame-based integration
- **Encoder**: Spiking convolutional blocks with LIF neurons
- **Output**: 512-dim feature vector for Global Workspace
- **Training Method**: Supervised classification with CrossEntropyLoss

**Current Training**: Synthetic DVS-like data or DVS-CIFAR10
- SpikingJelly's CIFAR10DVS loader
- 10 classes, 10,000 samples
- Resolution: 128×128 or 32×32 (resized)
- **Target Accuracy**: 75%+

### Production Dataset Requirements

For production-level event-driven training, datasets should:
1. Be captured with real event cameras (DVS, ATIS, etc.)
2. Include temporal event structure (not just converted frames)
3. Cover diverse lighting conditions and motion speeds
4. Include action/gesture recognition for spatio-temporal learning

### Recommended Production Datasets

#### Object/Image Classification

| Dataset | Classes | Samples | Resolution | Camera | Description |
|---------|---------|---------|------------|--------|-------------|
| **N-MNIST** | 10 | 70,000 | 34×34 | DVS128 | MNIST via saccadic eye movement |
| **DVS-CIFAR10** | 10 | 10,000 | 128×128 | DVS128 | CIFAR images on LCD monitor |
| **N-Caltech101** | 101 | 8,242 | 240×180 | ATIS | Caltech-101 via saccades |
| **N-Cars** | 2 | 24,029 | 120×100 | ATIS | Cars vs background |
| **DVS-SLR** | varies | varies | varies | DVS | Sign language recognition |

#### Action/Gesture Recognition

| Dataset | Classes | Samples | Resolution | Camera | Description |
|---------|---------|---------|------------|--------|-------------|
| **DVS128-Gesture** | 11 | 1,342 | 128×128 | DVS128 | Hand gesture classification |
| **DailyDVS-200** (ECCV 2024) | 200 | 22,046 | 320×240 | DVXplorer Lite | Large-scale action recognition |
| **ASL-DVS** | 24 | 100,800 | 240×180 | ATIS | American Sign Language letters |
| **NavGesture** | 6 | varies | 304×240 | DVS | Navigation gestures |
| **HARDVS** | varies | large | varies | varies | High-speed action recognition |

#### Automotive/Robotics

| Dataset | Task | Samples | Resolution | Description |
|---------|------|---------|------------|-------------|
| **DSEC** | Stereo depth | 53 sequences | 640×480 | Driving stereo events |
| **MVSEC** | Optical flow | multiple | 346×260 | Multi-vehicle stereo events |
| **EV-IMO** | Motion segmentation | varies | varies | Independent motion detection |

### Dataset Sources & Downloads

1. **SpikingJelly Datasets**: Built-in loaders for most neuromorphic datasets
   ```python
   from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
   from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
   ```

2. **Tonic Library**: https://github.com/neuromorphs/tonic
   - Unified interface for neuromorphic datasets
   - `pip install tonic`

3. **DailyDVS-200**: https://github.com/QiWang233/DailyDVS-200
   - 200 action categories, 22,000+ sequences
   - ECCV 2024 benchmark

4. **N-MNIST / N-Caltech101**: https://www.garrickorchard.com/datasets/

5. **IBM DVS-Gesture**: Request from IBM Research

### Training Methodology

1. **Event-to-Frame Conversion**: Integrate events into temporal bins
   - Voxel grid representation (T bins × 2 polarities × H × W)
   - Time-surface representation
   
2. **Data Augmentation**: Use Neuromorphic Data Augmentation (NDA)
   - Geometric transforms preserving event structure
   - Temporal jittering
   - Random polarity flips

3. **Loss Function**: CrossEntropyLoss on classification head

4. **Preprocessing**:
   ```python
   # SpikingJelly frame integration
   train_set = CIFAR10DVS(root, train=True, data_type='frame', 
                          frames_number=16, split_by='number')
   ```

### Expected Benchmarks

| Dataset | Architecture | Target Accuracy |
|---------|--------------|-----------------|
| N-MNIST | Conv-SNN | 99%+ |
| DVS-CIFAR10 | SNN + NDA | 70-80% |
| DVS128-Gesture | Recurrent SNN | 95%+ |
| DailyDVS-200 | SNN/Transformer | 50-70% (new benchmark) |
| N-Caltech101 | Conv-SNN | 80-85% |

---

## Phase 3: Hierarchical Temporal Memory (HTM)

### Architecture Summary

The HTM layer implements biologically-inspired sequence learning and anomaly detection using Sparse Distributed Representations (SDRs).

**Key Components:**
- **Spatial Pooler**: Converts input to sparse binary representations (2% sparsity)
- **Temporal Memory**: Learns sequences through dendritic connections between cells
- **Cells**: column_count × cells_per_column (256 × 8 = 2048 cells default)
- **Learning**: Online Hebbian learning (no backprop)
- **Anomaly Score**: Measures prediction failure rate

**Current Training**: Synthetic repeating sequences
- 200 sequences, 30 timesteps each
- 5 distinct repeating patterns
- 128-dim sparse input (~4% active bits)
- **Target Accuracy**: 90%+ prediction accuracy

### Production Dataset Requirements

For production-level HTM training, datasets should:
1. Have strong sequential/temporal dependencies
2. Include anomaly labels for detection benchmarking
3. Support online streaming evaluation (real-time detection)
4. Cover diverse domains (industrial, network, medical, financial)

### Recommended Production Datasets

#### Anomaly Detection Benchmarks

| Dataset | Series | Length | Anomalies | Domain | Description |
|---------|--------|--------|-----------|--------|-------------|
| **NAB** (Numenta) | 58 | varies | labeled | Mixed | Standard HTM benchmark with 7 categories |
| **TSB-AD** (NeurIPS 2024) | 1,070 | varies | labeled | Mixed | Largest curated benchmark from 40 sources |
| **Yahoo S5** | 367 | 1,400-1,680 | synthetic | Metrics | Server metrics with synthetic anomalies |
| **UCR Anomaly Archive** | 250+ | varies | labeled | Mixed | Extension of UCR time series archive |

#### Industrial Control Systems

| Dataset | Sensors | Duration | Attack Types | Description |
|---------|---------|----------|--------------|-------------|
| **SWaT** (Secure Water) | 51 | 11 days | 41 attacks | Water treatment testbed |
| **WADI** | 127 | 16 days | 15 attacks | Water distribution system |
| **BATADAL** | varies | varies | labeled | Water network intrusion detection |
| **HAI** | 79 | varies | labeled | Hardware-in-loop testbed |

#### Time Series Classification/Prediction

| Dataset | Series | Classes | Length | Description |
|---------|--------|---------|--------|-------------|
| **UCR Archive** | 128 datasets | varies | varies | Standard TS classification benchmark |
| **UEA Archive** | 30 datasets | varies | varies | Multivariate time series |
| **Monash Forecasting** | 25+ | N/A | varies | Time series forecasting benchmark |
| **TSDB** | 30+ | varies | varies | Time series data mining benchmark |

#### Domain-Specific Sequences

| Dataset | Type | Samples | Description |
|---------|------|---------|-------------|
| **ECG5000** | Medical | 5,000 | ECG heartbeat sequences |
| **MIT-BIH Arrhythmia** | Medical | 48 | Annotated ECG recordings |
| **NYC Taxi** | Traffic | 10,320 | NYC taxi demand time series |
| **Power Demand** | Energy | 35,064 | Electricity consumption |
| **Machine Temperature** | Industrial | 22,695 | Industrial sensor readings |

### Dataset Sources & Downloads

1. **NAB (Numenta Anomaly Benchmark)**:
   - GitHub: https://github.com/numenta/NAB
   - Kaggle: https://www.kaggle.com/datasets/boltzmannbrain/nab
   - 58 time series across 7 categories: artificial, AWS, AdExchange, KnownCause, realTraffic, realTweets, realKnownCause

2. **TSB-AD (NeurIPS 2024)**:
   - GitHub: https://github.com/TheDatumOrg/TSB-AD
   - 1,070 curated time series from 40 diverse sources
   - Best evaluated metric: VUS-PR (Volume Under Surface - Precision Recall)

3. **UCR Time Series Archive**:
   - URL: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
   - 128 univariate + 30 multivariate datasets
   - Password protected (see briefing document)

4. **SWaT/WADI** (iTrust Singapore):
   - Request: https://itrust.sutd.edu.sg/itrust-labs_datasets/
   - Industrial control system benchmark
   - Requires research agreement

5. **Yahoo S5 Anomaly Dataset**:
   - URL: https://webscope.sandbox.yahoo.com/
   - Synthetic and real anomalies in server metrics

### Training Methodology

1. **Online Learning**: HTM learns continuously without epochs
   - Feed sequences one timestep at a time
   - Spatial pooler stabilizes SDRs
   - Temporal memory learns transitions

2. **Sequence Prediction Evaluation**:
   ```python
   # Prediction accuracy = overlap between predicted and actual columns
   predicted_columns = (predictive_cells > threshold)
   actual_columns = spatial_pooler(current_input)
   accuracy = overlap(predicted_columns, actual_columns) / num_active
   ```

3. **Anomaly Detection**:
   - Anomaly score = 1 - prediction_accuracy
   - Anomaly likelihood = smoothed probability over window
   - Use NAB scoring profile (standard, reward_low_FP, reward_low_FN)

4. **Hyperparameter Tuning**:
   - `column_count`: 2048 (production), 256 (testing)
   - `cells_per_column`: 32 (production), 8 (testing)
   - `sparsity`: 0.02 (2% active columns)
   - `activation_threshold`: 13 (segments needed for prediction)

### Expected Benchmarks

| Dataset | Metric | Target Score |
|---------|--------|--------------|
| NAB (standard profile) | NAB Score | 65-70 |
| NAB (all profiles avg) | NAB Score | 60-65 |
| SWaT | F1 Score | 0.80+ |
| WADI | F1 Score | 0.75+ |
| Synthetic sequences | Prediction Acc | 90%+ |
| UCR classification | Accuracy | Varies by dataset |

### HTM-Specific Libraries

1. **htm.core**: https://github.com/htm-community/htm.core
   - Official Numenta implementation in C++ with Python bindings
   - `pip install htm.core` (may require compilation)

2. **NuPIC**: https://github.com/numenta/nupic-legacy
   - Original Python implementation (deprecated)

3. **PyTorch Fallback**: Current implementation in `brain_ai/temporal/htm.py`
   - Pure PyTorch implementation for GPU training
   - Compatible API with htm.core

---

## Phase 4: Global Workspace & Working Memory

### Architecture Summary

The Global Workspace implements Global Workspace Theory (GWT) for multi-modal integration with Liquid Neural Networks (CfC/LTC) for temporal working memory.

**Key Components:**
- **Global Workspace**: Attention-based competition for workspace access (capacity ~7 items)
- **Modality Projections**: Project different inputs into common workspace dimension (256-512)
- **Working Memory**: Liquid Neural Networks (CfC or LTC from ncps) for temporal context
- **Information Broadcast**: Winners broadcast information back to all modules
- **Classifier**: Multi-modal fusion for classification

**Current Training**: Synthetic multi-modal sequences
- 5000 training / 1000 test samples
- 3 modalities × 128 dimensions each
- 20 timesteps per sequence
- 5 classes (determined by cross-modal correlation + temporal patterns)
- **Target Accuracy**: Multi-modal temporal fusion

### Production Dataset Requirements

For production-level Global Workspace training, datasets should:
1. Include multiple synchronized modalities (audio, video, text)
2. Require cross-modal reasoning (answers from combining modalities)
3. Have temporal structure (sequential or video data)
4. Test attention-based selection (which modality matters when)

### Recommended Production Datasets

#### Multi-Modal Sentiment & Emotion (Audio + Video + Text)

| Dataset | Samples | Modalities | Task | Description |
|---------|---------|------------|------|-------------|
| **CMU-MOSEI** | 22,777 | Audio, Video, Text | Sentiment + Emotion | Largest multimodal sentiment dataset, 65+ hours |
| **CMU-MOSI** | 2,199 | Audio, Video, Text | Sentiment | Opinion video clips with per-frame annotations |
| **CH-SIMS** | 2,281 | Audio, Video, Text | Sentiment | Mandarin Chinese multimodal sentiment |
| **UR-FUNNY** | 16,514 | Audio, Video, Text | Humor | Humor detection in TED talks |
| **MUSTARD** | 690 | Audio, Video, Text | Sarcasm | Sarcasm detection in TV shows |

#### Video Question Answering (Vision + Audio + Language)

| Dataset | Videos | QA Pairs | Avg Length | Description |
|---------|--------|----------|------------|-------------|
| **ActivityNet-QA** | 5,800 | 58,000 | 180s | Complex web video understanding |
| **MSRVTT-QA** | 10,000 | 243,680 | 15s | Video description QA |
| **MSVD-QA** | 1,970 | 50,505 | 10s | Short video QA |
| **How2QA** | 22,000 | 44,000 | 60s | Instructional video QA |
| **CinePile** | 500+ | 300K+ | Long | Long video understanding benchmark |

#### Multi-Modal Fusion Benchmarks (MultiBench)

| Dataset | Modalities | Samples | Task | Research Area |
|---------|------------|---------|------|---------------|
| **AV-MNIST** | Image + Audio | 70,000 | Digit classification | Multimedia |
| **Kinetics-400** | Video + Audio + Optical Flow | 306,245 | Action recognition | Multimedia |
| **MM-IMDB** | Text + Image | 25,959 | Movie genre | Multimedia |
| **MIMIC** | Text + Tabular | 36,212 | Mortality, ICD-9 | Healthcare |
| **MuJoCo Push** | Image + Force + Proprio | 37,990 | Object pose | Robotics |
| **ENRICO** | Image + Screenshot | 1,460 | Design interface | HCI |

#### Emotion Recognition (Audio + Visual)

| Dataset | Duration | Annotations | Conditions | Description |
|---------|----------|-------------|------------|-------------|
| **Aff-Wild2** | 558 clips (2M frames) | Valence/Arousal/Action Units | In-the-wild | Large facial emotion dataset |
| **RECOLA** | 9.5 hours | Valence/Arousal | Laboratory | Audio + Visual + Physiological |
| **SEMAINE** | 5 hours | Valence/Arousal | Laboratory | Audio + Visual + Text |
| **SEWA** | 30 hours | Valence/Arousal | In-the-wild | Audio + Visual + Text |

### Dataset Sources & Downloads

1. **CMU-MOSEI / CMU-MOSI**:
   - URL: http://multicomp.cs.cmu.edu/resources/
   - SDK: https://github.com/A2Zadeh/CMU-MultimodalSDK
   - 23K+ annotated video clips

2. **MultiBench**:
   - GitHub: https://github.com/pliang279/MultiBench
   - Docs: https://cmu-multicomp-lab.github.io/multibench/
   - NeurIPS 2021 - 15 datasets, 10 modalities, 20 tasks

3. **ActivityNet-QA**:
   - URL: https://github.com/MILVLG/activitynet-qa
   - 58K QA pairs on 5.8K videos

4. **Kinetics-400/600/700**:
   - URL: https://www.deepmind.com/open-source/kinetics
   - Large-scale video action recognition

5. **VideoQA Resources**:
   - GitHub: https://github.com/chakravarthi589/Video-Question-Answering_Resources
   - Comprehensive list of VideoQA datasets and methods

### Training Methodology

1. **Modality Encoding**: Project each modality to common dimension
   ```python
   vision_features = vision_encoder(video_frames)  # (B, T, 512)
   audio_features = audio_encoder(mel_spectrogram)  # (B, T, 512)
   text_features = text_encoder(tokens)  # (B, T, 512)
   ```

2. **Cross-Modal Attention**: Use transformer attention for fusion
   - Concatenate modality features: (B, T, 3*512)
   - Apply multi-head self-attention
   - Extract attended features per modality

3. **Workspace Competition**:
   - Compute salience scores per modality
   - Top-K selection (K ≤ 7 for capacity limit)
   - Weighted fusion based on attention

4. **Temporal Processing**: Liquid NN for working memory
   - CfC/LTC for continuous-time dynamics
   - GRU fallback if ncps unavailable

5. **Loss Functions**:
   - Classification: CrossEntropyLoss
   - Regression: MSELoss (for sentiment)
   - Auxiliary: Reconstruction loss for missing modalities

### Expected Benchmarks

| Dataset | Metric | Target Score |
|---------|--------|--------------|
| CMU-MOSEI 7-class | Accuracy | 50-55% |
| CMU-MOSEI Binary | Accuracy | 80-85% |
| CMU-MOSI Binary | Accuracy | 83-87% |
| AV-MNIST | Accuracy | 99%+ |
| ActivityNet-QA | Accuracy | 40-45% |
| MSRVTT-QA | Accuracy | 45-50% |

### Key Libraries for Multi-Modal Learning

1. **MultiBench/MultiZoo**: https://github.com/pliang279/MultiBench
   - Standardized multimodal methods toolkit
   - Pre-implemented fusion paradigms

2. **CMU-MultimodalSDK**: https://github.com/A2Zadeh/CMU-MultimodalSDK
   - Data loaders for CMU datasets
   - Alignment utilities

3. **ncps (Liquid NNs)**: https://github.com/mlech26l/ncps
   - CfC and LTC implementations
   - `pip install ncps`

---

## Phase 5: Active Inference Decision System

### Architecture Summary

The Active Inference agent implements decision-making via Expected Free Energy (EFE) minimization, balancing pragmatic (goal-directed) and epistemic (exploration) value.

**Key Components:**
- **State Encoder**: q(s|o) - Encodes observations to latent state distribution
- **Generative Model**: P(o|s) likelihood + P(s'|s,a) transition model
- **Preferences**: Learned goal specifications (preferred observations)
- **EFE Computation**: Balances goal achievement + information gain
- **Planning Horizon**: Multi-step lookahead (3 steps default)

**Current Training**: Grid-world navigation
- 5×5 to 8×8 grid with stochastic transitions
- Goal-seeking with partial observability
- 5 actions: up, right, down, left, stay
- **Target**: Learn optimal navigation policy

### Production Dataset Requirements

For production-level Active Inference training, datasets should:
1. Support sequential decision-making with clear goals
2. Include uncertainty (partial observability, stochastic dynamics)
3. Allow both online (interactive) and offline (batch) training
4. Test exploration-exploitation trade-off

### Recommended Production Datasets

#### Offline RL Benchmarks (D4RL / Minari)

| Dataset | Environment | Episodes | Policy Types | Description |
|---------|-------------|----------|--------------|-------------|
| **HalfCheetah-v2** | MuJoCo | ~1M steps | expert/medium/random | Locomotion control |
| **Hopper-v2** | MuJoCo | ~1M steps | expert/medium/random | Hopping locomotion |
| **Walker2D-v2** | MuJoCo | ~1M steps | expert/medium/random | Bipedal walking |
| **Ant-v2** | MuJoCo | ~1M steps | expert/medium/random | Quadruped locomotion |
| **AntMaze** | MuJoCo | varies | diverse | Goal-conditioned navigation |
| **FrankaKitchen** | MuJoCo | varies | partial/complete | Multi-task manipulation |

#### Navigation & Exploration

| Dataset/Environment | Type | Features | Description |
|---------------------|------|----------|-------------|
| **MiniGrid** | Gymnasium | Procedural, sparse rewards | Grid-world navigation |
| **Minigrid-Fourrooms** | Gymnasium | Exploration required | Room navigation |
| **PointMaze** | D4RL | Goal-conditioned | Continuous maze navigation |
| **KeyDoor** | MiniGrid | Multi-step goals | Key collection task |
| **Dynamic Obstacles** | MiniGrid | Partial observability | Avoid moving obstacles |

#### Robotic Control & Manipulation

| Dataset | Robot | Task | Samples | Description |
|---------|-------|------|---------|-------------|
| **Meta-World** | Sawyer | 50 tasks | varies | Multi-task manipulation |
| **RLBench** | Multi-arm | 100+ tasks | varies | Vision-based manipulation |
| **RoboPush** | MuJoCo | Push/contact | 37,990 | Object pose prediction |
| **Vision&Touch** | Various | Multi-modal | 147,000 | Contact + robot pose |

#### Atari & Video Games

| Dataset | Games | Quality Levels | Description |
|---------|-------|----------------|-------------|
| **Atari 100K** | 26 games | limited data | Sample-efficient benchmark |
| **DQN Replay** | 60 games | full replays | Experience from DQN training |
| **Procgen** | 16 games | procedural | Generalization benchmark |

### Dataset Sources & Downloads

1. **D4RL (Original)**:
   - GitHub: https://github.com/Farama-Foundation/d4rl
   - Paper: "D4RL: Datasets for Deep Data-Driven RL"
   - Note: Being migrated to Minari

2. **Minari (New Standard)**:
   - GitHub: https://github.com/Farama-Foundation/Minari
   - `pip install minari`
   - Standardized offline RL datasets compatible with Gymnasium
   ```python
   import minari
   dataset = minari.load_dataset("halfcheetah-expert-v2")
   ```

3. **Gymnasium / MiniGrid**:
   - Gymnasium: https://github.com/Farama-Foundation/Gymnasium
   - MiniGrid: https://github.com/Farama-Foundation/MiniGrid
   - `pip install gymnasium minigrid`

4. **Gymnasium-Robotics**:
   - GitHub: https://github.com/Farama-Foundation/Gymnasium-Robotics
   - Includes Fetch, ShadowHand, Maze environments

5. **pymdp (Active Inference)**:
   - GitHub: https://github.com/infer-actively/pymdp
   - Active Inference agent implementation
   - `pip install inferactively-pymdp`

### Training Methodology

1. **Online Training (Interactive)**:
   ```python
   import gymnasium as gym
   env = gym.make("MiniGrid-FourRooms-v0")
   obs, info = env.reset()
   for step in range(max_steps):
       action = agent.select_action(obs)
       obs, reward, done, truncated, info = env.step(action)
   ```

2. **Offline Training (From Dataset)**:
   ```python
   import minari
   dataset = minari.load_dataset("antmaze-large-diverse-v2")
   for episode in dataset.iterate_episodes():
       states = episode.observations
       actions = episode.actions
       rewards = episode.rewards
       # Train on batch
   ```

3. **Active Inference Components**:
   - **State Encoder**: Train VAE on observations
   - **Generative Model**: Predict next state + observation
   - **EFE Minimization**: Select actions minimizing expected free energy
   ```python
   # EFE = -pragmatic_value - epistemic_value
   pragmatic = preference_distance(predicted_obs, goal_obs)
   epistemic = state_entropy(predicted_state_distribution)
   efe = -(pragmatic_weight * pragmatic + epistemic_weight * epistemic)
   ```

4. **Preference Learning**:
   - Learn from reward signals: `preference(obs) ≈ reward`
   - Or set manually: `agent.set_preference(goal_observation)`

### Expected Benchmarks

| Environment | Metric | Target Score |
|-------------|--------|--------------|
| Grid-world 5×5 | Success Rate | 90%+ |
| Grid-world 8×8 | Success Rate | 80%+ |
| HalfCheetah-expert | Normalized Score | 100+ |
| AntMaze-medium | Success Rate | 70%+ |
| MiniGrid-FourRooms | Success Rate | 80%+ |
| PointMaze | Success Rate | 60%+ |

### Active Inference Libraries

1. **pymdp**: https://github.com/infer-actively/pymdp
   - Discrete state-space Active Inference
   - POMDP-style agents

2. **SPM (Statistical Parametric Mapping)**:
   - MATLAB implementation by Karl Friston's group
   - Original Active Inference framework

3. **Custom PyTorch Implementation**: Current `brain_ai/decision/active_inference.py`
   - Neural network-based continuous Active Inference
   - Compatible with standard RL environments

---

## Phase 6: Neuro-Symbolic Reasoning

### Architecture Summary

The Neuro-Symbolic Reasoning module implements dual-process theory (System 1/System 2) with differentiable fuzzy logic for interpretable multi-step reasoning.

**Key Components:**
- **System 1**: Fast, automatic processing for familiar patterns
- **System 2**: Deliberate, multi-step reasoning for novel problems
- **Fuzzy Logic**: Differentiable AND, OR, NOT, IMPLIES, FORALL, EXISTS
- **Rule Networks**: Learned IF-THEN rules with confidence
- **Confidence Routing**: Routes to System 2 when confidence < 0.7
- **Reasoning Trace**: Interpretable step-by-step proof

**Current Training**: Synthetic logical reasoning
- Propositional logic (AND, OR, IMPLIES)
- Transitive reasoning (A > B, B > C → A > C)
- Multi-step deduction chains (2-4 hops)
- Fuzzy threshold tasks
- **Target Accuracy**: 85%+ with interpretable traces

### Production Dataset Requirements

For production-level neuro-symbolic training, datasets should:
1. Require multi-step logical inference
2. Have verifiable ground-truth proofs
3. Test compositionality and generalization
4. Include both textual and visual reasoning

### Recommended Production Datasets

#### Text-Based Logical Reasoning

| Dataset | Samples | Hops | Task | Description |
|---------|---------|------|------|-------------|
| **bAbI Tasks** | 20 tasks × 10K | 1-5 | QA | Facebook's reasoning benchmark |
| **ProofWriter** | varies | 0-5 | Deduction | Natural language proofs with facts/rules |
| **CLUTRR** | 30K+ | 2-10 | Relation | Family relationship reasoning |
| **FOLIO** | 1.4K | multi | FOL | Human-annotated first-order logic |
| **RuleTaker** | 100K+ | 0-5 | Deduction | Synthetic rule-based reasoning |
| **SLR-Bench** (2024) | 19K | 1-20 | Curriculum | 20-level logical complexity curriculum |

#### Visual Reasoning

| Dataset | Images | QA Pairs | Task | Description |
|---------|--------|----------|------|-------------|
| **CLEVR** | 100K | 853K | Compositional | 3D shapes with spatial/comparative reasoning |
| **CLEVR-Humans** | 100K | 32K | Compositional | Human-generated questions on CLEVR |
| **GQA** | 113K | 22M | Compositional | Real images with scene graphs |
| **VQA v2** | 200K | 1.1M | Open-ended | Visual question answering |
| **CLEVR-POC** | varies | varies | Constraint | Reasoning under partial observability |
| **ClevrSkills** | varies | varies | Robotics | Compositional reasoning for manipulation |

#### Knowledge Graph Reasoning

| Dataset | Entities | Relations | Task | Description |
|---------|----------|-----------|------|-------------|
| **FB15k-237** | 14,541 | 237 | Link prediction | Freebase subset |
| **WN18RR** | 40,943 | 11 | Link prediction | WordNet subset |
| **NELL-995** | 75,492 | 200 | Path reasoning | Never-Ending Language Learning |
| **Countries** | 272 | 2 | Inductive | Geographic relations |

#### Mathematical Reasoning

| Dataset | Samples | Task | Description |
|---------|---------|------|-------------|
| **GSM8K** | 8,500 | Math word problems | Grade school math |
| **MATH** | 12,500 | Advanced math | Competition-level problems |
| **AQuA-RAT** | 100K | Algebra | Algebraic word problems |
| **MathQA** | 37K | Multi-step math | Diverse math operations |

### Dataset Sources & Downloads

1. **bAbI Tasks (Facebook AI)**:
   - URL: https://research.fb.com/downloads/babi/
   - 20 QA tasks testing different reasoning types
   - Tasks 1-10: basic, Tasks 11-20: require memory

2. **ProofWriter / RuleTaker**:
   - URL: https://allenai.org/data/proofwriter
   - Natural language theories + proofs
   - Depths D0-D5 (0-5 reasoning hops)

3. **CLUTRR**:
   - GitHub: https://github.com/facebookresearch/clutrr
   - Family relationship reasoning
   - Tests systematic generalization

4. **CLEVR**:
   - URL: https://cs.stanford.edu/people/jcjohns/clevr/
   - Synthetic 3D scenes + questions
   - Tests compositional reasoning

5. **SLR-Bench (2024)**:
   - GitHub: Released with SLR framework
   - 19K tasks across 20 curriculum levels
   - Automated logical reasoning synthesis

6. **FOLIO**:
   - GitHub: https://github.com/Yale-LILY/FOLIO
   - Human-annotated first-order logic
   - Wikipedia-sourced premises

### Training Methodology

1. **System 1/2 Routing**:
   ```python
   s1_logits, s1_confidence = system1(x)
   if s1_confidence < confidence_threshold:
       s2_output = system2(x, max_iterations=5)
       final_output = s2_output
   else:
       final_output = s1_logits
   ```

2. **Fuzzy Logic Operations**:
   ```python
   fuzzy = FuzzyLogic(logic_type="product")
   
   # Conjunction: a AND b
   result = fuzzy.AND(a, b)  # a * b
   
   # Implication: a -> b  
   result = fuzzy.IMPLIES(a, b)  # 1 - a + a*b
   
   # Universal quantifier: FOR ALL x
   result = fuzzy.FORALL(x, dim=-1)  # x.prod(dim)
   ```

3. **Multi-Step Reasoning**:
   ```python
   state = encoder(input)
   for step in range(max_steps):
       state = reasoning_gru(state, context)
       # Apply symbolic operations
       and_result = predicate_and(state)
       or_result = predicate_or(state)
       state = state + symbolic_features
   output = classifier(state)
   ```

4. **Proof Generation**:
   - Track reasoning trace at each step
   - Verify proof against ground truth
   - Loss includes proof fidelity term

### Expected Benchmarks

| Dataset | Metric | Target Score |
|---------|--------|--------------|
| bAbI (average) | Accuracy | 95%+ |
| ProofWriter D3 | Accuracy | 95%+ |
| ProofWriter D5 | Accuracy | 85%+ |
| CLUTRR (k=3) | Accuracy | 90%+ |
| CLUTRR (k=10) | Accuracy | 70%+ |
| CLEVR | Accuracy | 95%+ |
| FOLIO | Accuracy | 80%+ |
| Synthetic logic | Accuracy | 85%+ |

### Neuro-Symbolic Libraries

1. **Logic Tensor Networks (LTN)**: https://github.com/logictensornetworks/LTN
   - Differentiable first-order logic
   - TensorFlow-based

2. **PyReason**: https://github.com/lab-v2/pyreason
   - Temporal logic reasoning
   - Graph-based inference

3. **NeurASP**: Neural Answer Set Programming
   - Combines neural networks + ASP

4. **DiffLog**: Differentiable Datalog
   - End-to-end differentiable logic programming

---

## Phase 7: Meta-Learning & Plasticity

### Architecture Summary

The Meta-Learning module implements MAML for few-shot adaptation with neuromodulatory gating for plasticity control and eligibility traces for biologically-plausible credit assignment.

**Key Components:**
- **MAML**: Model-Agnostic Meta-Learning for rapid task adaptation
- **Inner Loop**: Fast adaptation with 5-10 gradient steps
- **Outer Loop**: Meta-update across task distribution
- **Neuromodulatory Gate**: Controls which parameters adapt
- **Eligibility Traces**: Three-factor Hebbian learning
- **Learning Rate Modulator**: Task-specific learning rates

**Current Training**: Synthetic few-shot classification
- 100 base classes, 64-dim features
- 5-way 1-shot / 5-way 5-shot tasks
- Well-separated prototypes with low noise
- **Target Accuracy**: 80%+ (5-way 1-shot)

### Production Dataset Requirements

For production-level meta-learning training, datasets should:
1. Have large number of classes for task sampling
2. Support episodic (N-way K-shot) evaluation
3. Test generalization to unseen classes
4. Include cross-domain transfer scenarios

### Recommended Production Datasets

#### Image Classification (Few-Shot)

| Dataset | Classes | Images | Resolution | Description |
|---------|---------|--------|------------|-------------|
| **Omniglot** | 1,623 | 32,460 | 105×105 | 50 alphabets, 20 samples/character |
| **mini-ImageNet** | 100 | 60,000 | 84×84 | 100 classes from ImageNet |
| **tiered-ImageNet** | 608 | 779,165 | 84×84 | 34 superclasses, larger scale |
| **CIFAR-FS** | 100 | 60,000 | 32×32 | CIFAR-100 few-shot splits |
| **FC100** | 100 | 60,000 | 32×32 | CIFAR-100 with superclass split |
| **CUB-200** | 200 | 11,788 | varies | Fine-grained bird classification |
| **Stanford Dogs** | 120 | 20,580 | varies | Fine-grained dog breeds |

#### Meta-Dataset (Cross-Domain)

| Dataset | Domains | Task Types | Description |
|---------|---------|------------|-------------|
| **Meta-Dataset** | 10 | Classification | Cross-domain few-shot benchmark |
| **VTAB** | 19 | Transfer | Visual Task Adaptation Benchmark |
| **ORBIT** | Object recognition | Few-shot | Real-world object recognition |

#### Meta-RL Environments

| Environment | Type | Task Distribution | Description |
|-------------|------|-------------------|-------------|
| **Meta-World ML1** | Manipulation | 1 task variants | Single-task meta-learning |
| **Meta-World ML10** | Manipulation | 10 tasks | Multi-task meta-learning |
| **Meta-World ML45** | Manipulation | 45 tasks | Full benchmark |
| **MuJoCo locomotion** | Control | Parameter variations | Varied dynamics |
| **Half-Cheetah Dir/Vel** | Control | Direction/velocity | Goal-conditioned |

#### Few-Shot Regression

| Dataset | Input Dim | Task Type | Description |
|---------|-----------|-----------|-------------|
| **Sinusoid** | 1 | Function fitting | Sine waves with varying amplitude/phase |
| **Harmonic** | 1 | Function fitting | Sum of harmonics |
| **2D Regression** | 2 | Spatial | 2D function approximation |

### Dataset Sources & Downloads

1. **Omniglot**:
   - URL: https://github.com/brendenlake/omniglot
   - 1,623 character classes from 50 alphabets
   - Often augmented with rotations (×4)

2. **mini-ImageNet**:
   - Original splits by Vinyals et al.
   - 64 train / 16 val / 20 test classes
   - Available via torchmeta, learn2learn

3. **tiered-ImageNet**:
   - Larger scale with hierarchical splits
   - 351 train / 97 val / 160 test classes
   - Available via torchmeta

4. **learn2learn**:
   - GitHub: https://github.com/learnables/learn2learn
   - `pip install learn2learn`
   - Includes: Omniglot, mini-ImageNet, tiered-ImageNet, CIFAR-FS, FC100
   ```python
   import learn2learn as l2l
   dataset = l2l.vision.benchmarks.get_tasksets(
       'mini-imagenet',
       train_ways=5, train_samples=5,
       test_ways=5, test_samples=5
   )
   ```

5. **torchmeta**:
   - GitHub: https://github.com/tristandeleu/pytorch-meta
   - `pip install torchmeta`
   - Unified data loaders for meta-learning
   ```python
   from torchmeta.datasets import MiniImagenet
   from torchmeta.utils.data import BatchMetaDataLoader
   dataset = MiniImagenet("data", num_classes_per_task=5, 
                          meta_train=True, download=True)
   ```

6. **Meta-World**:
   - GitHub: https://github.com/Farama-Foundation/Metaworld
   - 50 robotic manipulation tasks
   - `pip install metaworld`

### Training Methodology

1. **MAML Inner Loop** (Task Adaptation):
   ```python
   adapted_params = model.parameters()
   for step in range(inner_steps):
       loss = cross_entropy(model(support_x), support_y)
       grads = torch.autograd.grad(loss, adapted_params, 
                                    create_graph=not first_order)
       adapted_params = [p - inner_lr * g for p, g in zip(adapted_params, grads)]
   ```

2. **MAML Outer Loop** (Meta-Update):
   ```python
   meta_loss = 0
   for task in task_batch:
       adapted_model = inner_loop(model, task.support)
       query_loss = cross_entropy(adapted_model(task.query_x), task.query_y)
       meta_loss += query_loss
   meta_loss.backward()
   meta_optimizer.step()
   ```

3. **Episodic Training**:
   - Sample N classes per episode
   - K support examples per class (training)
   - Q query examples per class (evaluation)
   - Typical: 5-way 1-shot, 5-way 5-shot

4. **Neuromodulatory Gating**:
   ```python
   gate = neuro_gate(task_embedding)  # [0, 1] per parameter
   adapted_params = [p - gate * lr * g for p, g in zip(params, grads)]
   ```

5. **Eligibility Traces** (Three-Factor Learning):
   ```python
   # trace = decay * trace + pre * post
   trace.update(pre_activity, post_activity)
   # ΔW = lr * reward * trace
   weight_update = lr * reward_signal * trace.get()
   ```

### Expected Benchmarks

| Dataset | Setting | Target Accuracy |
|---------|---------|-----------------|
| Omniglot | 5-way 1-shot | 98%+ |
| Omniglot | 20-way 1-shot | 95%+ |
| mini-ImageNet | 5-way 1-shot | 48-55% |
| mini-ImageNet | 5-way 5-shot | 63-70% |
| tiered-ImageNet | 5-way 1-shot | 50-58% |
| tiered-ImageNet | 5-way 5-shot | 68-75% |
| CIFAR-FS | 5-way 5-shot | 76-83% |
| Synthetic | 5-way 1-shot | 80%+ |

### Meta-Learning Libraries

1. **learn2learn**: https://github.com/learnables/learn2learn
   - MAML, Reptile, ProtoNet implementations
   - Meta-RL environments
   - `pip install learn2learn`

2. **torchmeta**: https://github.com/tristandeleu/pytorch-meta
   - Data loaders for all standard benchmarks
   - `pip install torchmeta`

3. **higher**: https://github.com/facebookresearch/higher
   - Differentiable inner loops
   - `pip install higher`

4. **Meta-World**: https://github.com/Farama-Foundation/Metaworld
   - Meta-RL manipulation tasks
   - `pip install metaworld`

---

## Research Notes for Claude Deep Research

### Priority Search Queries

1. "neuromorphic benchmark dataset 2024 2025 spiking neural network"
2. "DVS event camera dataset classification"
3. "spike-based audio classification dataset"
4. "temporal sequence prediction dataset HTM"
5. "multi-modal fusion dataset audio visual"
6. "active inference reinforcement learning environment"
7. "logical reasoning dataset neuro-symbolic AI"
8. "few-shot learning meta-learning benchmark dataset"

### Key Repositories to Check

- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- snnTorch: https://github.com/jeshraghian/snntorch
- Tonic (neuromorphic data): https://github.com/neuromorphs/tonic
- htm.core: https://github.com/htm-community/htm.core

