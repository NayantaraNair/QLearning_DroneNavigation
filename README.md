# Q-Learning Drone Navigation

A reinforcement learning project that uses Q-learning to train autonomous drones to navigate through 2D grid environments with obstacles. Watch AI learn in real-time through an interactive GUI visualization.

## Features

- **Q-Learning Implementation**: Classic reinforcement learning algorithm with configurable hyperparameters
- **Interactive Visualization**: Real-time GUI showing the learning process with color-coded Q-values
- **Parallel Training**: Support for multiple drones learning simultaneously and sharing knowledge
- **Obstacle Zones**: Configurable obstacle patterns including corridors, clusters, and barriers
- **Dynamic Target Setting**: Change goal positions during training to test adaptability
- **Comprehensive Metrics**: Detailed performance analysis including success rates, path efficiency, and learning trends
- **Multiple Training Modes**: Standard Q-learning and binary state space variants

## Demo

The GUI displays:
- **Heatmap**: Cells colored from red (low Q-value) to green (high Q-value) showing learned knowledge
- **Path**: Orange line showing the current greedy policy path
- **Drone**: Red circle indicating current position
- **Obstacles**: Gray blocks that the drone must avoid
- **Start/Goal**: Green and blue markers respectively

## Installation

### Prerequisites

- Python 3.7+
- tkinter (included with most Python installations)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NayantaraNair/QLearning_DroneNavigation.git
cd QLearning_DroneNavigation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Launch the Main GUI

```bash
python gui.py
```

### Launch the Binary State Space GUI

```bash
python binary_gui.py
```

### Launch the Comparison GUI

Compare standard Q-learning vs binary state space approach:

```bash
python comparison_gui.py
```

### GUI Controls

- **Play/Pause**: Toggle training on/off
- **Step Once**: Execute a single training step
- **Reset**: Clear learned knowledge and restart
- **Speed Slider**: Adjust training speed (steps per tick)
- **Parallel Drones**: Set number of drones learning simultaneously (1-5)
- **Target Position**: Change the goal location dynamically

## Project Structure

```
QLearning_DroneNavigation/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses for grid, learning, and simulation
│   ├── drone.py           # Q-learning drone agent implementation
│   ├── grid.py            # Grid environment and obstacle generation
│   ├── simulation.py      # High-level simulation controller
│   ├── trainer.py         # Training orchestration and metrics
│   └── binary_trainer.py  # Binary state space variant
├── test/
│   ├── example_zone_training.py      # Zone-based obstacle training examples
│   ├── obstacle_dense_test.py        # Dense obstacle environment tests
│   ├── performance_comparison.py     # Compare different approaches
│   ├── test_gui_configs.py           # GUI configuration tests
│   └── test_obstacle_zones.py        # Obstacle zone validation
├── gui.py                 # Main Tkinter visualization (2D grid)
├── binary_gui.py          # Binary state space visualization
├── comparison_gui.py      # Side-by-side comparison GUI
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

### Grid Settings

Customize the environment in [src/config.py](src/config.py):

```python
GridSettings(
    rows=20,              # Grid height
    cols=20,              # Grid width
    layers=1,             # Future: 3D support
    start=(0, 0, 0),      # Starting position
    goal=(0, 19, 0),      # Target position
    obstacle_count=7,     # Number of obstacles
    random_seed=11,       # Reproducibility
    obstacle_zones=None   # Optional: constrain obstacle placement
)
```

### Learning Hyperparameters

Tune Q-learning in [src/config.py](src/config.py):

```python
LearningSettings(
    learning_rate=0.3,           # How quickly to update Q-values (alpha)
    discount_factor=0.995,       # Future reward importance (gamma)
    exploration_rate=0.999,      # Initial exploration probability (epsilon)
    exploration_decay=0.99995,   # Epsilon decay rate per step
    min_exploration=0.4,         # Minimum epsilon value
    reward_step=-0.005,          # Penalty per step (encourages efficiency)
    reward_goal=2000.0,          # Reward for reaching goal
    reward_obstacle=-20.0,       # Penalty for hitting obstacle
)
```

### Predefined Obstacle Configurations

The project includes several preset obstacle patterns:

- **Center Cluster**: Obstacles concentrated in the middle
- **Right Corridor**: Obstacles forming a maze on the right side
- **Dual Barriers**: Two separate barrier zones
- **Horizontal Bands**: Horizontal obstacle strips
- **Corner Sparse**: Sparse obstacles in opposite corners
- **Dense Random**: 12% obstacle density across the entire grid

## How It Works

### Q-Learning Algorithm

The drone learns through trial and error using the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Where:
- **s**: current state (position)
- **a**: action taken (up, down, left, right)
- **α**: learning rate
- **r**: reward received
- **γ**: discount factor
- **s'**: next state

### State Representation

**Standard Mode**: State is simply the drone's (row, col) position.

**Binary Mode**: State includes proximity sensors for nearby obstacles, reducing state space complexity.

### Action Space

The drone can move in 4 directions:
- **Up**: (row-1, col)
- **Down**: (row+1, col)
- **Left**: (row, col-1)
- **Right**: (row, col+1)

### Parallel Training

Multiple drones can train simultaneously:
1. Each drone explores independently
2. After each step, Q-tables are synchronized
3. Knowledge is shared across all agents
4. Faster convergence through distributed exploration

## Performance Metrics

The trainer tracks comprehensive statistics:

### Training Overview
- Total episodes completed
- Total steps taken
- Average steps per episode

### Learning Progress
- Early vs. final performance comparison
- Average reward improvement
- Episode length reduction

### Knowledge Acquisition
- States explored (Q-table coverage)
- Exploration rate decay
- Path efficiency vs. Manhattan distance

### Success Analysis
- Overall success rate (reaching goal)
- Collision rate (hitting obstacles)
- Recent performance trends

## Running Tests

Execute the test suite to validate different scenarios:

```bash
# Test obstacle zone configurations
python test/test_obstacle_zones.py

# Run dense obstacle environment
python test/obstacle_dense_test.py

# Compare performance across methods
python test/performance_comparison.py

# Test GUI configurations
python test/test_gui_configs.py
```

## Example Training Results

After sufficient training (varies by configuration):

```
Q-LEARNING TRAINING EVALUATION METRICS
============================================================
TRAINING OVERVIEW:
   Total Episodes: 50000
   Total Steps: 2453891
   Average Steps per Episode: 49.1

LEARNING PROGRESS:
   Early Performance (first 100 episodes):
     - Average Reward: -145.23
     - Best Reward: 2000.00
   Final Performance (last 100 episodes):
     - Average Reward: 1843.67
     - Best Reward: 2000.00
   Learning Improvement: +1988.90

SOLUTION QUALITY:
   Current Path Length: 20
   Manhattan Distance: 19
   Path Efficiency: 95.0%
   SUCCESS: Successfully finds path to goal!
============================================================
```

## Troubleshooting

### tkinter Not Found

**Linux/Ubuntu:**
```bash
sudo apt-get install python3-tk
```

**macOS:** Included with Python installation

**Windows:** Included with Python installation

### Slow Training

- Reduce grid size (e.g., 10x10 instead of 20x20)
- Increase `playback_delay_ms` for smoother visualization
- Decrease `exploration_decay` for faster exploitation
- Use fewer obstacles

### Not Reaching Goal

- Increase `max_episodes` for more training time
- Adjust `exploration_rate` and `exploration_decay`
- Verify goal is reachable (not surrounded by obstacles)
- Increase `reward_goal` value for stronger signal

## Future Enhancements

- 3D grid navigation (layers dimension)
- Deep Q-Networks (DQN) implementation
- Dynamic obstacles that move
- Multiple goals with prioritization
- Export/import trained Q-tables
- Additional visualization modes

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## Author

Created as a demonstration of reinforcement learning fundamentals with practical visualization.

## Acknowledgments

- Based on classic Q-learning algorithm by Watkins & Dayan (1992)
- Inspired by grid-world navigation problems in reinforcement learning literature
