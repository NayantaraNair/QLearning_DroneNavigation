# Q-Learning Drone Navigation

A visual comparison tool for Q-learning algorithms with different state representations. Watch two drones learn to navigate obstacle courses simultaneously - one using simple position-based learning, the other with obstacle awareness.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the comparison GUI (recommended)
python comparison_gui.py

# Or run individual GUIs
python gui.py              # Simple drone only
python binary_gui.py       # Binary obstacle drone only
```

## What This Demonstrates

### Simple Drone (Left Panel)
- **State:** `(row, col)` - knows only its position
- **State space:** 400 states (20×20 grid)
- **Learning:** Discovers obstacles through collision

### Binary Obstacle Drone (Right Panel)
- **State:** `(row, col, obstacle_N, obstacle_S, obstacle_W, obstacle_E)` - position + awareness
- **State space:** 6,400 states (20×20×16)
- **Learning:** Can detect obstacles 1 cell away in each direction

## Key Features

✅ **Side-by-side comparison** - Watch both drones learn simultaneously
✅ **Real-time metrics** - Success rate, states learned, average reward
✅ **Configurable scenarios** - 5 obstacle configurations to test
✅ **Adjustable speed** - Control training visualization speed
✅ **Performance tracking** - See which approach learns faster

## Obstacle Configurations

1. **Center Cluster** - Dense obstacles in the middle
2. **Right Corridor** - Obstacles blocking the right side
3. **Dual Barriers** - Two separate obstacle zones
4. **Horizontal Bands** - Scattered horizontal barriers
5. **Corner Sparse** - Light obstacles in corners

## Project Structure

```
QLearning_DroneNavigation/
├── comparison_gui.py       # Side-by-side comparison (recommended)
├── gui.py                  # Simple drone GUI
├── binary_gui.py           # Binary obstacle drone GUI
├── src/
│   ├── config.py           # Configuration settings
│   ├── drone.py            # Drone implementations
│   ├── grid.py             # Grid world environment
│   ├── trainer.py          # Training logic
│   ├── binary_trainer.py   # Binary drone trainer
│   └── simulation.py       # Simulation controller
├── test/                   # Test scripts and examples
└── requirements.txt        # Dependencies
```

## Learning Parameters

The system uses aggressive learning parameters optimized for fast convergence:

- **Learning rate:** 0.3 (stable learning)
- **Exploration rate:** 0.999 → 0.4 (extensive exploration)
- **Discount factor:** 0.995 (long-term reward focus)
- **Max episodes:** 50,000 (thorough training)
- **Max steps/episode:** 10,000 (allows complex paths)

## Running Tests

```bash
# Test different obstacle configurations
python test/test_obstacle_zones.py

# Test GUI configurations
python test/test_gui_configs.py

# Compare performance metrics
python test/performance_comparison.py

# Try custom zone training
python test/example_zone_training.py
```

## How It Works

### Q-Learning Algorithm

Both drones use Q-learning to learn optimal navigation:

1. **Explore** the environment (random actions)
2. **Exploit** learned knowledge (best known actions)
3. **Update** Q-values based on rewards
4. **Converge** to optimal policy over time

### State Representation Comparison

**Simple Drone:** Only knows "where am I?"
- Faster initial learning (smaller state space)
- Must hit obstacles to learn avoidance
- Good for sparse obstacle environments

**Binary Drone:** Knows "where am I + what's nearby?"
- Slower initial learning (larger state space)
- Can learn obstacle patterns without collision
- Better for dense obstacle environments

## Expected Results

In **sparse environments** (5-10 obstacles):
- Simple drone converges faster
- Similar final performance

In **dense environments** (20+ obstacles):
- Binary drone shows better collision avoidance
- May find safer paths
- Takes longer to converge initially

## Technical Details

- **Framework:** Tkinter (GUI), Python 3.8+
- **Algorithm:** Tabular Q-Learning with ε-greedy exploration
- **Grid:** 20×20 2D environment
- **Actions:** North, South, East, West
- **Rewards:** Goal (+2000), Obstacle (-20), Step (-0.005)

## License

Educational project - free to use and modify.
