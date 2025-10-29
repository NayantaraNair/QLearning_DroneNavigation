#!/usr/bin/env python3
"""Focused test on obstacle-dense environments where binary drone should excel."""

from src.grid import GridWorld
from src.drone import QLearningDrone, BinaryObstacleDrone
from src.config import GridSettings, LearningSettings


def test_obstacle_density_advantage():
    """Test binary vs simple drone with increasing obstacle density."""

    print("=" * 60)
    print("OBSTACLE DENSITY ADVANTAGE TEST")
    print("=" * 60)
    print("Testing where binary obstacle awareness provides clear benefits")

    # Test different obstacle densities
    densities = [
        (7, "Low Density (Original)"),
        (12, "Medium Density"),
        (18, "High Density"),
        (22, "Very High Density"),
    ]

    results = []

    for obstacle_count, density_name in densities:
        print(f"\n{'-' * 40}")
        print(f"TESTING: {density_name} - {obstacle_count} obstacles")
        print(f"{'-' * 40}")

        # Create challenging grid
        grid_settings = GridSettings(
            rows=20, cols=20, layers=1,
            start=(0, 0, 0), goal=(19, 19, 0),
            obstacle_count=obstacle_count,
            random_seed=42  # Consistent for fair comparison
        )

        learning_settings = LearningSettings(
            learning_rate=0.8,
            discount_factor=0.95,
            exploration_rate=0.9,
            exploration_decay=0.995,
            min_exploration=0.05,
            reward_step=-0.1,
            reward_goal=1000.0,
            reward_obstacle=-100.0,  # Higher penalty for hitting obstacles
            action_failure_rate=0.0
        )

        # Test both drones
        simple_metrics = train_drone(QLearningDrone, grid_settings, learning_settings, episodes=1000)
        binary_metrics = train_drone(BinaryObstacleDrone, grid_settings, learning_settings, episodes=1500)

        # Calculate key differences
        success_diff = binary_metrics['success_rate'] - simple_metrics['success_rate']
        collision_diff = simple_metrics['collision_rate'] - binary_metrics['collision_rate']  # Lower is better
        safety_score = binary_metrics['success_rate'] - binary_metrics['collision_rate']
        simple_safety = simple_metrics['success_rate'] - simple_metrics['collision_rate']
        safety_improvement = safety_score - simple_safety

        print(f"\nRESULTS for {density_name}:")
        print(f"Simple Drone  - Success: {simple_metrics['success_rate']:.1%}, Collisions: {simple_metrics['collision_rate']:.1%}")
        print(f"Binary Drone  - Success: {binary_metrics['success_rate']:.1%}, Collisions: {binary_metrics['collision_rate']:.1%}")
        print(f"Improvements  - Success: {success_diff:+.1%}, Collision Reduction: {collision_diff:+.1%}")
        print(f"Safety Score  - Simple: {simple_safety:.1%}, Binary: {safety_score:.1%} (Improvement: {safety_improvement:+.1%})")

        if collision_diff > 0.05:
            print(">>> ADVANTAGE: Binary drone shows significant collision reduction!")
        elif collision_diff > 0.02:
            print(">>> ADVANTAGE: Binary drone shows collision reduction")
        elif success_diff > 0.02:
            print(">>> ADVANTAGE: Binary drone shows success improvement")

        results.append({
            'obstacles': obstacle_count,
            'density_name': density_name,
            'success_diff': success_diff,
            'collision_diff': collision_diff,
            'safety_improvement': safety_improvement,
            'binary_collision_rate': binary_metrics['collision_rate'],
            'simple_collision_rate': simple_metrics['collision_rate']
        })

    # Summary analysis
    print(f"\n{'=' * 60}")
    print("OBSTACLE DENSITY ANALYSIS SUMMARY")
    print(f"{'=' * 60}")

    print(f"{'Density':<20} {'Success Diff':<12} {'Collision Reduction':<18} {'Safety Improvement':<18}")
    print("-" * 70)

    clear_advantages = 0
    for result in results:
        print(f"{result['density_name']:<20} {result['success_diff']:>+12.1%} {result['collision_diff']:>+18.1%} {result['safety_improvement']:>+18.1%}")

        if result['collision_diff'] > 0.05 or result['success_diff'] > 0.05:
            clear_advantages += 1

    print(f"\nKEY FINDINGS:")
    print(f"- Binary drone shows clear advantages in {clear_advantages} out of {len(results)} density levels")

    # Find best advantage
    best_collision_reduction = max(results, key=lambda x: x['collision_diff'])
    best_success_improvement = max(results, key=lambda x: x['success_diff'])

    print(f"- Best collision reduction: {best_collision_reduction['collision_diff']:.1%} in {best_collision_reduction['density_name']}")
    print(f"- Best success improvement: {best_success_improvement['success_diff']:.1%} in {best_success_improvement['density_name']}")

    # Conclusion
    if clear_advantages >= 2:
        print(f"\n>>> CONCLUSION: Binary obstacle drone shows clear advantages in dense environments!")
        print("The enhanced state representation is justified for obstacle-rich scenarios.")
    else:
        print(f"\n>>> CONCLUSION: Advantages are modest. Consider alternative approaches:")
        print("- Reward shaping for obstacle avoidance")
        print("- Dynamic environments")
        print("- Multi-objective optimization")

    return results


def train_drone(drone_class, grid_settings, learning_settings, episodes):
    """Train a drone and return key metrics."""

    world = GridWorld(settings=grid_settings, rewards=learning_settings)
    drone = drone_class(world, settings=learning_settings)

    print(f"Training {drone_class.__name__} for {episodes} episodes...")

    success_count = 0
    collision_count = 0
    episode_rewards = []

    for episode in range(episodes):
        drone.start_new_episode()
        episode_reward = 0.0
        episode_length = 0
        hit_obstacle = False

        for step in range(500):  # Max steps
            diagnostics = drone.step()
            episode_reward += diagnostics['reward']
            episode_length += 1

            # Track obstacle hits
            if diagnostics['reward'] < -50:
                hit_obstacle = True

            if diagnostics['done']:
                if diagnostics['reward'] > 500:  # Goal reached
                    success_count += 1
                break

        if hit_obstacle:
            collision_count += 1

        episode_rewards.append(episode_reward)

        if (episode + 1) % 200 == 0:
            recent_success = sum(1 for r in episode_rewards[-50:] if r > 500) / min(50, len(episode_rewards))
            print(f"  Episode {episode + 1}: Recent success rate: {recent_success:.1%}")

    return {
        'success_rate': success_count / episodes,
        'collision_rate': collision_count / episodes,
        'avg_reward': sum(episode_rewards) / len(episode_rewards),
        'q_table_size': len(drone.q_table)
    }


if __name__ == "__main__":
    test_obstacle_density_advantage()