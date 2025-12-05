"""
Evaluation Script for Tennis Strategy RL Agent
UPDATED: Works with fixed environment + Dueling DQN + Better analysis
"""

import numpy as np
from tennis_env import TennisEnvironment
from dqn_model import DQNAgent
import matplotlib.pyplot as plt
from pathlib import Path
import json


def evaluate_agent(
    model_path: str,
    num_episodes: int = 100,
    opponent_skill: float = 0.5,
    visualize: bool = True,
    verbose: bool = True
):
    """
    Evaluate trained agent
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        opponent_skill: Opponent difficulty (0.35=easy, 0.5=balanced, 0.6=hard)
        visualize: Whether to create plots
        verbose: Whether to print detailed progress
    """
    
    # Initialize environment with specified opponent
    env = TennisEnvironment(opponent_skill=opponent_skill)
    state_size = len(env.get_state_vector())
    action_size = len(env.actions)
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon_start=0.0  # No exploration during evaluation
    )
    
    # Load trained model
    try:
        agent.load_model(model_path)
        print(f"✓ Model loaded from {model_path}\n")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train.py")
        return None
    
    print("="*70)
    print("Evaluating Agent Performance")
    print("="*70)
    print(f"Opponent Skill Level: {opponent_skill:.2f}")
    print(f"Number of Episodes: {num_episodes}")
    print("="*70 + "\n")
    
    # Evaluation metrics
    wins = 0
    total_rewards = []
    episode_lengths = []
    action_counts = {action: 0 for action in range(action_size)}
    
    # NEW: Phase-specific metrics
    serve_action_counts = {i: 0 for i in [0, 1, 2]}
    return_action_counts = {i: 0 for i in [3, 4, 5]}
    rally_action_counts = {i: 0 for i in [6, 7, 8, 9]}
    
    # NEW: Game statistics
    points_won_serving = 0
    points_total_serving = 0
    points_won_returning = 0
    points_total_returning = 0
    games_won_serving = 0
    games_total_serving = 0
    games_won_returning = 0
    games_total_returning = 0
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Track within episode
        was_serving = env.state['is_player_serving']
        
        while True:
            # Agent selects action (no exploration)
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            
            # Track action usage by phase
            action_counts[action] += 1
            if action in [0, 1, 2]:
                serve_action_counts[action] += 1
            elif action in [3, 4, 5]:
                return_action_counts[action] += 1
            elif action in [6, 7, 8, 9]:
                rally_action_counts[action] += 1
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Track point outcomes
            if info['point_over']:
                if was_serving:
                    points_total_serving += 1
                    if info['point_won']:
                        points_won_serving += 1
                else:
                    points_total_returning += 1
                    if info['point_won']:
                        points_won_returning += 1
                
                # Update serving status for next point
                was_serving = env.state['is_player_serving']
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                # Determine win/loss
                p_sets = env.state['player_sets']
                o_sets = env.state['opponent_sets']
                p_games = env.state['player_games']
                o_games = env.state['opponent_games']
                p_points = env.state['player_points']
                o_points = env.state['opponent_points']

                is_win = (p_sets, p_games, p_points) > (o_sets, o_games, o_points)
                if is_win:
                    wins += 1
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose and (episode + 1) % 10 == 0:
            current_wr = wins / (episode + 1)
            avg_reward_so_far = np.mean(total_rewards)
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"WR: {current_wr:5.1%} | "
                  f"Avg Reward: {avg_reward_so_far:6.2f} | "
                  f"Last Reward: {episode_reward:6.2f}")
    
    # Calculate statistics
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    # Calculate serve/return percentages
    serve_win_pct = (points_won_serving / points_total_serving * 100) if points_total_serving > 0 else 0
    return_win_pct = (points_won_returning / points_total_returning * 100) if points_total_returning > 0 else 0
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Opponent Skill: {opponent_skill:.2f}")
    print(f"Total Episodes: {num_episodes}")
    print(f"\n📊 MATCH STATISTICS:")
    print(f"  Matches Won:  {wins}")
    print(f"  Matches Lost: {num_episodes - wins}")
    print(f"  Win Rate:     {win_rate:.2%}")
    print(f"\n💰 REWARD METRICS:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best Reward:    {np.max(total_rewards):.2f}")
    print(f"  Worst Reward:   {np.min(total_rewards):.2f}")
    print(f"\n⏱️  EPISODE LENGTH:")
    print(f"  Average: {avg_length:.1f} steps")
    print(f"  Min:     {np.min(episode_lengths):.0f} steps")
    print(f"  Max:     {np.max(episode_lengths):.0f} steps")
    print(f"\n🎾 SERVE/RETURN PERFORMANCE:")
    print(f"  Points Won When Serving:   {points_won_serving}/{points_total_serving} ({serve_win_pct:.1f}%)")
    print(f"  Points Won When Returning: {points_won_returning}/{points_total_returning} ({return_win_pct:.1f}%)")
    print("="*70)
    
    print("\n📋 ACTION DISTRIBUTION:")
    print("-" * 70)
    
    # Serve actions
    total_serves = sum(serve_action_counts.values())
    if total_serves > 0:
        print("SERVE ACTIONS:")
        for action_idx in [0, 1, 2]:
            count = serve_action_counts[action_idx]
            percentage = (count / total_serves) * 100
            print(f"  {env.actions[action_idx]:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Return actions
    total_returns = sum(return_action_counts.values())
    if total_returns > 0:
        print("\nRETURN ACTIONS:")
        for action_idx in [3, 4, 5]:
            count = return_action_counts[action_idx]
            percentage = (count / total_returns) * 100
            print(f"  {env.actions[action_idx]:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Rally actions
    total_rallies = sum(rally_action_counts.values())
    if total_rallies > 0:
        print("\nRALLY ACTIONS:")
        for action_idx in [6, 7, 8, 9]:
            count = rally_action_counts[action_idx]
            percentage = (count / total_rallies) * 100
            print(f"  {env.actions[action_idx]:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print("-" * 70)
    
    # Visualize results
    if visualize:
        visualize_evaluation(
            total_rewards, episode_lengths, action_counts, 
            env.actions, win_rate, opponent_skill,
            serve_win_pct, return_win_pct
        )
    # In evaluate.py, after evaluation
    total_returns = sum(return_action_counts.values())
    total_rallies = sum(rally_action_counts.values())

    block_pct = return_action_counts[5] / total_returns * 100
    lob_pct = rally_action_counts[9] / total_rallies * 100

    print(f"\n🔍 DEFENSIVE PLAY ANALYSIS:")
    print(f"return_block usage:  {block_pct:.1f}% (healthy: <30%)")
    print(f"defensive_lob usage: {lob_pct:.1f}% (healthy: <15%)")

    if block_pct > 35:
        print("⚠️  WARNING: Agent is too defensive on returns!")
    if lob_pct > 20:
        print("⚠️  WARNING: Agent is too defensive in rallies!")
    
    
    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'action_counts': action_counts,
        'serve_win_pct': serve_win_pct,
        'return_win_pct': return_win_pct,
        'opponent_skill': opponent_skill
    }


def visualize_evaluation(rewards, lengths, action_counts, action_names, 
                        win_rate, opponent_skill, serve_win_pct, return_win_pct):
    """Create enhanced visualization of evaluation results"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Reward Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rewards):.2f}')
    ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax2.axvline(np.mean(lengths), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(lengths):.1f}')
    ax2.set_xlabel('Episode Length (steps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*35}
    
    Win Rate:     {win_rate:.1%}
    Opponent:     {opponent_skill:.2f}
    
    Avg Reward:   {np.mean(rewards):.2f}
    Std Reward:   {np.std(rewards):.2f}
    
    Avg Length:   {np.mean(lengths):.0f} steps
    
    Serve Win%:   {serve_win_pct:.1f}%
    Return Win%:  {return_win_pct:.1f}%
    
    Total Episodes: {len(rewards)}
    
    {'='*35}
    """
    ax3.text(0.05, 0.5, summary_text, fontsize=11, 
            family='monospace', verticalalignment='center')
    
    # Plot 4: Serve Actions Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    serve_actions = [0, 1, 2]
    serve_counts = [action_counts[i] for i in serve_actions]
    serve_names = [action_names[i].replace('serve_', '') for i in serve_actions]
    colors_serve = ['#FF6B6B', '#FF8E8E', '#FFB3B3']
    ax4.bar(serve_names, serve_counts, color=colors_serve, edgecolor='black', alpha=0.8)
    ax4.set_title('Serve Action Usage')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Return Actions Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    return_actions = [3, 4, 5]
    return_counts = [action_counts[i] for i in return_actions]
    return_names = [action_names[i].replace('return_', '') for i in return_actions]
    colors_return = ['#4ECDC4', '#45B7AF', '#3AA39A']
    ax5.bar(return_names, return_counts, color=colors_return, edgecolor='black', alpha=0.8)
    ax5.set_title('Return Action Usage')
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Rally Actions Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    rally_actions = [6, 7, 8, 9]
    rally_counts = [action_counts[i] for i in rally_actions]
    rally_names = [action_names[i].replace('rally_', '').replace('_', ' ') for i in rally_actions]
    colors_rally = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
    ax6.bar(rally_names, rally_counts, color=colors_rally, edgecolor='black', alpha=0.8)
    ax6.set_title('Rally Action Usage')
    ax6.set_ylabel('Count')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Reward Timeline
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.plot(rewards, alpha=0.6, linewidth=1, color='blue', label='Episode Reward')
    ax7.axhline(np.mean(rewards), color='r', linestyle='--', linewidth=2, label='Mean')
    ax7.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    # Add moving average
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax7.plot(range(window-1, len(rewards)), ma, color='darkblue', 
                linewidth=2, label=f'{window}-Episode MA')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Reward')
    ax7.set_title('Reward Over Evaluation Episodes')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Win Rate Indicator
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create a gauge-like visualization
    fig_temp = plt.figure(figsize=(4, 4))
    ax_temp = fig_temp.add_subplot(111, projection='polar')
    
    # Gauge parameters
    theta = np.linspace(0, np.pi, 100)
    radii = np.ones_like(theta)
    
    # Color based on win rate
    if win_rate >= 0.55:
        color = 'green'
        rating = 'EXCELLENT'
    elif win_rate >= 0.48:
        color = 'gold'
        rating = 'GOOD'
    elif win_rate >= 0.40:
        color = 'orange'
        rating = 'FAIR'
    else:
        color = 'red'
        rating = 'NEEDS WORK'
    
    # Draw gauge
    ax_temp.fill_between(theta, 0, radii, color='lightgray', alpha=0.3)
    win_theta = win_rate * np.pi
    ax_temp.fill_between(theta[theta <= win_theta], 0, 
                        radii[theta <= win_theta], color=color, alpha=0.7)
    
    # Add indicator
    ax_temp.plot([win_theta, win_theta], [0, 1], 'k-', linewidth=3)
    ax_temp.scatter([win_theta], [1], s=200, c='black', zorder=5)
    
    ax_temp.set_ylim(0, 1)
    ax_temp.set_theta_zero_location('W')
    ax_temp.set_theta_direction(1)
    ax_temp.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax_temp.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_temp.set_yticks([])
    ax_temp.set_title(f'Win Rate: {win_rate:.1%}\n{rating}', 
                     fontsize=14, fontweight='bold', pad=20)
    
    plt.close(fig_temp)
    
    # Instead, add text-based win rate display
    win_rate_text = f"""
    
    WIN RATE
    {'='*20}
    
    {win_rate:.1%}
    
    Rating: {rating}
    
    {'='*20}
    
    vs Opponent {opponent_skill:.2f}
    """
    ax8.text(0.5, 0.5, win_rate_text, fontsize=14, fontweight='bold',
            family='monospace', verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.suptitle('Tennis Agent Evaluation Report', fontsize=16, fontweight='bold', y=0.98)
    
    Path('logs').mkdir(exist_ok=True)
    plt.savefig('logs/evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to logs/evaluation_results.png")
    plt.close()


def demo_single_match(model_path: str, opponent_skill: float = 0.5, max_display_steps: int = 50):
    """Run a single match with detailed output"""
    
    env = TennisEnvironment(opponent_skill=opponent_skill)
    state_size = len(env.get_state_vector())
    action_size = len(env.actions)
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon_start=0.0)
    agent.load_model(model_path)
    
    print("\n" + "="*70)
    print("🎾 DEMO MATCH - Detailed Play-by-Play 🎾")
    print("="*70)
    print(f"Opponent Skill Level: {opponent_skill:.2f}")
    print("="*70 + "\n")
    
    state = env.reset()
    step = 0
    total_reward = 0
    display_steps = 0
    
    while True:
        step += 1
        
        # Get Q-values for analysis
        q_values = agent.get_q_values(state)
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=False)
        
        # Only display first N steps to avoid clutter
        if display_steps < max_display_steps:
            print(f"Step {step}:")
            print(f"  Game State: {env.get_state_description()}")
            print(f"  Rally Length: {env.state['rally_length']}")
            print(f"  Valid Actions: {[env.actions[i] for i in valid_actions]}")
            print(f"  Action Chosen: {env.actions[action]}")
            print(f"  Top 3 Q-values:")
            
            # Sort and show top 3 valid actions
            valid_q = [(i, q_values[i]) for i in valid_actions]
            valid_q.sort(key=lambda x: x[1], reverse=True)
            for i, (act_idx, q_val) in enumerate(valid_q[:3], 1):
                marker = " ← CHOSEN" if act_idx == action else ""
                print(f"    {i}. {env.actions[act_idx]:20s}: {q_val:7.3f}{marker}")
        elif display_steps == max_display_steps:
            print(f"\n... [Hiding detailed steps, showing summary only] ...\n")
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if display_steps < max_display_steps and info['point_over']:
            result = "WON ✓" if info['point_won'] else "LOST ✗"
            print(f"  → Point {result} | Reward: {reward:+.2f}")
            print()
        
        display_steps += 1
        state = next_state
        
        if done:
            print("\n" + "="*70)
            print("MATCH OVER!")
            print("="*70)
            player_sets = env.state['player_sets']
            opponent_sets = env.state['opponent_sets']
            player_games = env.state['player_games']
            opponent_games = env.state['opponent_games']
            
            is_win = player_sets > opponent_sets
            
            print(f"Final Score:")
            print(f"  Sets:  Player {player_sets} - {opponent_sets} Opponent")
            print(f"  Games: Player {player_games} - {opponent_games} Opponent")
            print(f"\nTotal Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"\nResult: {'VICTORY! 🏆' if is_win else 'Defeat 😞'}")
            print("="*70)
            break


def compare_opponent_skills(model_path: str, num_episodes: int = 50):
    """Evaluate agent against multiple opponent skill levels"""
    
    print("\n" + "="*70)
    print("🎾 MULTI-LEVEL OPPONENT EVALUATION 🎾")
    print("="*70 + "\n")
    
    skill_levels = [0.35, 0.40, 0.45, 0.50, 0.55]
    results = {}
    
    for skill in skill_levels:
        print(f"\n{'─'*70}")
        print(f"Testing against opponent skill: {skill:.2f}")
        print(f"{'─'*70}")
        
        result = evaluate_agent(
            model_path=model_path,
            num_episodes=num_episodes,
            opponent_skill=skill,
            visualize=False,
            verbose=False
        )
        
        if result:
            results[skill] = result
            print(f"✓ Win Rate: {result['win_rate']:.1%} | Avg Reward: {result['avg_reward']:.2f}")
    
    # Visualize comparison
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        skills = list(results.keys())
        win_rates = [results[s]['win_rate'] * 100 for s in skills]
        avg_rewards = [results[s]['avg_reward'] for s in skills]
        
        # Win rates
        ax1.plot(skills, win_rates, 'o-', linewidth=2, markersize=10, color='green')
        ax1.axhline(50, color='gray', linestyle='--', label='50% baseline')
        ax1.set_xlabel('Opponent Skill Level')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate vs Opponent Difficulty')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Average rewards
        ax2.plot(skills, avg_rewards, 'o-', linewidth=2, markersize=10, color='blue')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel('Opponent Skill Level')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Avg Reward vs Opponent Difficulty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/opponent_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to logs/opponent_comparison.png")
        plt.close()
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default model path (updated for Dueling DQN)
    model_path = "models/tennis_dueling_dqn_final.pth"
    
    # Check if alternative model specified
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("\nAvailable models:")
        model_dir = Path("models")
        if model_dir.exists():
            models = list(model_dir.glob("*.pth"))
            if models:
                for model in models:
                    print(f"  - {model}")
            else:
                print("  No models found. Please train first using train.py")
        else:
            print("  No models directory found. Please train first using train.py")
        sys.exit(1)
    
    print("\n🎾 Tennis Strategy Optimizer - Advanced Evaluation 🎾\n")
    
    # Run standard evaluation
    print("Option 1: Standard Evaluation (vs balanced opponent)")
    print("Option 2: Demo Match (detailed play-by-play)")
    print("Option 3: Multi-Level Comparison (vs different opponents)")
    
    choice = input("\nSelect option (1/2/3) or press Enter for option 1: ").strip()
    
    if choice == "2":
        skill = float(input("Opponent skill (0.35-0.55, default 0.5): ") or "0.5")
        demo_single_match(model_path, opponent_skill=skill)
    elif choice == "3":
        episodes = int(input("Episodes per skill level (default 50): ") or "50")
        compare_opponent_skills(model_path, num_episodes=episodes)
    else:
        skill = float(input("Opponent skill (0.35-0.55, default 0.5): ") or "0.5")
        results = evaluate_agent(model_path, num_episodes=100, 
                               opponent_skill=skill, visualize=True)
        
        if results:
            user_input = input("\nRun a detailed demo match? (y/n): ")
            if user_input.lower() == 'y':
                demo_single_match(model_path, opponent_skill=skill)
    