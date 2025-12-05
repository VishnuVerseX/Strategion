"""
Training Script for Tennis Strategy RL Agent
UPDATED: Curriculum learning + Dueling Double DQN
"""

import numpy as np
import matplotlib.pyplot as plt
from tennis_env import TennisEnvironment
from dqn_model import DQNAgent  # This will be your Dueling Double DQN
import time
from pathlib import Path
import json


def get_opponent_skill(episode: int) -> float:
    """Curriculum learning: gradually increase opponent difficulty"""
    if episode < 400:
        return 0.40  # Solid start - agent wins ~58-62%
    elif episode < 800:
        return 0.44  # Moderate challenge - agent wins ~53-56%
    elif episode < 1200:
        return 0.47  # Getting tough - agent wins ~50-52%
    else:
        return 0.50  # Balanced - agent wins ~48-50%


def train_agent(
    num_episodes: int = 1500,
    max_steps_per_episode: int = 750,  # INCREASED from 500
    save_interval: int = 100,
    model_dir: str = 'models',
    log_dir: str = 'logs',
    use_curriculum: bool = True
):
    """Train the Dueling Double DQN agent with curriculum learning"""
    
    # Create directories
    Path(model_dir).mkdir(exist_ok=True)
    Path(log_dir).mkdir(exist_ok=True)
    
    # Initialize environment with easy opponent first
    initial_skill = 0.40 if use_curriculum else 0.45
    env = TennisEnvironment(
        opponent_skill=initial_skill,
        best_of=3  # Use best_of=1 for faster testing, 3 for realistic training
    )
    
    state_size = len(env.get_state_vector())
    action_size = len(env.actions)
    
    # Initialize Dueling Double DQN agent with optimized hyperparameters
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0005,  # Lower for Dueling DQN stability
        gamma=0.99,  # Higher for long matches
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9975,  # Slower decay for curriculum
        memory_size=20000,  # Larger memory
        batch_size=128,  # Larger batches
        target_update=5  # More frequent for Double DQN
    )
    
    print("="*70)
    print("Tennis Strategy Optimizer - ADVANCED TRAINING")
    print("="*70)
    print(f"Architecture: Dueling Double DQN")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Actions: {env.actions}")
    print(f"Curriculum Learning: {'ENABLED' if use_curriculum else 'DISABLED'}")
    print(f"Initial Opponent Skill: {initial_skill:.2f}")
    print("="*70)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    losses = []
    episode_results = []
    opponent_skills = []
    epsilons = [] #NOTE - to correct the epsilon value calc
    
    # Training loop
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Update opponent skill (curriculum learning)
        if use_curriculum:
            new_skill = get_opponent_skill(episode)
            if new_skill != env.opponent_skill:
                env.opponent_skill = new_skill
                print(f"\n🎓 Curriculum Update at Episode {episode}: "
                      f"Opponent Skill → {new_skill:.2f}\n")
        
        opponent_skills.append(env.opponent_skill)
        
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            # Select action from valid actions only
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Handle incomplete matches (timeout)
        if not done:
            # Match didn't finish - apply small penalty
            episode_reward -= 1.0
            # Still determine winner based on current score
            print(f"  [Episode {episode+1} reached max steps - match incomplete]")
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        epsilons.append(agent.epsilon) #NOTE - NEW: log epsilon calc
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine win/loss
        p_sets = env.state['player_sets']
        o_sets = env.state['opponent_sets']
        p_games = env.state['player_games']
        o_games = env.state['opponent_games']
        p_points = env.state['player_points']
        o_points = env.state['opponent_points']
        
        episode_win = 1 if (p_sets, p_games, p_points) > (o_sets, o_games, o_points) else 0
        episode_results.append(episode_win)
        
        # Calculate win rate (last 100 episodes)
        if len(episode_results) >= 100:
            win_rate = sum(episode_results[-100:]) / 100
        else:
            win_rate = sum(episode_results) / len(episode_results) if episode_results else 0.0
        win_rates.append(win_rate)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else 0
            
            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"Opp: {env.opponent_skill:.2f} | "
                  f"WR: {win_rate:5.1%} | "
                  f"Rew: {avg_reward:6.2f} | "
                  f"Len: {avg_length:5.0f} | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = f"{model_dir}/tennis_dueling_dqn_episode_{episode+1}.pth"
            agent.save_model(model_path)
            
            # Save training metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'win_rates': win_rates,
                'losses': losses,
                'opponent_skills': opponent_skills,
                'episode_results': episode_results
            }
            
            metrics_path = f"{log_dir}/training_metrics_episode_{episode+1}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            print(f"  ✓ Checkpoint saved at episode {episode+1}")
    
    # Final save
    final_model_path = f"{model_dir}/tennis_dueling_dqn_final.pth"
    agent.save_model(final_model_path)
    
    training_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Final Win Rate: {win_rates[-1]:.2%}")
    print(f"Final Opponent Skill: {env.opponent_skill:.2f}")
    print(f"Final model saved to {final_model_path}")
    print("="*70)
    
    # Plot training results
    plot_training_results(
        episode_rewards, episode_lengths, win_rates, 
        losses, opponent_skills, epsilons, log_dir #NOTE - Added epsilons to make chnages to calc
    )
    
    return agent


def plot_training_results(episode_rewards, episode_lengths, win_rates, 
                         losses, opponent_skills, log_dir, epsilons): #NOTE - Added epsilons to make chnages to calc
    """Plot and save training metrics with curriculum visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, label='Episode Reward', color='blue')
    if len(episode_rewards) > 50:
        window = 50
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       label=f'{window}-Ep MA', linewidth=2, color='darkblue')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Win Rate with Curriculum
    ax2 = axes[0, 1]
    ax2.plot(win_rates, linewidth=2, color='green', label='Win Rate')
    ax2.axhline(y=0.5, color='gold', linestyle='--', label='50% baseline', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate', color='green')
    ax2.set_title('Win Rate & Opponent Difficulty')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Overlay opponent skill
    ax2_twin = ax2.twinx()
    ax2_twin.plot(opponent_skills, linewidth=1.5, color='red', 
                  linestyle='--', alpha=0.7, label='Opponent Skill')
    ax2_twin.set_ylabel('Opponent Skill', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim([0.3, 0.6])
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 3: Episode Lengths
    axes[0, 2].plot(episode_lengths, alpha=0.4, label='Episode Length', color='purple')
    if len(episode_lengths) > 50:
        window = 50
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(range(window-1, len(episode_lengths)), moving_avg, 
                       label=f'{window}-Ep MA', linewidth=2, color='darkviolet')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Episode Lengths Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Training Loss
    if losses:
        axes[1, 0].plot(losses, alpha=0.4, label='Loss', color='orange')
        if len(losses) > 50:
            window = 50
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(losses)), moving_avg, 
                           label=f'{window}-Ep MA', linewidth=2, color='darkorange')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot 5: Performance Distribution
    axes[1, 1].hist(episode_rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 1].axvline(np.mean(episode_rewards), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Learning Progress Summary
    axes[1, 2].axis('off')
    
    # Calculate phase-wise statistics
    total_eps = len(episode_rewards)
    phase1_end = min(400, total_eps)
    phase2_end = min(800, total_eps)
    phase3_end = min(1200, total_eps)
    
    phase1_wr = np.mean(win_rates[:phase1_end]) if phase1_end > 0 else 0
    phase2_wr = np.mean(win_rates[phase1_end:phase2_end]) if phase2_end > phase1_end else 0
    phase3_wr = np.mean(win_rates[phase2_end:phase3_end]) if phase3_end > phase2_end else 0
    final_wr = np.mean(win_rates[phase3_end:]) if total_eps > phase3_end else np.mean(win_rates[-100:])
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}
    
    Total Episodes: {total_eps}
    Final Win Rate: {win_rates[-1]:.1%}
    
    CURRICULUM PHASES:
    Phase 1 (0-400):    {phase1_wr:.1%} avg WR
    Phase 2 (400-800):  {phase2_wr:.1%} avg WR
    Phase 3 (800-1200): {phase3_wr:.1%} avg WR
    Final (1200+):      {final_wr:.1%} avg WR
    
    PERFORMANCE:
    Avg Reward:  {np.mean(episode_rewards):.2f}
    Avg Length:  {np.mean(episode_lengths):.0f} steps
    Best Reward: {np.max(episode_rewards):.2f}
    Worst Reward: {np.min(episode_rewards):.2f}
    
    Final ε: {epsilons[-1] if epsilons else 0:.3f} #NOTE - to make chnages to calc
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                    family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{log_dir}/training_results_dueling_dqn.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to {log_dir}/training_results_dueling_dqn.png")
    plt.close()


if __name__ == "__main__":
    # Training configuration
    NUM_EPISODES = 1500
    MAX_STEPS = 750
    SAVE_INTERVAL = 100
    USE_CURRICULUM = True
    
    print("\n🎾 Starting Advanced Tennis Strategy Optimizer Training 🎾")
    print("🏆 Using Dueling Double DQN + Curriculum Learning\n")
    
    agent = train_agent(
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        save_interval=SAVE_INTERVAL,
        use_curriculum=USE_CURRICULUM
    )
    
    print("\n✓ Training Complete! Use evaluate.py to test the agent.\n")