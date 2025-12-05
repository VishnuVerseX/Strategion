"""
Enhanced Tennis Strategy Predictor
Combines tactical reasoning with comprehensive predictions
Provides both serve/return and rally recommendations
"""

import numpy as np
from tennis_env import TennisEnvironment
from dqn_model import DQNAgent
from pathlib import Path
import math


class EnhancedTennisPredictor:
    def __init__(
        self, 
        model_path: str = "models/tennis_dueling_dqn_final.pth",
        opponent_skill: float = 0.5
    ):
        """Initialize predictor with full features"""
        
        self.env = TennisEnvironment(opponent_skill=opponent_skill)
        state_size = len(self.env.get_state_vector())
        action_size = len(self.env.actions)
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon_start=0.0
        )
        
        if Path(model_path).exists():
            self.agent.load_model(model_path)
            print(f"✓ Model loaded successfully")
            print(f"✓ Opponent skill level: {opponent_skill:.2f}\n")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def _q_to_confidence(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values to confidence percentages using softmax"""
        q_shifted = q_values - np.max(q_values)
        temperature = 2.0
        exp_q = np.exp(q_shifted / temperature)
        softmax = exp_q / np.sum(exp_q)
        return softmax * 100
    
    def _explain_action_choice(
        self,
        action: str,
        action_idx: int,
        state: dict,
        q_values: np.ndarray,
        valid_actions: list
    ) -> str:
        """Generate human-readable tactical explanation"""
        reasons = []
        
        p_points = state['player_points']
        o_points = state['opponent_points']
        is_serving = state['is_player_serving']
        rally_length = state['rally_length']
        player_fatigue = state['player_fatigue']
        opponent_fatigue = state['opponent_fatigue']
        
        is_big_point = (p_points >= 3 and o_points >= 3) or \
                       (p_points >= 3) or (o_points >= 3)
        
        # SERVING EXPLANATIONS
        if is_serving and rally_length == 0:
            if 'aggressive' in action or 'flat' in action:
                if is_big_point:
                    reasons.append("Big point - going for first serve")
                if opponent_fatigue > 0.4:
                    reasons.append("Opponent tired - attacking serve")
                if player_fatigue < 0.3:
                    reasons.append("Fresh legs - can afford risk")
                    
            elif 'kick' in action or 'body' in action:
                if player_fatigue > 0.4:
                    reasons.append("Managing fatigue with safer serve")
                if is_big_point:
                    reasons.append("Critical point - playing percentage")
                reasons.append("High-percentage serve to start rally")
        
        # RETURNING EXPLANATIONS
        elif not is_serving and rally_length == 0:
            if 'aggressive' in action:
                if o_points <= 1:
                    reasons.append("Opponent's weak serve - attacking")
                if opponent_fatigue > player_fatigue:
                    reasons.append("Opponent more tired - pressing advantage")
                if is_big_point:
                    reasons.append("Break point opportunity - being aggressive")
                    
            elif 'block' in action or 'chip' in action:
                reasons.append("Powerful serve - defensive return")
                if player_fatigue > 0.5:
                    reasons.append("High fatigue - just getting ball back")
                    
            elif 'neutral' in action:
                reasons.append("Solid return to start rally")
        
        # RALLY EXPLANATIONS
        elif rally_length > 0:
            if 'aggressive' in action or 'winner' in action:
                if rally_length > 5:
                    reasons.append(f"Long rally ({rally_length} shots) - going for winner")
                if opponent_fatigue > player_fatigue + 0.1:
                    reasons.append("Opponent showing fatigue - pressing")
                    
            elif 'approach' in action or 'net' in action:
                if rally_length > 3:
                    reasons.append(f"Rally established - approaching net")
                reasons.append("Taking control of point at net")
                
            elif 'neutral' in action or 'baseline' in action:
                if rally_length <= 3:
                    reasons.append("Rally just started - building point")
                if is_big_point:
                    reasons.append("Important point - staying consistent")
                reasons.append("Maintaining rally from baseline")
                
            elif 'lob' in action or 'defensive' in action:
                if player_fatigue > 0.6:
                    reasons.append("High fatigue - buying recovery time")
                if rally_length > 8:
                    reasons.append(f"Very long rally - defensive reset")
                reasons.append("Under pressure - defensive shot")
        
        if not reasons:
            reasons.append("Highest expected value in current position")
        
        # Add confidence-based reasoning
        confidences = self._q_to_confidence(q_values)
        top_confidence = confidences[action_idx]
        
        if top_confidence > 40:
            reasons.append(f"Strong conviction ({top_confidence:.0f}% confidence)")
        elif top_confidence < 25:
            reasons.append(f"Close decision among multiple options")
        
        return " | ".join(reasons)
    
    def _analyze_situation(self, state: dict) -> dict:
        """Analyze current game situation"""
        p_pts, o_pts = state['player_points'], state['opponent_points']
        p_games, o_games = state['player_games'], state['opponent_games']
        
        situation = {
            'pressure_level': 'normal',
            'momentum': 'neutral',
            'game_phase': 'mid-game',
            'tactical_priority': 'balanced'
        }
        
        # Pressure analysis
        if p_pts >= 3 and o_pts >= 3:
            situation['pressure_level'] = 'very high (deuce)'
        elif p_pts >= 3 or o_pts >= 3:
            situation['pressure_level'] = 'high (game point)'
        elif p_pts == o_pts:
            situation['pressure_level'] = 'normal (even)'
        
        # Momentum
        if p_pts > o_pts + 1:
            situation['momentum'] = 'with you'
        elif o_pts > p_pts + 1:
            situation['momentum'] = 'against you'
        else:
            situation['momentum'] = 'neutral'
        
        # Game phase
        if p_games + o_games < 3:
            situation['game_phase'] = 'early set'
        elif p_games >= 5 or o_games >= 5:
            situation['game_phase'] = 'crucial games'
        else:
            situation['game_phase'] = 'mid set'
        
        # Tactical priority
        if state['is_player_serving']:
            if p_pts <= 1:
                situation['tactical_priority'] = 'control serve game'
            else:
                situation['tactical_priority'] = 'close out service game'
        else:
            if o_pts >= 3:
                situation['tactical_priority'] = 'break opportunity - aggressive'
            else:
                situation['tactical_priority'] = 'stay in return game'
        
        return situation
    
    def predict_from_state(
        self,
        player_points: int,
        opponent_points: int,
        player_games: int,
        opponent_games: int,
        player_sets: int,
        opponent_sets: int,
        is_player_serving: bool,
        court_side: str = 'deuce',
        player_fatigue: float = 0.0,
        opponent_fatigue: float = 0.0,
        rally_length: int = 0,
    ):
        """
        Predict optimal strategies with full explanations.
        Returns BOTH serve/return AND rally recommendations.
        """
        # Set base environment state
        self.env.state['player_points'] = player_points
        self.env.state['opponent_points'] = opponent_points
        self.env.state['player_games'] = player_games
        self.env.state['opponent_games'] = opponent_games
        self.env.state['player_sets'] = player_sets
        self.env.state['opponent_sets'] = opponent_sets
        self.env.state['is_player_serving'] = is_player_serving
        self.env.state['court_side'] = court_side
        self.env.state['player_fatigue'] = player_fatigue
        self.env.state['opponent_fatigue'] = opponent_fatigue
        self.env.state['rally_length'] = rally_length

        # Handle deuce/advantage situations
        self.env.state['is_deuce'] = False
        self.env.state['player_advantage'] = False
        self.env.state['opponent_advantage'] = False

        if player_points >= 3 and opponent_points >= 3:
            if player_points == opponent_points:
                self.env.state['is_deuce'] = True
            elif player_points > opponent_points:
                self.env.state['player_advantage'] = True
            else:
                self.env.state['opponent_advantage'] = True

        # Reset positions
        self.env.state['player_position'] = self.env.POS_BASELINE
        self.env.state['opponent_position'] = self.env.POS_BASELINE
        self.env.state['ball_depth'] = self.env.DEPTH_NEUTRAL

        # Save base state
        base_state_dict = dict(self.env.state)

        def _phase_recommendation(rally_length_value: int):
            """Get recommendation for specific rally phase"""
            self.env.state = dict(base_state_dict)
            self.env.state['rally_length'] = rally_length_value

            state_vec = self.env.get_state_vector()
            q_values = self.agent.get_q_values(state_vec)
            valid_actions = self.env.get_valid_actions()
            confidences = self._q_to_confidence(q_values)

            # Find best valid action
            best_idx = max(valid_actions, key=lambda a: q_values[a])
            best_action = self.env.actions[best_idx]

            # Get tactical explanation
            explanation = self._explain_action_choice(
                best_action, best_idx, self.env.state, 
                q_values, valid_actions
            )

            # Create rankings for all actions
            action_rankings = []
            for idx, q_val in enumerate(q_values):
                action_rankings.append({
                    'action': self.env.actions[idx],
                    'q_value': float(q_val),
                    'confidence': float(confidences[idx]),
                    'is_valid': idx in valid_actions,
                    'rank': 0,
                })
            
            # Sort by Q-value and assign ranks
            action_rankings.sort(key=lambda x: x['q_value'], reverse=True)
            for rank, info in enumerate(action_rankings, 1):
                info['rank'] = rank

            return {
                'recommended_action': best_action,
                'q_value': float(q_values[best_idx]),
                'confidence': float(confidences[best_idx]),
                'tactical_reasoning': explanation,
                'all_actions': action_rankings,
                'valid_actions': [self.env.actions[i] for i in valid_actions],
            }

        # Get recommendations for both phases
        serve_return_rec = _phase_recommendation(rally_length_value=0)
        rally_rec = _phase_recommendation(rally_length_value=1)

        # Restore base state for description
        self.env.state = base_state_dict
        game_state_desc = self.env.get_state_description()
        situation = self._analyze_situation(self.env.state)

        return {
            'game_state': game_state_desc,
            'situation_analysis': situation,
            'serve_or_return': serve_return_rec,
            'rally': rally_rec,
        }
    
    def predict_from_score_string(
        self, 
        score_string: str, 
        is_serving: bool = True,
        player_fatigue: float = 0.0,
        opponent_fatigue: float = 0.0
    ):
        """
        Predict from tennis score string
        
        Examples:
            "0-0, 0-0, 30-40" -> 0 sets, 0 games, 30-40 in current game
            "1-0, 3-2, 15-30" -> 1-0 in sets, 3-2 in games, 15-30 in game
        """
        try:
            parts = score_string.split(',')
            
            if len(parts) != 3:
                return {'error': 'Format must be: sets, games, points (e.g., "0-0, 2-1, 30-40")'}
            
            # Parse sets
            sets = parts[0].strip().split('-')
            player_sets = int(sets[0])
            opponent_sets = int(sets[1])
            
            # Parse games
            games = parts[1].strip().split('-')
            player_games = int(games[0])
            opponent_games = int(games[1])
            
            # Parse points
            points = parts[2].strip().split('-')
            score_to_points = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}
            player_points = score_to_points.get(points[0], 0)
            opponent_points = score_to_points.get(points[1], 0)
            
            return self.predict_from_state(
                player_points=player_points,
                opponent_points=opponent_points,
                player_games=player_games,
                opponent_games=opponent_games,
                player_sets=player_sets,
                opponent_sets=opponent_sets,
                is_player_serving=is_serving,
                player_fatigue=player_fatigue,
                opponent_fatigue=opponent_fatigue,
            )
            
        except (ValueError, IndexError) as e:
            return {'error': f"Invalid score format: {e}. Use format: '0-0, 2-1, 30-40'"}


def interactive_predictor():
    """Interactive command-line interface"""
    
    print("\n" + "="*70)
    print("🎾 Enhanced Tennis Strategy Predictor 🎾")
    print("="*70 + "\n")
    
    model_path = input("Model path (Enter for default): ").strip()
    if not model_path:
        model_path = "models/tennis_dueling_dqn_final.pth"
    
    opp_skill = input("Opponent skill (0.35-0.55, default 0.5): ").strip()
    try:
        opponent_skill = float(opp_skill) if opp_skill else 0.5
        opponent_skill = max(0.3, min(0.6, opponent_skill))
    except ValueError:
        opponent_skill = 0.5
    
    try:
        predictor = EnhancedTennisPredictor(model_path, opponent_skill)
    except FileNotFoundError:
        print("❌ No trained model found. Please run train.py first.")
        return
    
    print("\n" + "="*70)
    print("Commands: 'score format' | 'quick' | 'quit'")
    print("="*70 + "\n")
    
    while True:
        print("-" * 70)
        score_input = input("\nEnter score (sets, games, points): ").strip()
        
        if score_input.lower() == 'quit':
            print("\n👋 Goodbye!\n")
            break
        
        if score_input.lower() == 'quick':
            # Random scenario
            p_pts = np.random.randint(0, 4)
            o_pts = np.random.randint(0, 4)
            p_games = np.random.randint(0, 6)
            o_games = np.random.randint(0, 6)
            p_sets = np.random.randint(0, 2)
            o_sets = np.random.randint(0, 2)
            is_serving = bool(np.random.randint(0, 2))
            p_fatigue = np.random.uniform(0, 0.5)
            o_fatigue = np.random.uniform(0, 0.5)
            
            result = predictor.predict_from_state(
                p_pts, o_pts, p_games, o_games, p_sets, o_sets,
                is_serving, player_fatigue=p_fatigue, 
                opponent_fatigue=o_fatigue
            )
        else:
            serving = input("Are you serving? (y/n): ").strip().lower() == 'y'
            
            fatigue_q = input("Include fatigue? (y/n): ").strip().lower()
            p_fatigue, o_fatigue = 0.0, 0.0
            
            if fatigue_q == 'y':
                try:
                    p_fatigue = float(input("Your fatigue (0-1): ") or "0")
                    o_fatigue = float(input("Opponent fatigue (0-1): ") or "0")
                    p_fatigue = max(0.0, min(1.0, p_fatigue))
                    o_fatigue = max(0.0, min(1.0, o_fatigue))
                except ValueError:
                    p_fatigue, o_fatigue = 0.0, 0.0
            
            result = predictor.predict_from_score_string(
                score_input, serving, p_fatigue, o_fatigue
            )
        
        if 'error' in result:
            print(f"\n❌ {result['error']}")
            continue
        
        # Display results
        print("\n" + "="*70)
        print("TACTICAL ANALYSIS")
        print("="*70)
        print(f"Game State: {result['game_state']}")
        
        sit = result['situation_analysis']
        print(f"\n📊 SITUATION:")
        print(f"  Pressure: {sit['pressure_level']}")
        print(f"  Momentum: {sit['momentum']}")
        print(f"  Game Phase: {sit['game_phase']}")
        print(f"  Priority: {sit['tactical_priority']}")

        sr = result['serve_or_return']
        rl = result['rally']

        print("\n🎯 START-OF-POINT STRATEGY (Serve/Return):")
        print(f"   Recommended: {sr['recommended_action'].upper()}")
        print(f"   Confidence (Q-value): {sr['q_value']:+.3f}")
        print(f"   Confidence %: {sr['confidence']:.1f}%")
        print(f"   Valid options: {', '.join(sr['valid_actions'])}")
        print(f"   💡 Reasoning: {sr['tactical_reasoning']}")

        print("\n🎾 RALLY STRATEGY (Ball in Play):")
        print(f"   Recommended: {rl['recommended_action'].upper()}")
        print(f"   Confidence (Q-value): {rl['q_value']:+.3f}")
        print(f"   Confidence %: {rl['confidence']:.1f}%")
        print(f"   Valid options: {', '.join(rl['valid_actions'])}")
        print(f"   💡 Reasoning: {rl['tactical_reasoning']}")

        # Top 3 for each phase
        print("\n📊 Start-of-Point Top 3:")
        print("-" * 70)
        shown = 0
        for info in sr['all_actions']:
            if not info['is_valid']:
                continue
            emoji = "🥇" if shown == 0 else "🥈" if shown == 1 else "🥉"
            print(f"{emoji} {info['action']:25s} Q={info['q_value']:+7.3f} ({info['confidence']:5.1f}%)")
            shown += 1
            if shown >= 3:
                break

        print("\n📊 Rally Top 3:")
        print("-" * 70)
        shown = 0
        for info in rl['all_actions']:
            if not info['is_valid']:
                continue
            emoji = "🥇" if shown == 0 else "🥈" if shown == 1 else "🥉"
            print(f"{emoji} {info['action']:25s} Q={info['q_value']:+7.3f} ({info['confidence']:5.1f}%)")
            shown += 1
            if shown >= 3:
                break
        
        print("="*70)


if __name__ == "__main__":
    interactive_predictor()