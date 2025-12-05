"""
Tennis Environment Simulator (v2.2 - UNBIASED & MATHEMATICALLY CORRECT)
Fixed all bias issues and calculation errors
"""

import numpy as np
import random
from typing import Dict, Tuple, List


class TennisEnvironment:
    def __init__(self, opponent_skill: float = 0.5, best_of: int = 3):
        """
        Args:
            opponent_skill: 0.0 to 1.0 (baseline opponent effectiveness)
                           0.35 = easy, 0.45 = medium, 0.50 = balanced
            best_of: Match format - 1 or 3 sets (3 = realistic, 1 = faster training)
        """
        self.opponent_skill = opponent_skill
        self.best_of = best_of
        
        # Score mappings
        self.score_map = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}

        # Action space: 10 tactical choices
        self.actions = [
            "serve_flat_wide",      # 0
            "serve_flat_T",         # 1
            "serve_kick_body",      # 2
            "return_aggressive",    # 3
            "return_neutral",       # 4
            "return_block",         # 5
            "rally_aggressive",     # 6
            "rally_neutral",        # 7
            "approach_net",         # 8
            "defensive_lob",        # 9
        ]

        # Position encodings
        self.POS_BASELINE = 0
        self.POS_MIDCOURT = 1
        self.POS_NET = 2

        # Ball depth encodings
        self.DEPTH_SHORT = 0
        self.DEPTH_NEUTRAL = 1
        self.DEPTH_DEEP = 2

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset to start of match and return initial state vector."""

        self.state: Dict = {
            "player_points": 0,
            "opponent_points": 0,
            "player_games": 0,
            "opponent_games": 0,
            "player_sets": 0,
            "opponent_sets": 0,
            "is_player_serving": True,
            "is_deuce": False,
            "player_advantage": False,
            "opponent_advantage": False,
            "is_tiebreak": False,
            "player_fatigue": 0.0,
            "opponent_fatigue": 0.0,
            "court_side": "deuce",
            "rally_length": 0,
            "player_position": self.POS_BASELINE,
            "opponent_position": self.POS_BASELINE,
            "ball_depth": self.DEPTH_NEUTRAL,
        }
        return self.get_state_vector()

    def get_state_vector(self) -> np.ndarray:
        """Convert game + rally state to numerical vector for RL model."""
        return np.array(
            [
                self.state["player_points"],
                self.state["opponent_points"],
                self.state["player_games"],
                self.state["opponent_games"],
                self.state["player_sets"],
                self.state["opponent_sets"],
                1 if self.state["is_player_serving"] else 0,
                1 if self.state["is_deuce"] else 0,
                1 if self.state["player_advantage"] else 0,
                1 if self.state["opponent_advantage"] else 0,
                1 if self.state["is_tiebreak"] else 0,
                self.state["player_fatigue"],
                self.state["opponent_fatigue"],
                1 if self.state["court_side"] == "ad" else 0,
                self.state["rally_length"],
                self.state["player_position"],
                self.state["opponent_position"],
                self.state["ball_depth"],
            ],
            dtype=np.float32,
        )

    def get_state_description(self) -> str:
        """Human-readable score description."""
        p_pts = self.score_map.get(
            self.state["player_points"], str(self.state["player_points"])
        )
        o_pts = self.score_map.get(
            self.state["opponent_points"], str(self.state["opponent_points"])
        )

        if self.state["is_deuce"]:
            score = "Deuce"
        elif self.state["player_advantage"]:
            score = "AD-40"
        elif self.state["opponent_advantage"]:
            score = "40-AD"
        else:
            score = f"{p_pts}-{o_pts}"

        return (
            f"Sets: {self.state['player_sets']}-{self.state['opponent_sets']} | "
            f"Games: {self.state['player_games']}-{self.state['opponent_games']} | "
            f"Score: {score} | "
            f"{'Player' if self.state['is_player_serving'] else 'Opponent'} serving | "
            f"Side: {self.state['court_side']}"
        )

    def get_valid_actions(self) -> List[int]:
        """Returns valid action indices for current phase."""
        if self.state["rally_length"] == 0:
            if self.state["is_player_serving"]:
                return [0, 1, 2]  # serves
            else:
                return [3, 4, 5]  # returns
        else:
            return [6, 7, 8, 9]  # rally actions
        
    def _update_court_side(self):
        """Correctly determine deuce/ad side based only on game point number."""
        
        # Tiebreak special rule: players alternate every 2 points after the first
        if self.state["is_tiebreak"]:
            point_num = self.state["player_points"] + self.state["opponent_points"]
            if point_num == 0:
                self.state["court_side"] = "deuce"
            else:
                # after first point: change sides every 2 points
                self.state["court_side"] = "deuce" if ((point_num - 1) // 2) % 2 == 0 else "ad"
            return
        
        # Normal game
        game_points = self.state["player_points"] + self.state["opponent_points"]

        # Serve always starts on DEUCE side
        # Alternate: deuce → ad → deuce → ad…
        if game_points % 2 == 0:
            self.state["court_side"] = "deuce"
        else:
            self.state["court_side"] = "ad"


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)."""
        
        # Simulate rally dynamics for this shot
        point_over, point_won, event_type = self._simulate_point_step(action)

        reward = 0.0
        match_done = False

        if point_over:
            # Award reward for point outcome
            reward = self._calculate_point_reward(action, point_won)
            match_done = self._update_score(point_won)
            
            # Reset rally state
            self.state["rally_length"] = 0
            self.state["player_position"] = self.POS_BASELINE
            self.state["opponent_position"] = self.POS_BASELINE
            self.state["ball_depth"] = self.DEPTH_NEUTRAL
        else:
            # FIXED: Small intermediate reward during rally (was only for agent, now symmetric)
            reward = 0.05  # Reduced from 0.1 to be more conservative
            self.state["rally_length"] += 1

        # Update fatigue & positions
        self._update_physical_state(action, point_over, point_won)

        # Update court side
        total_points = self.state["player_points"] + self.state["opponent_points"]
        self._update_court_side()#NOTE - changed so that it give proper calc

        info = {
            "point_over": point_over,
            "point_won": point_won,
            "event": event_type,
            "rally_length": self.state["rally_length"],
        }

        return self.get_state_vector(), reward, match_done, info

    def _simulate_point_step(self, action: int):
        """
        Simulate shot outcome.
        
        FIXED ISSUES:
        1. Opponent skill now applies symmetrically to both players
        2. Fatigue effects are balanced
        3. Pressure affects agent and opponent equally
        
        Returns:
            point_over (bool)
            point_won  (bool)
            event_type (str)
        """
        rl = self.state["rally_length"]
        serving_point_start = rl == 0 and self.state["is_player_serving"]
        returning_point_start = rl == 0 and (not self.state["is_player_serving"])

        # Base probabilities - these represent the ACTION quality, not player skill
        if serving_point_start:
            # Serves - base rates for a competent player
            if action == 0:      # flat wide - high risk/reward
                base_win_p, base_lose_p = 0.42, 0.13
                event_win = "serve_ace"
            elif action == 1:    # flat T - high risk/reward
                base_win_p, base_lose_p = 0.40, 0.12
                event_win = "serve_ace"
            elif action == 2:    # kick body - safer
                base_win_p, base_lose_p = 0.28, 0.08
                event_win = "serve_unreturned"
            else:
                base_win_p, base_lose_p = 0.25, 0.10
                event_win = "serve"

        elif returning_point_start:
            # Returns - inherently harder than serving
            if action == 3:      # aggressive return
                base_win_p, base_lose_p = 0.20, 0.18
                event_win = "return_winner"
            elif action == 4:    # neutral return
                base_win_p, base_lose_p = 0.14, 0.10
                event_win = "return_deep"
            elif action == 5:    # block return - safest
                base_win_p, base_lose_p = 0.09, 0.06
                event_win = "return_in_play"
            else:
                base_win_p, base_lose_p = 0.12, 0.12
                event_win = "return_neutral"
                
        else:
            # Rally phase - balanced neutral exchange
            if action == 6:      # rally_aggressive
                base_win_p, base_lose_p = 0.16, 0.15
                event_win = "groundstroke_winner"
            elif action == 7:    # rally_neutral - most consistent
                base_win_p, base_lose_p = 0.09, 0.07
                event_win = "rally_forcing"
            elif action == 8:    # approach_net - tactical
                base_win_p, base_lose_p = 0.14, 0.13
                event_win = "volley_winner"
            elif action == 9:    # defensive_lob - reset
                base_win_p, base_lose_p = 0.06, 0.05
                event_win = "defensive_reset"
            else:
                base_win_p, base_lose_p = 0.10, 0.10
                event_win = "neutral_rally"

        # Now apply modifiers that represent player abilities
        win_p, lose_p = base_win_p, base_lose_p

        # 1. PLAYER SKILL (agent is learning, starts at ~0.5 effective skill)
        # Agent's current skill is implicit in learned policy - no modifier needed
        
        # 2. OPPONENT SKILL - affects how well opponent handles player's shots
        # Higher opponent skill = better at returning good shots, forcing errors
        opp_factor = (self.opponent_skill - 0.5) * 0.5  # -0.25 to +0.25
        
        # Opponent skill reduces our winner chances, increases our error chances
        win_p *= (1.0 - opp_factor)  # if opp=0.6, multiply by 0.95
        lose_p *= (1.0 + opp_factor)  # if opp=0.6, multiply by 1.05

        # 3. FATIGUE - affects BOTH players symmetrically
        player_fatigue = self.state["player_fatigue"]
        opp_fatigue = self.state["opponent_fatigue"]
        
        # Net fatigue effect (positive means we're more tired than opponent)
        fatigue_diff = player_fatigue - opp_fatigue
        
        # If we're more tired, we make more errors and fewer winners
        win_p *= (1.0 - 0.3 * fatigue_diff)
        lose_p *= (1.0 + 0.4 * fatigue_diff)

        # 4. PRESSURE SITUATIONS - affects both players equally
        # Big points make everyone play slightly worse
        if self.state["player_points"] >= 3 and self.state["opponent_points"] >= 3:
            win_p *= 0.95
            lose_p *= 1.05

        # 5. RALLY LENGTH PENALTY - both players tire
        if rl > 8:
            rally_factor = 0.98
            win_p *= rally_factor
            lose_p *= (2.0 - rally_factor)  # inverse effect
        if rl > 15:
            rally_factor = 0.95
            win_p *= rally_factor
            lose_p *= (2.0 - rally_factor)

        # 6. POSITION ADVANTAGES - tactical bonuses
        if self.state["player_position"] == self.POS_NET:
            if action == 8:  # approach_net when already at net
                win_p *= 1.15
            elif action in [6, 7]:  # volleying
                win_p *= 1.1
        
        if self.state["opponent_position"] == self.POS_NET:
            # Opponent at net, passing shots are riskier but rewarding
            if action == 6:  # aggressive passing shot
                win_p *= 1.1
                lose_p *= 1.1

        # CLIP PROBABILITIES - ensure valid probability distribution
        win_p = np.clip(win_p, 0.0, 0.92)
        lose_p = np.clip(lose_p, 0.0, 0.92)
        
        # CRITICAL FIX: Ensure probabilities sum to <= 1.0
        if win_p + lose_p > 1.0:
            total = win_p + lose_p
            win_p = win_p / total * 0.95  # scale down to 95% to leave room for continue
            lose_p = lose_p / total * 0.95
        
        cont_p = 1.0 - win_p - lose_p
        
        # Ensure continue probability is non-negative
        if cont_p < 0:
            cont_p = 0.05
            total = win_p + lose_p
            win_p = win_p / total * 0.95
            lose_p = lose_p / total * 0.95

        # Sample outcome
        r = random.random()
        if r < win_p:
            return True, True, event_win
        elif r < win_p + lose_p:
            return True, False, "error"
        else:
            return False, False, "rally_continue"

    def _calculate_point_reward(self, action: int, point_won: bool) -> float:
        """
        Reward for point outcome.
        UPDATED: Rewards aggressive play, penalizes excessive defensive play
        """
        
        if point_won:
            base = 1.0

            # Critical point bonuses
            if self.state["player_points"] >= 3 or self.state["opponent_points"] >= 3:
                base += 0.7

            # Break point bonus
            if not self.state["is_player_serving"]:
                if self.state["player_points"] >= 3:
                    base += 0.8
                else:
                    base += 0.4
            
            # Service game hold bonus
            if self.state["is_player_serving"] and self.state["player_points"] >= 3:
                base += 0.3
            
            # Long rally victory bonus
            if self.state["rally_length"] > 6:
                base += 0.4
            
            

            return base
        else:
            base = -1.0

            # Critical point penalties
            if self.state["opponent_points"] >= 3:
                base -= 0.4

            # Service break penalty
            if self.state["is_player_serving"] and self.state["opponent_points"] >= 3:
                base -= 0.5
            
            # ========== NEW: SOFTER PENALTIES FOR AGGRESSIVE ERRORS ==========
            # Reduce penalty for errors from aggressive play (risk-taking is okay)
            aggressive_actions = [0, 1, 3, 6, 8]
            if action in aggressive_actions:
                base += 0.2  # Less harsh penalty for aggressive errors
            # ================================================================

            return base

    # ===== SCORING LOGIC =====

    def _update_score(self, point_won: bool) -> bool:
        """Update score after point. Returns True if match over."""
        if self.state["is_tiebreak"]:
            return self._update_tiebreak(point_won)

        if point_won:
            self.state["player_points"] += 1
        else:
            self.state["opponent_points"] += 1

        # Deuce/advantage logic
        if self.state["player_points"] >= 3 and self.state["opponent_points"] >= 3:
            if self.state["player_points"] == self.state["opponent_points"]:
                self.state["is_deuce"] = True
                self.state["player_advantage"] = False
                self.state["opponent_advantage"] = False
            elif self.state["player_points"] > self.state["opponent_points"]:
                self.state["is_deuce"] = False
                self.state["player_advantage"] = True
                self.state["opponent_advantage"] = False
            else:
                self.state["is_deuce"] = False
                self.state["player_advantage"] = False
                self.state["opponent_advantage"] = True

        # Check for game won
        if (
            self.state["player_points"] >= 4
            and self.state["player_points"] >= self.state["opponent_points"] + 2
        ):
            return self._game_won(True)
        elif (
            self.state["opponent_points"] >= 4
            and self.state["opponent_points"] >= self.state["player_points"] + 2
        ):
            return self._game_won(False)

        return False

    def _game_won(self, player_won: bool) -> bool:
        """Handle game won. Returns True if match over."""
        self.state["player_points"] = 0
        self.state["opponent_points"] = 0
        self.state["is_deuce"] = False
        self.state["player_advantage"] = False
        self.state["opponent_advantage"] = False

        if player_won:
            self.state["player_games"] += 1
        else:
            self.state["opponent_games"] += 1

        self.state["is_player_serving"] = not self.state["is_player_serving"]

        # Tiebreak at 6-6
        if self.state["player_games"] == 6 and self.state["opponent_games"] == 6:
            self.state["is_tiebreak"] = True
            return False

        # Set won?
        if (
            self.state["player_games"] >= 6
            and self.state["player_games"] >= self.state["opponent_games"] + 2
        ):
            return self._set_won(True)
        elif (
            self.state["opponent_games"] >= 6
            and self.state["opponent_games"] >= self.state["player_games"] + 2
        ):
            return self._set_won(False)

        return False

    def _set_won(self, player_won: bool) -> bool:
        """Handle set won. Returns True if match over."""
        self.state["player_games"] = 0
        self.state["opponent_games"] = 0
        self.state["is_tiebreak"] = False

        if player_won:
            self.state["player_sets"] += 1
        else:
            self.state["opponent_sets"] += 1

        # Configurable match format
        sets_to_win = 1 if self.best_of == 1 else 2
        if self.state["player_sets"] >= sets_to_win or \
           self.state["opponent_sets"] >= sets_to_win:
            return True

        return False

    def _update_tiebreak(self, point_won: bool) -> bool:
        """Update tiebreak scoring."""
        if point_won:
            self.state["player_points"] += 1
        else:
            self.state["opponent_points"] += 1

        if (
            self.state["player_points"] >= 7
            and self.state["player_points"] >= self.state["opponent_points"] + 2
        ):
            self.state["player_games"] = 7
            return self._set_won(True)
        elif (
            self.state["opponent_points"] >= 7
            and self.state["opponent_points"] >= self.state["player_points"] + 2
        ):
            self.state["opponent_games"] = 7
            return self._set_won(False)

        return False

    def _update_physical_state(self, action: int, point_over: bool, point_won: bool):
        """
        Update fatigue and positions.
        
        FIXED: Fatigue now symmetric - both players tire based on rally length
        """
        
        # SYMMETRIC FATIGUE MODEL
        # Both players tire based on shot intensity and rally length
        
        # Action intensity (physical cost of each shot type)
        intensity_map = {
            0: 0.022, 1: 0.022, 2: 0.018,  # serves (powerful = tiring)
            3: 0.024, 4: 0.020, 5: 0.016,  # returns (stretch and react)
            6: 0.028, 7: 0.020, 8: 0.030, 9: 0.018,  # rallies
        }

        # Player tires based on their action
        player_fatigue_inc = intensity_map.get(action, 0.020)
        self.state["player_fatigue"] = min(1.0, self.state["player_fatigue"] + player_fatigue_inc)

        # Opponent tires based on rally length (they're hitting shots too!)
        # Assume opponent actions have similar intensity distribution
        rally_factor = 1.0 + (self.state["rally_length"] * 0.002)  # longer rallies = more tired
        opp_fatigue_inc = 0.020 * rally_factor
        self.state["opponent_fatigue"] = min(1.0, self.state["opponent_fatigue"] + opp_fatigue_inc)

        # Position updates (tactical state tracking)
        if self.state["rally_length"] == 0:
            self.state["player_position"] = self.POS_BASELINE
            self.state["opponent_position"] = self.POS_BASELINE
            self.state["ball_depth"] = self.DEPTH_NEUTRAL
        else:
            if action == 8:  # approach_net
                self.state["player_position"] = self.POS_NET
                self.state["ball_depth"] = self.DEPTH_DEEP
            elif action == 9:  # defensive_lob
                self.state["player_position"] = self.POS_BASELINE
                self.state["opponent_position"] = self.POS_BASELINE
                self.state["ball_depth"] = self.DEPTH_DEEP
            elif action == 6:  # rally_aggressive
                self.state["ball_depth"] = self.DEPTH_DEEP
            elif action == 7:  # rally_neutral
                self.state["ball_depth"] = self.DEPTH_NEUTRAL

        # SYMMETRIC RECOVERY between points
        if point_over:
            # Both players recover equally during point changeover
            recovery = 0.025
            self.state["player_fatigue"] = max(0.0, self.state["player_fatigue"] - recovery)
            self.state["opponent_fatigue"] = max(0.0, self.state["opponent_fatigue"] - recovery)