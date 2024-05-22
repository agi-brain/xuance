"""
Football Benchmarks:
    - 11_vs_11_stochastic: A full 90 minutes football game (medium difficulty)
    - 11_vs_11_easy_stochastic: A full 90 minutes football game (easy difficulty)
    - 11_vs_11_hard_stochastic: A full 90 minutes football game (hard difficulty)

Football Academy - with a total of 11 scenarios
    - academy_empty_goal_close: Our player starts inside the box with the ball, and needs to score against an empty goal.
    - academy_empty_goal: Our player starts in the middle of the field with the ball, and needs to score against an empty goal.
    - academy_run_to_score: Our player starts in the middle of the field with the ball, and needs to score against an empty goal. Five opponent players chase ours from behind.
    - academy_run_to_score_with_keeper: Our player starts in the middle of the field with the ball, and needs to score against a keeper. Five opponent players chase ours from behind.
    - academy_pass_and_shoot_with_keeper: Two of our players try to score from the edge of the box, one is on the side with the ball, and next to a defender. The other is at the center, unmarked, and facing the opponent keeper.
    - academy_run_pass_and_shoot_with_keeper: Two of our players try to score from the edge of the box, one is on the side with the ball, and unmarked. The other is at the center, next to a defender, and facing the opponent keeper.
    - academy_3_vs_1_with_keeper: Three of our players try to score from the edge of the box, one on each side, and the other at the center. Initially, the player at the center has the ball and is facing the defender. There is an opponent keeper.
    - academy_corner: Standard corner-kick situation, except that the corner taker can run with the ball from the corner.
    - academy_counterattack_easy: 4 versus 1 counter-attack with keeper; all the remaining players of both teams run back towards the ball.
    - academy_counterattack_hard: 4 versus 2 counter-attack with keeper; all the remaining players of both teams run back towards the ball.
    - academy_single_goal_versus_lazy: Full 11 versus 11 games, where the opponents cannot move but they can only intercept the ball if it is close enough to them. Our center back defender has the ball at first.
"""

from xuance.environment.football.gfootball_vec_env import SubprocVecEnv_GFootball, DummyVecEnv_GFootball
