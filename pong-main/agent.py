import numpy as np

ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class Qlearning:
    """
     class for Q-learning .
    Parameters
    ----------
    alpha : float
        learning rate
    gamma : float
        temporal discounting rate
    eps : float
        probability of random action vs. greedy action
    eps_decay : float
        epsilon decay rate. Larger value = more decay
    """

    def __init__(self, screen_height, bar_height, alpha, gamma):
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros([int(screen_height / bar_height), 2])
        self.rewards = []
        self.state = int((screen_height / bar_height) / 2)
        self.action = 0

    def get_action(self, s):
        """
        Returns an action for a given state.
        Parameters
        ----------
        s : string
            state
        Returns
        -------
        (i,j) tuple
            action to take in state s
        """
        return np.argmax(self.Q[s, :])

    def update(self, s, bar, ball, screen_height, ball_speed_x, is_permanent):
        """
        Perform the Q-Learning update of Q values.
        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        r : int
            reward received after executing action "a" in state "s"
        """
        s_ = s
        position_cal = bar.right + 10
        speed = ball_speed_x * (-1)
        ballX = ball.x
        if not is_permanent:
            position_cal = bar.left - 10 - ball.width
            speed = ball_speed_x
            ballX = position_cal
            position_cal = ball.x
        if position_cal <= ballX and speed > 0:
            reward = self.calculate_reward(bar, ball)
            self.rewards.append(reward)
            self.action = self.get_action(s)
            if self.action != 0:
                s_ = self.center_to_state(ball.centery, screen_height, bar.height)
            else:
                s_ = s
            if s_ < 0:
                s_ = 0
            elif s_ > int(screen_height / bar.height) - 1:
                s_ = int(screen_height / bar.height) - 1
            self.state = s_
            self.Q[s, self.action] += self.alpha * (
                        reward + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, self.action])
        return s_ * bar.height

    def calculate_reward(rect, bar, ball):
        if bar.top <= ball.centery <= bar.bottom:
            return 1
        else:
            return -1

    def center_to_state(self, center, screen_height, bar_height):
        a = 0
        b = bar_height
        s = 0
        for i in range(int(screen_height / bar_height)):
            if a < center < b:
                s = (b / bar_height) - 1
            else:
                a += bar_height
                b += bar_height
        return int(s)

