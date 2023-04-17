import game as g
import agent as ag
import matplotlib.pylab as plt
import numpy as np


def plot_agent_reward(rewards):
    """ Function to plot agent's accumulated reward vs. iteration """
    plt.plot(np.cumsum(rewards))
    plt.title('Agent Cumulative Reward vs. Iteration')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()


class GameLearning:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.01):
        while True:
            print('\n----------- Choose the play mode: ----------- ')
            type = input('1. AgentRL vs AgentAI \n2. AgentRL vs Human \n3. AgentRL vs AgentRL\nYour choice : ')
            if type == '1' or type == '2' or type == '3':
                break
        if type == '1':
            self.game = g.Game('agentAI')
        elif type == '2':
            self.game = g.Game('human')
        else:
            self.game = g.Game('agentRL')

    def beginPlaying(self, episodes):
        self.game.play()


if __name__ == '__main__':
    gl = GameLearning()
    gl.beginPlaying(200)
