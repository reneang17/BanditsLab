#!/usr/bin/env python

from BaseEnvironment import BaseEnvironment
from scipy.stats import norm

import numpy as np

class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Required methods:
        env_init, env_start, env_step, env_cleanup, and env_message
    """

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.arms = None
        self.seed = None

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        arms_number = env_info.get('num_arms',10)

        self.arms = norm(0.0, 1.0).rvs(size=arms_number)
        #np.random.randn(arms_number)#[np.random.normal(0.0, 1.0) for _ in range(10)]

        self.arm_distributions = [norm(i,1) for i in self.arms]

        local_observation = 0

        self.reward_obs_term = (0.0, local_observation, False)


    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        reward = self.arm_distributions[action].rvs() # np.random.randn() #norm(0, 1).rvs()

        obs = self.reward_obs_term[1]

        self.reward_obs_term = (reward, obs, False)

        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
