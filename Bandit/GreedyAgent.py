#!/usr/bin/env python
import numpy as np
import main_agent

class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):

        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent recieved from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        ### Useful Class Variables ###
        # self.q_values : An array with what the agent believes each of the values of the arm are.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step
        #######################

        # Update Q values
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] += 1/self.arm_count[self.last_action]*(
            reward-self.q_values[self.last_action])

        # update current action
        current_action = self.argmax(self.q_values)

        self.last_action = current_action
        return current_action


    def argmax(self, q_values):
        """
        Takes in a list of q_values and returns the index of the item
        with the highest value. Breaks ties randomly.
        returns: int - the index of the highest value in q_values
        """
        top_value = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i]>top_value:
                ties = [i]
                top_value = q_values[i]
            elif q_values[i] == top_value:
                ties.append(i)

        return np.random.choice(ties)
