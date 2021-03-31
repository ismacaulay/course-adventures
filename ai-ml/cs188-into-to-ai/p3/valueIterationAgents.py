# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def _compute_value_for_state(self, state):
        max_a = -10000
        for a in self.mdp.getPossibleActions(state):
            value = 0
            for ns, p in self.mdp.getTransitionStatesAndProbs(state, a):
                value += p * (self.mdp.getReward(state, a, ns) + self.discount * self.values[ns])
            if value > max_a:
                max_a = value
        return max_a


    def runValueIteration(self):
        # Write value iteration code here
        for i in range(0, self.iterations):
            vector_k = util.Counter()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    vector_k[s] = self.mdp.getReward(s, None, None)
                    continue

                vector_k[s] = self._compute_value_for_state(s)
            self.values = vector_k

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0
        for ns, p in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += p * (self.mdp.getReward(state, action, ns) + self.discount * self.values[ns])
        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best = None
        best_v = -10000
        for a in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, a)
            if q_value > best_v:
                best = a
                best_v = q_value
        return best


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            s = states[i % len(states)]
            if self.mdp.isTerminal(s):
                continue

            self.values[s] = self._compute_value_for_state(s)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = self._compute_predecessors()
        queue = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue

            diff = abs(self.values[s] - self._compute_max_q_value(s))
            queue.update(s, -diff)

        for i in range(0, self.iterations):
            if queue.isEmpty():
                break

            s = queue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = self._compute_value_for_state(s)

            for p in predecessors.get(s, set()):
                if self.mdp.isTerminal(p):
                    continue

                diff = abs(self.values[p] - self._compute_max_q_value(p))
                if diff > self.theta:
                    queue.push(p, -diff)


    def _compute_predecessors(self):
        predecessors = {}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for ns, p in self.mdp.getTransitionStatesAndProbs(s, a):
                    if p > 0:
                        # predecessors needs to a be an ORDERED set
                        cur = predecessors.get(ns, [])
                        if s not in cur:
                            cur.append(s)
                        predecessors[ns] = cur
        return predecessors

    def _compute_max_q_value(self, state):
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return 0

        q_value = -10000
        for a in self.mdp.getPossibleActions(state):
            v = self.computeQValueFromValues(state, a)
            if v > q_value:
                q_value = v
        return q_value
