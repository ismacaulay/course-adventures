# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        # util.pause()
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        curPos = currentGameState.getPacmanPosition()

        # PUSH FORWARD!, you can turn, or you can go straight, but dont stop or go backwards
        if action == Directions.STOP or action == Directions.REVERSE[currentGameState.getPacmanState().getDirection()]:
            return -99999

        res = successorGameState.getScore()
        # lets try to maximize our average distance from ghosts
        avg_dist_from_ghosts = 0
        if len(newGhostStates) > 0:
            avg_dist_from_ghosts = sum([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
            avg_dist_from_ghosts /= len(newGhostStates)
        res += avg_dist_from_ghosts

        # lets get as close to food as possible
        foodList = newFood.asList()
        avg_dist_from_food = 0
        if len(foodList) > 0:
            avg_dist_from_food = sum([util.manhattanDistance(newPos, food) for food in foodList])
            avg_dist_from_food /= len(foodList)
        res -= avg_dist_from_food

        return res

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        value, action = self._max_value(gameState, 0, 1)
        return action

    def _value(self, gameState, agentIndex, depth):
        # if we are in a win state, or lose state there will be no more
        # legal actions, so just return the value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            # if we are checking pacman (agent 0) then we have started
            # a new ply. if we have reached the max depth, just return
            # the value, otherwise start the next ply
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            return self._max_value(gameState, agentIndex, depth + 1)

        return self._min_value(gameState, agentIndex, depth)

    def _max_value(self, gameState, agentIndex, depth):
        v = float("-inf")
        a = None
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, depth)
            if value > v:
                v = value
                a = action
        return v, a

    def _min_value(self, gameState, agentIndex, depth):
        v = float("inf")
        a = None
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, depth)
            if value < v:
                v = value
                a = action
        return v, a


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha, beta = float("-inf"), float("inf")
        value, action = self._max_value(gameState, 0, alpha, beta, 1)
        return action

    def _value(self, gameState, agentIndex, alpha, beta, depth):
        # if we are in a win state, or lose state there will be no more
        # legal actions, so just return the value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            # if we are checking pacman (agent 0) then we have started
            # a new ply. if we have reached the max depth, just return
            # the value, otherwise start the next ply
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            return self._max_value(gameState, agentIndex, alpha, beta, depth + 1)

        return self._min_value(gameState, agentIndex, alpha, beta, depth)

    def _max_value(self, gameState, agentIndex, alpha, beta, depth):
        v = float("-inf")
        a = None
        local_alpha = alpha

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, local_alpha, beta, depth)
            if value > v:
                v = value
                a = action
                if v > beta:
                    return v, a
                local_alpha = max(local_alpha, v)

        return v, a

    def _min_value(self, gameState, agentIndex, alpha, beta, depth):
        v = float("inf")
        a = None
        local_beta = beta

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, alpha, local_beta, depth)
            if value < v:
                v = value
                a = action
                if v < alpha:
                    return v, a
                local_beta = min(local_beta, v)

        return v, a


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        value, action = self._max_value(gameState, 0, 1)
        return action

    def _value(self, gameState, agentIndex, depth):
        # if we are in a win state, or lose state there will be no more
        # legal actions, so just return the value
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            # if we are checking pacman (agent 0) then we have started
            # a new ply. if we have reached the max depth, just return
            # the value, otherwise start the next ply
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            return self._max_value(gameState, agentIndex, depth + 1)

        return self._exp_value(gameState, agentIndex, depth)

    def _max_value(self, gameState, agentIndex, depth):
        v = float("-inf")
        a = None
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, depth)
            if value > v:
                v = value
                a = action
        return v, a

    def _exp_value(self, gameState, agentIndex, depth):
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        p = 1 / len(actions)

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        for action in actions:
            value, _ = self._value(gameState.generateSuccessor(agentIndex, action), next_agent, depth)
            v += p * value
        return v, None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Prefer states that maximize score, while keeping pacman
                 near the food with scared ghosts
    """
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    power_pellets = currentGameState.getCapsules()

    # prefer states that maximize our score
    res = currentGameState.getScore()

    # lets prefer states that maximize scared times
    res += sum(scaredTimes)

    # lets prefer states that get us close to the food
    foodList = food.asList()
    avg_dist_from_food = 0
    if len(foodList) > 0:
        avg_dist_from_food = sum([util.manhattanDistance(pos, food) for food in foodList])
        avg_dist_from_food /= len(foodList)
    res -= avg_dist_from_food

    return res

# Abbreviation
better = betterEvaluationFunction
