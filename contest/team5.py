# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
from game import Directions
import game
import random, time, util, sys
from util import nearestPoint
from winerror import SUCCEEDED


#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        #How big the playing grid is
        self.x, self.y = gameState.getWalls().asList()[-1]
        #List of walls on grid
        self.walls = list(gameState.getWalls())
        #Positions that aren't walls
        self.valid_positions = [position for position in gameState.getWalls().asList(False) if position[1] > 1]
        #Pathways on agent's side of grid
        self.valid_paths = []
        
        #Set the offset for each agent from the middle of the grid
        if self.red:
            offset = -3
        else:
            offset = 4
        
        #Check vertical paths...    
        for i in range(self.y):
            #If there is no wall at 'i' on the current side...
            if not self.walls[self.x / 2 + offset][i]:
                #If not a wall on current side...
                self.valid_paths.append(((self.x /2 + offset), i))
        
        #Set different starting positions for different agents; self.index = index for this agent
        if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
            x, y = self.valid_paths[3 * len(self.valid_paths) / 4]
        else:
            x, y = self.valid_paths[len(self.valid_paths) / 4]
        
        #Point the agent needs to go to
        self.goto = (x, y)
        
        self.o_weights = {'successorScore': 100,'distanceToFood': -1, 'numDefenders': -1000, 'defenderDistance': -10, 
                          'distanceToGoal': -1, 'stop': -100, 'reverse': -2}
        self.d_weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'distanceToGoal': -1, 
                          'stop': -100, 'reverse': -2}
    
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        """foodLeft = len(self.getFood(gameState).asList())
        
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
            return bestAction"""
        
        return random.choice(bestActions)
    
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        
        return features * weights
    
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        
        return features
    
    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        
        return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
        If an enemy is a ghost, the offensive agent labels it as a defender and determines its position 
        relative to its own. It also considers the position of the closest food relative to its own. It 
        checks the food list and if the distance from offensive agent to the closest food is less than 
        the distance to half the distance of the closest defender, then it goes for the food.
    """ 
    def registerInitialState(self, gameState):
        self.food_count = 0
        self.goal_food_count = 4 
    
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_state = successor.getAgentState(self.index)
        current_position = current_state.getPosition()
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)#self.getScore(successor)
        better = 999
        enemies = []
        defenders = []
        distances_to_defenders = []
        distances_to_food = []
        
        """
            Play Offensively
        """
        #Check the indices of the opponents...
        for agent in self.getOpponents(successor):
            #Add opponents to list of enemies
            enemies.append(successor.getAgentState(agent))
            
        #Check enemies...    
        for enemy in enemies:        
            #If there is an enemy position that we can see...
            if not enemy.isPacman and enemy.getPosition() != None:
                #Add that enemy to the list of defenders
                defenders.append(enemy)
                features['numDefenders'] = len(defenders)
                
        #If there is a defender...
        if len(defenders) > 0:
            #Check the indices of defenders...
            for d in defenders:  
                #Find the shortest distance to the defender from current position and add to list of defender distances      
                distances_to_defenders.append(self.getMazeDistance(current_position, d.getPosition()))
                features['defenderDistance'] = min(distances_to_defenders)
        
        # Compute distance to the nearest food
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            for food in foodList:
                distances_to_food.append(self.getMazeDistance(current_position, food))
                features['distanceToFood'] = min(distances_to_food)
        
        #Check food and determine the location to intercept        
        for food in foodList:
            if distances_to_food < distances_to_defenders:
                #Set the distance equal to the distance from the current position to the food
                distances_to_food = self.getMazeDistance(current_position, food)
                
                #If a distance is less than the value of the more important area... 
                if distances_to_food < better:
                    #Set the value of the more important area equal to that distance
                    better = distances_to_food
                    
                #Set the point of interception to that food
                intercept = food
            
            if distances_to_food < 9 and distances_to_food <= better and intercept != 0:
                #Go to that point to intercept
                self.goto = intercept
                
        features['distanceToGoal'] = self.getMazeDistance(current_position, self.goto)
        
        #If the agent is at the goto point...
        if self.getMazeDistance(current_position, self.goto) == 0:
            self.food_count = self.food_count + 1                
            
            if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
                self.goto = self.valid_paths[5 * len(self.valid_paths) / 6]
            else:
                self.goto = self.valid_paths[1 * len(self.valid_paths) / 6]
        
        if action == Directions.STOP:
            features['stop'] = 1
            
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        
        if action == rev:
            features['reverse'] = 1
        
        return features           

    def getWeights(self, gameState, action):
        return self.o_weights

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
        Our defensive agent patrols the middle border for any PacMan agents who dare to cross.
        If a PacMan agent is daring enough, the defensive agent labels it as an invader and
        determines its position relative to its own. It also considers any available paths and 
        compares the distance between those paths and the enemy's position with the distance between 
        its own position and the enemy's. Once a path has been chosen, our defensive agent goes in 
        for the kill!
    """
    
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_state = successor.getAgentState(self.index)
        current_position = current_state.getPosition()
        better = 999
        
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if current_state.isPacman: features['onDefense'] = 0
        
        enemies = []
        
        #Check the indices of the opponents...
        for agent in self.getOpponents(successor):
            #Add opponents to list of enemies
            enemies.append(successor.getAgentState(agent))
        
        invaders = []
        
        #Check enemies...    
        for enemy in enemies:        
            #If there is an enemy position that we can see...
            if enemy.isPacman and enemy.getPosition() != None:
                #Add that enemy to the list of invaders
                invaders.append(enemy)
                features['numInvaders'] = len(invaders)
        
        distances = []
        
        #If there is an invader...
        if len(invaders) > 0:
            #Check the indices of invaders...
            for e in invaders:  
                #Find the shortest distance to the invader from current position and add to list of distances      
                distances.append(self.getMazeDistance(current_position, e.getPosition()))
                features['invaderDistance'] = min(distances)
                #Intercept tracker
                intercept = 0;
        
        for e in invaders:
            #Check paths and determine the location to intercept
            for path in self.valid_paths:
                """
                    If the distance between the path and enemy location is less than the distance from the defensive 
                    agent's current position to the invader's position...
                """
                if self.getMazeDistance(path, e.getPosition()) < distances:
                    #Set the distance equal to the distance from the path to the enemy
                    distances = self.getMazeDistance(path, e.getPosition())
                    
                    #If a distance is less than the value of the more important area... 
                    if distances < better:
                        #Set the value of the more important area equal to that distance
                        better = distances
                        
                    #Set the point of interception to that path
                    intercept = path
            """
                If distance is less than 9 and greater than or equal to the value of the more important area and
                the agent has a path to intercept...
            """
            if distances < 9 and distances <= better and intercept != 0:
                #Go to that point to intercept
                self.goto = intercept
                
        #Check the indices of the opponents...        
        for e in invaders:
            #Coordinates of invader
            x, y = e.getPosition()
            
            #If on the red team and the enemy is on the agent's left...
            if self.red and x < self.x / 2:
                #Get him/her
                self.goto = e.getPosition()
            #Else if on the blue team and the enemy is on the agent's right...
            elif not self.red and x > self.x / 2:
                #Get him/her
                self.goto = e.getPosition()
                
        features['distanceToGoal'] = self.getMazeDistance(current_position, self.goto)
        
        #If the agent is at the goto point...
        if self.getMazeDistance(current_position, self.goto) == 0:
            #The defensive agent (on either team) will continue patrolling that area
            if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
                self.goto = self.valid_paths[5 * len(self.valid_paths) / 6]
            else:
                self.goto = self.valid_paths[1 * len(self.valid_paths) / 6]
        
        if action == Directions.STOP:
            features['stop'] = 1
            
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        
        if action == rev:
            features['reverse'] = 1
        
        return features
    
    def getWeights(self, gameState, action):
        return self.d_weights