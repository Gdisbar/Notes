import random
from typing import List, Dict

class Dice(object):
	def __init__(self, noOfDice: int):
		self.noOfDice = noOfDice

	def rollDice(self) -> int:
		diceValue = random.randint(1*self.noOfDice,6*self.noOfDice)
		return diceValue
	

class Player(object):
	def __init__(self, playerName: str,id: int):
		self.playerName = playerName
		self.id = id

class Jumper(object):
	def __init__(self, startPoint: int,endPoint: int):
		self.startPoint = startPoint
		self.endPoint = endPoint


class GameBoard(object):
    def __init__(self, dice: Dice, nextTurn: List, snakes: List[Jumper], ladders: List[Jumper], playersCurrentPosition: Dict[str, int], boardSize: int):
        super(GameBoard, self).__init__()
        self.dice = dice
        self.nextTurn = nextTurn
        self.snakes = snakes
        self.ladders = ladders
        self.playersCurrentPosition = playersCurrentPosition
        self.boardSize = boardSize


    def startGame(self):
        while len(self.nextTurn) > 1:
            nextPlayer = self.nextTurn.pop(0)
            currentPosition = self.playersCurrentPosition.get(nextPlayer.playerName)
            diceValue = self.dice.rollDice()
            nextCell = currentPosition + diceValue
            if nextCell > self.boardSize:
                self.nextTurn.append(nextPlayer)
            elif nextCell == self.boardSize:
                print(f"Player {nextPlayer.playerName} has won ")
                break
            else:
                for snake in self.snakes:
                    if snake.startPoint == nextCell:
                        nextCell = snake.endPoint
                        print(f"Player {nextPlayer.playerName} has bitten by snake, now at position {nextCell}")
                for ladder in self.ladders:
                    if ladder.startPoint == nextCell:
                        nextCell = ladder.endPoint
                        print(f"Player {nextPlayer.playerName} has got ladder, now at position {nextCell}")
                if nextCell == self.boardSize:
                    print(f"Player {nextPlayer.playerName} has won ")
                    break
                else:
                    self.playersCurrentPosition[nextPlayer.playerName] = nextCell
                    self.nextTurn.append(nextPlayer)
                    print(f"Player {nextPlayer.playerName} is at position {nextCell}")


if __name__=="__main__":
    dice = Dice(1)
    p1 = Player("Alice",1)
    p2 = Player("Bob",2)
    allPlayers = list()
    allPlayers.append(p1)
    allPlayers.append(p2)
    s1 = Jumper(10,2)
    s2 = Jumper(99,12)
    snakes = list()
    snakes.append(s1)
    snakes.append(s2)
    l1 = Jumper(5,25)
    l2 = Jumper(40,89)
    ladders = list()
    ladders.append(l1)
    ladders.append(l2)
    playerCurrentPosition = dict()
    playerCurrentPosition["Alice"]=0
    playerCurrentPosition["Bob"]=0
    boardSize=100
    gb = GameBoard(dice,allPlayers,snakes,ladders,playerCurrentPosition,boardSize)
    gb.startGame()
		
