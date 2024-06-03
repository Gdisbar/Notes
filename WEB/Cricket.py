# class BatsmanStats(object):
# 	def __init__(self,name):
# 		self.name = name
# 		self.score = 0
# 		self.Four = 0
# 		self.Six = 0
# 		self.TotalBalls = 0
# 		self.hasPlayed = 0


# class PlayInnings(object):
# 	def __init__(self,noOfOvers,team,teamOut,TotalRun,TotalWC,teamName,TotalOver):
# 		self.noOfOvers = noOfOvers
# 		self.team = team
# 		self.teamOut = teamOut
# 		self.TotalRun = TotalRun
# 		self.TotalWC = TotalWC
# 		self.teamName  = teamName
# 		self.TotalOver = TotalOver
			

# 	# @staticmethod
# 	def printScore(self,striker,nonStriker,overNo):
# 		# print(f"After over {overNo} ")
# 		print(f"Scorecard for Team {self.teamName}:")
# 		print("Player Name Score 4s 6s Balls")
# 		for member in self.teamOut:
# 			# if member.name in [striker.name,nonStriker.name]: continue
# 			print(f"{member.name} {member.score} {member.Four} {member.Six} {member.TotalBalls}")

# 		# if striker.name not None:
# 		if striker!=None:
# 			print(f"{striker.name}* {striker.score} {striker.Four} {striker.Six} {striker.TotalBalls}")
# 		print(f"{nonStriker.name}* {nonStriker.score} {nonStriker.Four} {nonStriker.Six} {nonStriker.TotalBalls}")

# 		for member in self.team:
# 			print(f"{member.name} {member.score} {member.Four} {member.Six} {member.TotalBalls}")
		

# 		print(f"Total: {self.TotalRun}/{self.TotalWC}")
# 		overs = self.TotalOver//6
# 		leftOver = self.TotalOver - overs
# 		if leftOver==0:
# 			print(f"Overs: {overs}")
# 		else:
# 			print(f"Overs: {overs}.{leftOver}")


# 	def PlayCurrentOver(self,striker,nonStriker,overNo):
# 		ballCount = 6
# 		while ballCount>0:
# 			ball = input()
# 			if ball=='Wd':
# 				# striker.score+=1 ## wd run goes to tam not strikr
# 				striker.TotalBalls+=1  
# 				# ballCount+=1 ## already itrated over the loop 
# 				self.TotalRun+=1
# 				print(f"After Ball : {ball} striker : {striker.name} nonStriker : {nonStriker.name}")
# 				continue
# 			else:
# 				ballCount-=1
# 				striker.TotalBalls+=1 ## out is also a bllfaced by striker
# 				self.TotalOver+=1
# 				if ball=='W':
# 					self.TotalWC+=1
# 					self.teamOut.append(striker)
# 					if self.team:
# 						striker = self.team.pop(0)
# 						striker.hasPlayed = 1
# 						# striker,nonStriker = nonStriker,striker
# 					else:
# 						striker = None
# 						break
# 					print(f"After Ball : {ball} striker : {striker.name} nonStriker : {nonStriker.name}")
# 				else:
					
# 					ball = int(ball)
# 					if ball in [1,3]:
# 						striker.score+=ball 
# 						self.TotalRun+=ball
# 						striker,nonStriker = nonStriker,striker
# 					elif ball in [0,2,4,6]:
# 						striker.score+=ball 
# 						self.TotalRun+=ball
# 						if ball==4:
# 							striker.Four+=1
# 						if ball==6:
# 							striker.Six+=1
# 					print(f"After Ball : {ball} striker : {striker.name} nonStriker : {nonStriker.name}")
						
# 			# if len(self.team) < 2:
# 			# 	break
		
# 		self.printScore(striker,nonStriker,overNo)
# 		striker,nonStriker = nonStriker,striker
# 		# if isinstance(ballScore[-1],int) and int(ballScore[-1])%2==0:
# 		# 	striker,nonStriker = nonStriker,striker
# 		return striker, nonStriker

# 	def StartGame(self):
# 		## assuming team has at least 2 players
# 		striker = self.team.pop(0)
# 		nonStriker = self.team.pop(0)
# 		striker.hasPlayed = 1
# 		nonStriker.hasPlayed = 1
# 		print(f"Opener(striker) : {striker.name} Opener(nonStriker) : {nonStriker.name}")
# 		for i in range(self.noOfOvers):
# 			if self.TotalWC == len(self.team)-1:
# 				break
# 			print(f"Over {i+1}:")
# 			striker,nonStriker = self.PlayCurrentOver(striker,nonStriker,i)
# 		return self.TotalRun

# noOfPlayers = int(input("No. of players for each team: "))
# noOfOvers = int(input("No. of overs:"))


# print("Batting Order for team 1:")
# team1 = list()
# team1_out = list()
# for i in range(noOfPlayers):
# 	name = input()
# 	x = BatsmanStats(name)
# 	team1.append(x)

# sobj_1 = PlayInnings(noOfOvers,team1,team1_out,0,0,"team 1",0) 
# team1_TotalRun = sobj_1.StartGame()

# print("Batting Order for team 2:")
# team2 = list()
# team2_out = list()
# for i in range(noOfPlayers):
# 	name = input()
# 	x = BatsmanStats(name)
# 	team2.append(x)

# sobj_2 = PlayInnings(noOfOvers,team2,team2_out,0,0,"team 2",0) 
# team2_TotalRun = sobj_2.StartGame()

# if team1_TotalRun>team2_TotalRun:
# 	print("Team 1 Wins")
# elif team1_TotalRun<team2_TotalRun:
# 	print("Team 2 Wins")
# else:
# 	print("It's a Draw")


class BatsmanStats:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.fours = 0
        self.sixes = 0
        self.balls_faced = 0


class ScoreDisplay:
    def print_scorecard(self, team, team_out, total_runs, total_wickets, total_overs):
        print(f"Scorecard for Team {team.team_name}:")
        print("Player Name Score 4s 6s Balls")
        for member in team_out:
            print(f"{member.name} {member.score} {member.fours} {member.sixes} {member.balls_faced}")
        for member in team.players:
            print(f"{member.name} {member.score} {member.fours} {member.sixes} {member.balls_faced}")
        print(f"Total: {total_runs}/{total_wickets}")
        overs = total_overs // 6
        left_over = total_overs % 6
        print(f"Overs: {overs}.{left_over}")


class CricketGame:
    def __init__(self, innings):
        self.innings = innings

    def play_game(self):
        total_runs, total_wickets, total_overs = self.innings.play()
        return total_runs, total_wickets, total_overs


class CricketInnings:
    def __init__(self, team, no_of_overs):
        self.team = team
        self.no_of_overs = no_of_overs
        self.total_runs = 0
        self.total_wickets = 0
        self.total_overs = 0
        self.team_out = []

    def play(self):
        striker, non_striker = self.team.pop(0), self.team.pop(0)
        for i in range(self.no_of_overs):
            if len(self.team) < 2:
                break
            print(f"Over {i + 1}:")
            striker, non_striker = self.play_current_over(striker, non_striker)
        return self.total_runs, self.total_wickets, self.total_overs

    def play_current_over(self, striker, non_striker):
        ball_count = 6
        while ball_count > 0:
            ball = input()
            if ball == 'Wd':
                self.total_runs += 1
            elif ball == 'W':
                self.total_wickets += 1
                self.team_out.append(striker)
                if self.team:
                    striker = self.team.pop(0)
            else:
                ball = int(ball)
                self.total_runs += ball
                striker.score += ball
                striker.balls_faced += 1
                if ball == 4:
                    striker.fours += 1
                elif ball == 6:
                    striker.sixes += 1
            ball_count -= 1
            self.total_overs += 1
            striker, non_striker = non_striker, striker
        return striker, non_striker


# Main program
no_of_players = int(input("No. of players for each team: "))
no_of_overs = int(input("No. of overs: "))

print("Batting Order for team 1:")
team1 = [BatsmanStats(input()) for _ in range(no_of_players)]
innings1 = CricketInnings(team1, no_of_overs)

game = CricketGame(innings1)
total_runs, total_wickets, total_overs = game.play_game()

score_display = ScoreDisplay()
score_display.print_scorecard(innings1.team, innings1.team_out, total_runs, total_wickets, total_overs)
