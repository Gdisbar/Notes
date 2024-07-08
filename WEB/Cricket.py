"""

Problem:
Design a cricket scorecard that will show the score for a team along with score of each player.

You will be given the number of players in each team, the number of overs and their batting order as input. Then, we can input overs ball by ball with the runs scored on that ball (could be wide, no ball or a wicket as well).
You are expected to print individual scores, number of balls faced, number of 4s, number of 6s for all the players from the batting side at the end of every over. You also need to print total score, total wickets. Essentially, you need to keep a track of all the players, strike changes (at the end of the over or after taking singles or 3s) and maintain their scores, also keep track of extra bowls that are being bowled (like wides or no balls). You also need to print which team won the match at the end.
This is the bare minimum solution which is expected for the problem. You can add some more features once you are done with these, like maintaining a bowlers record (total overs bowled, runs conceded, wickets taken, maiden overs, dot balls, economy, etc.). Total team extras, batsman strike rates, etc. can be added too. But these are "good to have" features, please try to complete the bare minimum first.

Make sure your code is readable and maintainable and preferably object oriented. It should be modular and extensible, to add new features if needed.

Sample input and output:
No. of players for each team: 5
No. of overs: 2
Batting Order for team 1:
P1
P2
P3
P4
P5
Over 1:
1
1
1
1
1
2

Scorecard for Team 1:
Player Name Score 4s 6s Balls
P1* 3 0 0 3
P2* 4 0 0 3
P3 0 0 0 0
P4 0 0 0 0
P5 0 0 0 0
Total: 7/0
Overs: 1

Over 2:
W
4
4
Wd
W
1
6

Scorecard for Team 1:
Player Name Score 4s 6s Balls
P1 3 0 0 4
P2* 10 0 1 4
P3 8 2 0 3
P4* 1 0 0 1
P5 0 0 0 0
Total: 23/2
Overs: 2

Batting Order for team 2:
P6
P7
P8
P9
P10

Over 1:
4
6
W
W
1
1

Scorecard for Team 2:
Player Name Score 4s 6s Balls
P6 10 1 1 3
P7* 1 0 0 1
P8 0 0 0 1
P9* 1 0 0 1
P10 0 0 0 0
Total: 12/2
Overs: 1

Over 2:
6
1
W
W

Scorecard for Team 2:
Player Name Score 4s 6s Balls
P6 10 1 1 2
P7* 8 0 1 3
P8 0 0 0 1
P9 1 0 0 2
P10 0 0 0 1
Total: 19/4
Overs: 1.4

Result: Team 1 won the match by 4 runs

"""

class Ball:
    def __init__(self, run):
        self.bowler = None  # Bowler information (not used in this implementation)
        self.run = run      # Run scored on the ball (can be 'Wd', 'W', '1', '2', '3', '4', '6')

class Over:
    def __init__(self):
        self.balls = []     # List to store balls in the over

class Player:
    def __init__(self, name):
        self.name = name    # Player's name
        self.six = 0        # Number of sixes hit
        self.four = 0       # Number of fours hit
        self.wide = 0       # Number of wides faced
        self.total_balls = 0    # Total balls faced
        self.score = 0      # Total score
        self.is_out = False     # Player out status
        self.team_id = 0    # Team ID the player belongs to

    def update_score(self, run):
        if run == 4:
            self.four += 1   # Increment fours if run is 4
        elif run == 6:
            self.six += 1    # Increment sixes if run is 6

        self.score += run    # Add run to player's score
        self.total_balls += 1    # Increment total balls faced

    def set_out(self):
        self.is_out = True   # Set player as out
        self.total_balls += 1    # Increment total balls faced

class ScoreCard:
    def __init__(self):
        self.player_details = []    # List to store player-specific score details
        self.total = ""             # Total score and wickets information
        self.overs = ""             # Total overs bowled information

class Team:
    def __init__(self, team_id):
        self.id = team_id           # Team ID
        self.first_player = None    # First player in batting order
        self.second_player = None   # Second player in batting order
        self.current_striker = None # Current striker (either first or second player)
        self.players_map = {}       # Dictionary to map player name to Player objects
        self.wide_balls = 0         # Count of wides bowled

    def add_player(self, name):
        if name in self.players_map:
            return  # Player already exists in team, no need to add again

        self.players_map[name] = Player(name)   # Add player to team's player map

        if self.first_player is None:
            self.first_player = name    # Set as first player if no first player exists
            self.current_striker = self.first_player

        elif self.second_player is None:
            self.second_player = name   # Set as second player if no second player exists

    def play_ball(self, ball):
        if self.first_player is None or self.second_player is None:
            return False    # If batting order not complete, return False

        if ball.run == "Wd":
            self.wide_balls += 1    # Increment wide balls count

        elif ball.run == "W":
            self.players_map[self.current_striker].set_out()   # Set current striker out
            return self.find_next_player(self.current_striker)    # Find next available player

        else:
            run_int = int(ball.run)
            self.players_map[self.current_striker].update_score(run_int)  # Update striker's score

            if run_int % 2 != 0:
                self.strike_change()    # Change strike if odd runs scored

        return True

    def total_overs(self):
        total_balls = sum(player.total_balls for player in self.players_map.values())
        complete_overs = total_balls // 6    # Complete overs bowled
        partial_overs = total_balls % 6      # Remaining balls (partial over)
        
        if partial_overs > 0:
            return f"{complete_overs}.{partial_overs}"    # Return formatted total overs
        return str(complete_overs)

    def total_score(self):
        return sum(player.score for player in self.players_map.values()) + self.wide_balls

    def total_wickets(self):
        return sum(1 for player in self.players_map.values() if player.is_out)

    def strike_change(self):
        if self.current_striker == self.first_player:
            self.current_striker = self.second_player
        else:
            self.current_striker = self.first_player

    def find_next_player(self, current_out):
        for player in self.players_map.values():
            if not player.is_out and player.name != self.first_player and player.name != self.second_player:
                if current_out == self.first_player:
                    self.first_player = player.name
                else:
                    self.second_player = player.name

                self.current_striker = player.name
                return True

        if current_out == self.first_player:
            self.first_player = None
        else:
            self.second_player = None

        return False

class Game:
    def __init__(self):
        self.team_one = None    # Initialize team one
        self.team_two = None    # Initialize team two

    def add_team(self, team_id):
        if self.team_one and self.team_two:
            return False    # Return false if both teams are already added

        if (self.team_one and self.team_one.id == team_id) or (self.team_two and self.team_two.id == team_id):
            return False    # Return false if team ID already exists

        team = Team(team_id)    # Create new team object

        if self.team_one is None:
            self.team_one = team    # Assign as team one if team one is not set

        elif self.team_two is None:
            self.team_two = team    # Assign as team two if team two is not set

        return True

    def add_player(self, player_name, team_id):
        team = self.get_team_from_id(team_id)

        if not team:
            return False    # Return false if team not found

        team.add_player(player_name)    # Add player to the team
        return True

    def play_over(self, team_id, over):
        team = self.get_team_from_id(team_id)

        if not team:
            return False    # Return false if team not found

        for ball in over.balls:
            team.play_ball(Ball(ball))  # Play each ball in the over

        if len(over.balls) >= 6:
            team.strike_change()    # Change strike after completing an over

        return True

    def scorecard(self, team_id):
        team = self.get_team_from_id(team_id)

        if not team:
            return None    # Return None if team not found

        score_card = ScoreCard()    # Create new scorecard object
        score_card.player_details = self.player_scorecard(team)    # Get player score details
        score_card.overs = team.total_overs()    # Get total overs bowled
        score_card.total = f"{team.total_score()}/{team.total_wickets()}"    # Get total score and wickets

        return score_card

    def final_result(self):
        if not self.team_one or not self.team_two:
            return None    # Return None if either team not found

        team_one_score = self.team_one.total_score()    # Get total score of team one
        team_two_score = self.team_two.total_score()    # Get total score of team two

        if team_one_score > team_two_score:
            return f"Team 1 won the match by {team_one_score - team_two_score} runs"    # Team one wins

        elif team_two_score > team_one_score:
            return f"Team 2 won the match by {team_two_score - team_one_score} runs"    # Team two wins

        else:
            return "Match draw"    # Match is draw

    def get_team_from_id(self, team_id):
        if self.team_one and self.team_one.id == team_id:
            return self.team_one    # Return team one if ID matches

        if self.team_two and self.team_two.id == team_id:
            return self.team_two    # Return team two if ID matches

        return None    # Return None if team not found

    @staticmethod
    def player_scorecard(team):
        player_scorecard = ["Player_Name Score 4s 6s Balls"]    # Header for player scorecard

        for player in team.players_map.values():
            player_name = player.name

            if player_name == team.first_player or player_name == team.second_player:
                player_name += "*"    # Add '*' if player is current striker

            score = str(player.score)
            fours = str(player.four)
            sixes = str(player.six)
            balls = str(player.total_balls)

            player_detail = f"{player_name} {score} {fours} {sixes} {balls}"
            player_scorecard.append(player_detail)    # Add player details to scorecard list

        return player_scorecard

def print_score(game, team_id):
    score_card = game.scorecard(team_id)

    if score_card:
        for player_detail in score_card.player_details:
            print(player_detail)    # Print each player detail in scorecard

        print(score_card.total)    # Print total score and wickets
        print(score_card.overs)    # Print total overs

    else:
        print(f"No scorecard available for Team {team_id}")    # Print if scorecard not available


if __name__ == "__main__":
    game = Game()

    game.add_team(1)
    game.add_player("P1", 1)
    game.add_player("P2", 1)
    game.add_player("P3", 1)
    game.add_player("P4", 1)
    game.add_player("P5", 1)

    over = Over()
    over.balls.extend(["1", "1", "1", "1", "1", "2"])
    game.play_over(1, over)

    print_score(game, 1)

    over = Over()
    over.balls.extend(["W", "4", "4", "Wd", "W", "1", "6"])
    game.play_over(1, over)

    print_score(game, 1)

    game.add_team(2)
    game.add_player("P6", 2)
    game.add_player("P7", 2)
    game.add_player("P8", 2)
    game.add_player("P9", 2)
    game.add_player("P10", 2)

    over = Over()
    over.balls.extend(["4", "6", "W", "W", "1", "1"])
    game.play_over(2, over)

    print_score(game, 2)

    over = Over()
    over.balls.extend(["6", "1", "W", "W"])
    game.play_over(2, over)

    print_score(game, 2)
    print(game.final_result())

def print_score(game, team_id):
    score_card = game.scorecard(team_id)
    for player_detail in score_card.player_details:
        print(player_detail)
    print(score_card.total)
    print(score_card.overs)

