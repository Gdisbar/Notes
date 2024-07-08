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
Total: 12/1
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
        self.bowler = None  # No direct equivalent in Python translation
        self.run = run

class Over:
    def __init__(self):
        self.balls = []

class Player:
    def __init__(self, name):
        self.name = name
        self.six = 0
        self.four = 0
        self.wide = 0
        self.total_balls = 0
        self.score = 0
        self.is_out = False
        self.team_id = 0

    def update_score(self, run):
        if run == 4:
            self.four += 1
        elif run == 6:
            self.six += 1

        self.score += run
        self.total_balls += 1

    def set_out(self):
        self.is_out = True
        self.total_balls += 1


class ScoreCard:
    def __init__(self):
        self.player_details = []
        self.total = ""
        self.overs = ""


class Team:
    def __init__(self, team_id):
        self.id = team_id
        self.first_player = None
        self.second_player = None
        self.current_striker = None
        self.players_map = {}
        self.wide_balls = 0

    def add_player(self, name):
        if name in self.players_map:
            return
        self.players_map[name] = Player(name)

        if self.first_player is None:
            self.first_player = name
            self.current_striker = self.first_player
        elif self.second_player is None:
            self.second_player = name

    def play_ball(self, ball):
        if self.first_player is None or self.second_player is None:
            return False

        if ball.run == "Wd":
            self.wide_balls += 1
            return True
        if ball.run == "W":
            self.players_map[self.current_striker].set_out()
            return self.find_next_player(self.current_striker)
        
        run_int = int(ball.run)
        self.players_map[self.current_striker].update_score(run_int)
        if run_int % 2 != 0:
            self.strike_change()
        return True

    def total_overs(self):
        total_balls = sum(player.total_balls for player in self.players_map.values())
        complete_overs = total_balls // 6
        partial_overs = total_balls % 6
        if partial_overs > 0:
            return f"{complete_overs}.{partial_overs}"
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
        self.team_one = None
        self.team_two = None

    def add_team(self, team_id):
        if self.team_one and self.team_two:
            return False

        if (self.team_one and self.team_one.id == team_id) or (self.team_two and self.team_two.id == team_id):
            return False

        team = Team(team_id)
        if self.team_one is None:
            self.team_one = team
        elif self.team_two is None:
            self.team_two = team
        return True

    def add_player(self, player_name, team_id):
        team = self.get_team_from_id(team_id)
        if not team:
            return False
        team.add_player(player_name)
        return True

    def play_over(self, team_id, over):
        team = self.get_team_from_id(team_id)
        if not team:
            return False

        for ball in over.balls:
            team.play_ball(Ball(ball))

        if len(over.balls) >= 6:
            team.strike_change()
        return True

    def scorecard(self, team_id):
        team = self.get_team_from_id(team_id)
        if not team:
            return None

        score_card = ScoreCard()
        score_card.player_details = self.player_scorecard(team)
        score_card.overs = team.total_overs()
        score_card.total = f"{team.total_score()}/{team.total_wickets()}"
        return score_card

    def final_result(self):
        if not self.team_one or not self.team_two:
            return None

        team_one_score = self.team_one.total_score()
        team_two_score = self.team_two.total_score()

        if team_one_score > team_two_score:
            return f"Team 1 won the match by {team_one_score - team_two_score} runs"
        elif team_two_score > team_one_score:
            return f"Team 2 won the match by {team_two_score - team_one_score} runs"
        else:
            return "Match draw"

    def get_team_from_id(self, team_id):
        if self.team_one and self.team_one.id == team_id:
            return self.team_one
        if self.team_two and self.team_two.id == team_id:
            return self.team_two
        return None

    @staticmethod
    def player_scorecard(team):
        player_scorecard = ["Player_Name Score 4s 6s Balls"]

        for player in team.players_map.values():
            player_name = player.name
            if player_name == team.first_player or player_name == team.second_player:
                player_name += "*"
            score = str(player.score)
            fours = str(player.four)
            sixes = str(player.six)
            balls = str(player.total_balls)
            player_detail = f"{player_name} {score} {fours} {sixes} {balls}"
            player_scorecard.append(player_detail)

        return player_scorecard

def print_score(game, team_id):
    score_card = game.scorecard(team_id)
    if score_card:
        for player_detail in score_card.player_details:
            print(player_detail)
        print(score_card.total)
        print(score_card.overs)
    else:
        print(f"No scorecard available for Team {team_id}")

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

