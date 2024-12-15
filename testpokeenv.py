import sys
import asyncio
import numpy as np
# sys.path.append('C:\\Users/DSCISTUDENT6\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages')

from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env import cross_evaluate
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.teambuilder import Teambuilder

team1_file = open('team1.txt')
team2_file = open('team2.txt')
team_1 = team1_file.read()
team_2 = team2_file.read()

team1_file.close()
team2_file.close()

class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.packed_teams = []

        for team in teams:
            parsed_team = self.parse_showdown_team(team)
            packed_team = self.join_team(parsed_team)
            self.packed_teams.append(packed_team)

    def yield_team(self):
        return np.random.choice(self.packed_teams)


server = ServerConfiguration(
    authentication_url="http://localhost:8000/showdown/action.php",
    websocket_url="ws://localhost:8000/showdown/websocket"
)

format = 'gen9randombattle'
p1 = RandomPlayer(
    battle_format=format,
    server_configuration=server
    # team=team_1
)
p2 = RandomPlayer(
    battle_format=format,
    server_configuration=server
    # team=team_2
)

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            print(f"Turn: {battle.turn}")
            # Iterating over available moves to find the one with the highest base power
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            print(best_move,end='\n')
            # Creating an order for the selected move
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)
        
maxx = MaxDamagePlayer(
    battle_format=format
    # team=team_2
)
# custom = CustomPlayer(
#     battle_format='gen9vgc2024regh',
#     team=team_usa
# )

async def vgc2024(p1,p2):
    await p1.battle_against(p2, n_battles=1)

asyncio.run(vgc2024(p1,maxx))

# print(f'max won {maxx.n_won_battles} times')
# print(f'p1 won {p1.n_won_battles} times')
