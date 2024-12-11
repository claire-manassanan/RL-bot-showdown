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

team_file = open('team.txt')
team_player = team_file.read()
all_team = team_player.split("***split***")

team_usa = all_team[0]
team_thai = all_team[1]
class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.packed_teams = []

        for team in teams:
            parsed_team = self.parse_showdown_team(team)
            packed_team = self.join_team(parsed_team)
            self.packed_teams.append(packed_team)

    def yield_team(self):
        return np.random.choice(self.packed_teams)
    

# teams = [team_usa, team_thai]

# team = RandomTeamFromPool(teams)
# print(sys.path)

server = ServerConfiguration(
    authentication_url="http://localhost:8000/showdown/action.php",
    websocket_url="ws://localhost:8000/showdown/websocket"
)

p1 = RandomPlayer(
    battle_format='gen9vgc2024regh',
    server_configuration=server,
    team=team_thai
)
p2 = RandomPlayer(
    battle_format='gen9vgc2024regh',
    server_configuration=server,
    team=team_usa
)


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # Chooses a move with the highest base power when possible
        if battle.available_moves:
            # Iterating over available moves to find the one with the highest base power
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            # Creating an order for the selected move
            return self.create_order(best_move)
        else:
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)
        
class CustomPlayer(Player):
    def choose_move(self, battle):
        # Log some basic battle info
        print(f"Turn: {battle.turn}")
        # print(f"My active Pokémon: {battle.active_pokemon}")
        # print(f"Opponent's active Pokémon: {battle.opponent_active_pokemon}")

        if battle.available_moves:
            # Log available moves
            print(f"Available moves: {[move.id for move in battle.available_moves]}")
            
            # Example strategy: Use the first available move
            return self.create_order(battle.available_moves[0])
        
        # Check available switches
        if battle.available_switches:
            print(f"Available switches: {[poke.species for poke in battle.available_switches]}")
            
            # Example strategy: Switch to the first available Pokémon
            return self.create_order(battle.available_switches[0])
        
        # Default to a random move
        return self.choose_random_move()

# maxx = MaxDamagePlayer(team=team_usa)
# custom = CustomPlayer(
#     battle_format='gen9vgc2024regh',
#     team=team_usa
# )

async def vgc2024(p1,p2):
    await p1.battle_against(p2, n_battles=1)

asyncio.run(vgc2024(p1,p2))
# async def battle_basic(maxx,p1):
#     await p1.battle_against(maxx, n_battles=1)
team_file.close()
# asyncio.run(battle_basic(p2,p1))

# print(f'max won {maxx.n_won_battles} times')
# print(f'p1 won {p1.n_won_battles} times')
