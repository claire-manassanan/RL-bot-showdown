import sys
import asyncio
import numpy as np
from typing import List
import random
# sys.path.append('C:\\Users/DSCISTUDENT6\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages')

from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env import cross_evaluate
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.teambuilder import Teambuilder
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.target import Target
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)



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
format = 'gen9vgc2024regh'

p1 = RandomPlayer(
    battle_format=format,
    server_configuration=server,
    team=team_1
)
p2 = RandomPlayer(
    battle_format=format,
    server_configuration=server,
    team=team_2
)

class MaxDamagePlayerrr(RandomPlayer):
    def choose_move(self, battle):
        if battle.available_moves:
            print(f"Turn: {battle.turn}")
            # for i in battle.available_moves:
            #     print(f"Move : {i} base : {i.base_power}")
            best_move = max(battle.available_moves, key=lambda move: move.base_power)

            if battle.can_tera:
                return self.create_order(best_move, terastallize=True)
            # print(f"create order : {self.create_order(best_move)}") # this is from __str__ of BattleOrder
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)
        
class MaxDamagePlayer(Player):
    def choose_move(self, battle: DoubleBattle):
        orders: List[BattleOrder] = []
        switched_in = None

        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        can_target_first_opponent = (
            battle.opponent_active_pokemon[0]
            and not battle.opponent_active_pokemon[0].fainted
        )
        can_target_second_opponent = (
            battle.opponent_active_pokemon[1]
            and not battle.opponent_active_pokemon[1].fainted
        )
        can_double_target = can_target_first_opponent and can_target_second_opponent

        for mon, moves, switches in zip(
            battle.active_pokemon, battle.available_moves, battle.available_switches
        ):
            switches = [s for s in switches if s != switched_in]

            if not mon or mon.fainted:
                orders.append(DefaultBattleOrder())
                continue
            elif not moves and switches:
                mon_to_switch_in = random.choice(switches)
                orders.append(BattleOrder(mon_to_switch_in))
                switched_in = mon_to_switch_in
                continue
            elif not moves:
                orders.append(DefaultBattleOrder())
                continue

            def move_power_with_double_target(move):
                if move.target in {Target.NORMAL, Target.ANY} or not can_double_target:
                    return move.base_power
                return move.base_power * 1.5

            best_move = max(moves, key=move_power_with_double_target)

            # randomly picks between the two opponents for normal move targeting
            targets = battle.get_possible_showdown_targets(best_move, mon)
            opp_targets = [
                t
                for t in targets
                if t in {battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION}
            ]
            if opp_targets:
                target = random.choice(opp_targets)
            else:
                target = random.choice(targets)

            orders.append(BattleOrder(best_move, move_target=target))

        if orders[0] or orders[1]:
            return DoubleBattleOrder(orders[0], orders[1])

        return self.choose_random_move(battle)

class DoublesPlayer(Player):
    def choose_move(self, battle: Battle):
        """
        Custom move logic for doubles battles.
        Issues two actions: one for each Pokémon on your side.
        """
        # Initialize the actions list
        actions = []
        
        # Iterate through your active Pokémon
        for i, pokemon in enumerate(battle.active_pokemon_list):
            if pokemon is None:  # Skip if this slot has no Pokémon (e.g., fainted)
                continue

            # If there are moves available for this Pokémon
            if pokemon.available_moves:
                # Select the first move as a placeholder strategy
                move = pokemon.available_moves[0]
                print(f"Active Pokémon {i+1}: {pokemon.species} will use {move.id}")
                actions.append(self.create_order(move))
            elif battle.available_switches:
                # Fallback: switch to the first available Pokémon
                switch = battle.available_switches[0]
                print(f"Active Pokémon {i+1}: {pokemon.species} will switch to {switch.species}")
                actions.append(self.create_order(switch))
        
        # Return the two chosen actions
        return self.format_move_order(actions)

maxx = MaxDamagePlayer(
    server_configuration=server,
    battle_format=format,
    team=team_2
)
async def vgc2024(p1,p2):
    await p1.battle_against(p2, n_battles=500)

asyncio.run(vgc2024(p1,maxx))
# print(team_1)

print(f'dou won {maxx.n_won_battles} times')
print(f'p1 won {p1.n_won_battles} times')
