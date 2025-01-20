import asyncio

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObservationType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)

from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder, DoubleBattleOrder
from poke_env.player.env_player import EnvPlayer
from poke_env.player.player import Player

class Gen9VGCEnvDoublePlayer(EnvPlayer):
    _ACTION_SPACE = list(range(784))  # 28 actions for each Pokémon → 28 * 28 = 784
    _DEFAULT_BATTLE_FORMAT = "gen9vgc2024"

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """
        Converts actions to move orders for Double Battle with Terastallization.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 784:
            Action is split into two parts:
                - first_action: action // 28 → Action for Pokémon 1
                - second_action: action % 28 → Action for Pokémon 2

            Pokémon Action Breakdown (0-27):
                0  <= action < 12:
                    Normal move (0-3) targeting (Opponent 1, Opponent 2, Ally).
                12 <= action < 24:
                    Terastallized move (0-3) targeting (Opponent 1, Opponent 2, Ally).
                24 <= action < 28:
                    Switch action (0-3).

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: The order to send to the server.
        :rtype: BattleOrder
        """
        if action == -1:
            return ForfeitBattleOrder()

        first_action = action // 28
        second_action = action % 28

        def map_action(pokemon_action, active_pokemon):
            if pokemon_action < 12:  # Normal Move
                move_id = pokemon_action // 3
                target_id = pokemon_action % 3
                if move_id < len(active_pokemon.available_moves):
                    return Player.create_order(active_pokemon.available_moves[move_id], target=battle.get_target(target_id))
                
            elif 12 <= pokemon_action < 24:  # Terastallized Move
                move_id = (pokemon_action - 12) // 3
                target_id = (pokemon_action - 12) % 3
                if active_pokemon.can_terastallize and move_id < len(active_pokemon.available_moves):
                    return Player.create_order(active_pokemon.available_moves[move_id], terastallize=True, target=battle.get_target(target_id))
                
            elif 24 <= pokemon_action < 28:  # Switch
                switch_id = pokemon_action - 24
                if switch_id < len(battle.available_switches):
                    return Player.create_order(battle.available_switches[switch_id])
            return Player.choose_random_move(battle)

        action_1 = map_action(first_action, battle.active_pokemon[0])
        action_2 = map_action(second_action, battle.active_pokemon[1])

        return DoubleBattleOrder(action_1, action_2)
    
class SimpleRLPlayer(Gen9VGCEnvDoublePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        # Moves of Pokémon 1 and Pokémon 2
        moves_p1 = -np.ones(4)
        moves_p2 = -np.ones(4)

        # Multipliers of Pokémon 1's moves against both enemies
        multipliers_p1_e1 = np.ones(4)
        multipliers_p1_e2 = np.ones(4)

        # Multipliers of Pokémon 2's moves against both enemies
        multipliers_p2_e1 = np.ones(4)
        multipliers_p2_e2 = np.ones(4)

        # Get available moves for Pokémon 1
        for i, move in enumerate(battle.available_moves):
            moves_p1[i] = move.base_power / 100
            if move.type:
                multipliers_p1_e1[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                multipliers_p1_e2[i] = move.type.damage_multiplier(
                    battle.opponent_team.values()[1].type_1,
                    battle.opponent_team.values()[1].type_2,
                )

        # Get available moves for Pokémon 2
        for i, move in enumerate(battle.available_moves[4:]):
            moves_p2[i] = move.base_power / 100
            if move.type:
                multipliers_p2_e1[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                multipliers_p2_e2[i] = move.type.damage_multiplier(
                    battle.opponent_team.values()[1].type_1,
                    battle.opponent_team.values()[1].type_2,
                )

        # Weather condition
        weather = 0
        if battle.weather:
            weather = {
                "harsh_sunlight": 1,
                "rain": 2,
                "sandstorm": 3,
                "snow": 4,
            }.get(battle.weather, 0)

        # Fainted Pokémon
        fainted_ally = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_enemy = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        # HP of all Pokémon (normalized)
        hp_status = np.array(
            [
                mon.current_hp_fraction
                for mon in battle.team.values()
            ] + [
                mon.current_hp_fraction
                for mon in battle.opponent_team.values()
            ]
        )

        # Final embedding vector
        final_vector = np.concatenate(
            [
                moves_p1,
                moves_p2,
                multipliers_p1_e1,
                multipliers_p1_e2,
                multipliers_p2_e1,
                multipliers_p2_e2,
                [weather, fainted_ally, fainted_enemy],
                hp_status,
            ]
        )
        return final_vector.astype(np.float32)

    def describe_embedding(self) -> Space:
        low = [-1] * 20 + [0] * 12 + [0, 0, 0] + [0] * 8
        high = [3] * 20 + [4] * 12 + [4, 1, 1] + [1] * 8
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 1235"  # Lock team position
    
async def main():
    # Lock agent team
    team1 = """
Archaludon @ Assault Vest  
Ability: Stamina  
Level: 50  
Tera Type: Fairy  
EVs: 220 HP / 4 Def / 52 SpA / 116 SpD / 116 Spe  
Bold Nature  
IVs: 26 Atk  
- Electro Shot  
- Draco Meteor  
- Flash Cannon  
- Body Press  

Rillaboom @ Loaded Dice  
Ability: Grassy Surge  
Level: 50  
Tera Type: Fire  
EVs: 204 HP / 116 Atk / 4 Def / 60 SpD / 124 Spe  
Adamant Nature  
- Bullet Seed  
- Grassy Glide  
- Fake Out  
- High Horsepower  

Basculegion @ Choice Band  
Ability: Swift Swim  
Level: 50  
Tera Type: Grass  
EVs: 252 Atk / 4 Def / 252 Spe  
Adamant Nature  
- Wave Crash  
- Last Respects  
- Flip Turn  
- Aqua Jet  

Kingambit @ Black Glasses  
Ability: Defiant  
Level: 50  
Tera Type: Dark  
EVs: 236 HP / 228 Atk / 4 Def / 4 SpD / 36 Spe  
Adamant Nature  
- Kowtow Cleave  
- Sucker Punch  
- Swords Dance  
- Protect  

Pelipper @ Focus Sash  
Ability: Drizzle  
Level: 50  
Tera Type: Ghost  
EVs: 4 HP / 252 SpA / 252 Spe  
Modest Nature  
- Weather Ball  
- Hurricane  
- Tailwind  
- Protect  

Electabuzz @ Eviolite  
Ability: Vital Spirit  
Level: 50  
Tera Type: Ghost  
EVs: 244 HP / 180 Def / 4 SpA / 20 SpD / 60 Spe  
Bold Nature  
IVs: 20 Atk  
- Electroweb  
- Taunt  
- Follow Me  
- Protect 
"""
    agent = SimpleRLPlayer(
        battle_format="gen9vgc2024",
        start_challenging=True,
        team=team1,
    )

    opponent = RandomPlayer(
        battle_format="gen9vgc2024",
        team=team1,
    )

    # Train, evaluate, and save the model
    model = Sequential([
        Dense(128, activation="elu", input_shape=(agent.observation_space.shape,)),
        Flatten(),
        Dense(64, activation="elu"),
        Dense(agent.action_space.n, activation="linear"),
    ])
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=agent.action_space.n,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    dqn.fit(agent, nb_steps=10000)

    # Save the model
    model.save("dqn_gen9vgc_model.h5")
    print("Model saved to dqn_gen9vgc_model.h5")

    # Evaluate and print results
    eval_env = SimpleRLPlayer(
        battle_format="gen9vgc2024", opponent=opponent, start_challenging=True
    )
    dqn.test(eval_env, nb_episodes=100, verbose=False)
    print(f"Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles}")


if __name__ == "__main__":
    asyncio.run(main())