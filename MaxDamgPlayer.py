from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
import random
from typing import List
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.target import Target
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)


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