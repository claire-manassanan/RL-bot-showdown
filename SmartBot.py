team = """
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
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.weather import Weather
from poke_env.player import Player
import random
from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from typing import List
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
) 
class SmartBot(Player):
    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 1235"  # Archaludon, Rillaboom, Basculegion, Pelipper

    def choose_move(self, battle: DoubleBattle):
        orders: List[BattleOrder] = []
        switched_in = None

        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        for idx, (mon, moves, switches) in enumerate(
            zip(battle.active_pokemon, battle.available_moves, battle.available_switches)
        ):
            if not mon or mon.fainted:
                orders.append(DefaultBattleOrder())
                continue

            # บังคับการกระทำของ Rillaboom (ตัวที่ 2) ในเทิร์นแรก
            if idx == 1 and battle.turn == 1:
                bullet_seed = next((move for move in moves if move.id == "bulletseed"), None)
                if bullet_seed:
                    # -2 หมายถึงตำแหน่งของ Archaludon (ฝ่ายเรา)
                    orders.append(BattleOrder(bullet_seed, move_target=-1))
                    continue

            # กระบวนการปกติสำหรับโปเกมอนอื่นๆ
            best_move = None
            best_damage = 0
            best_target = None

            for move in moves:
                possible_targets = battle.get_possible_showdown_targets(move, mon)
                opponent_targets = [
                    t for t in possible_targets if t > 0
                ]  # กรองเป้าหมายฝ่ายตรงข้าม

                for target in opponent_targets:
                    target_pokemon = battle.opponent_active_pokemon[target - 1]
                    if target_pokemon and not target_pokemon.fainted:
                        effectiveness = target_pokemon.damage_multiplier(move)
                        damage = move.base_power * effectiveness
                        if damage > best_damage:
                            best_damage = damage
                            best_move = move
                            best_target = target

            if best_move and best_target is not None:
                orders.append(BattleOrder(best_move, move_target=best_target))
            else:
                orders.append(DefaultBattleOrder())

        return DoubleBattleOrder(orders[0], orders[1])