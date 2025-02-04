import asyncio
import nest_asyncio
from stable_baselines3 import A2C
from poke_env.player import RandomPlayer
from A2C_RLbot import SimpleRLPlayer
from stable_baselines3.common.vec_env import DummyVecEnv

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
- Protect  
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

TEST_EPISODES = 100  # จำนวนเกมที่ต้องการทดสอบ

async def evaluate():
    # สร้าง Environment สำหรับ Evaluate
    random_opponent = RandomPlayer(team=team1, battle_format="gen9vgc2024regh")
    env_player = SimpleRLPlayer(battle_format="gen9vgc2024regh",
                                team=team1, opponent=random_opponent, start_challenging=True)
    
    env = DummyVecEnv([lambda: env_player])

    # ✅ โหลดโมเดลที่เทรนไว้
    model = A2C.load("a2c_gen9vgc_model", env=env)
    print("Model loaded successfully!")

    # ✅ เริ่มการประเมินผล
    env_player.reset_battles()
    obs, _ = env_player.reset()
    finished_episodes = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        try:
            obs, reward, done, _, info = env_player.step(action)
            print("Reward:", reward, "Done:", done, "Info:", info)
            if done:
                finished_episodes += 1
                if finished_episodes >= TEST_EPISODES:
                    break
        except RuntimeError:
            obs, _ = env_player.reset()

    print("Evaluation against RandomPlayer: ", env_player.n_won_battles, "wins out of", TEST_EPISODES)

if __name__ == "__main__":
    asyncio.run(evaluate())