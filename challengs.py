import asyncio
from A2C_RLbot import SimpleRLPlayer, team1
from stable_baselines3 import A2C
from poke_env import AccountConfiguration, ShowdownServerConfiguration

async def main():
    opponent = "somying"

    # โหลดโมเดล A2C
    model = A2C.load("RL_gen9vgcRH_v5")

    # ตั้งค่าบัญชีสำหรับล็อกอินเข้า Showdown
    account = AccountConfiguration("somsak", "somsak1")

    # ตั้งค่าเซิร์ฟเวอร์หลักของ Showdown
    server = ShowdownServerConfiguration

    # สร้างบอท
    player = SimpleRLPlayer(
        account_configuration=account,  # ✅ ใส่บัญชีล็อกอิน
        server_configuration=server,  # ✅ ใช้เซิร์ฟเวอร์หลัก
        battle_format="gen9vgc2025regg",
        team=team1,
        opponent=opponent
    )

    # ส่งคำท้าทายไปยังคู่ต่อสู้
    await player.send_challenges(opponent, n_challenges=1)

    # เริ่มเกม
    obs, _ = player.reset()
    done = False
    while not done:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = player.step(action)
            await asyncio.sleep(1)
        except RuntimeError:
            print("RuntimeError detected, resetting battle...")
            break

    # แสดงเรตติ้งของเกม
    for battle in player.battles.values():
        print(f"Your rating: {battle.rating}, Opponent rating: {battle.opponent_rating}")

if __name__ == "__main__":
    asyncio.run(main())
