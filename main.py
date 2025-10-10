import subprocess

def run_script(path):
    subprocess.run(["python", path], check=True)

def menu():
    print("Choose what to run:")
    print("1. Train CartPole (A2C/PPO)")
    print("2. Train CartPole (DQN)")
    print("3. Play CartPole DQN")
    print("4. Play CartPole (A2C/PPO)")
    print("5. Train FrozenLake (A2C/PPO)")
    print("6. Train FrozenLake (DQN)")
    print("7. Play FrozenLake DQN")
    print("8. Play FrozenLake (A2C/PPO)")
    print("9. Train LunarLander (A2C/PPO)")
    print("10. Train LunarLander (DQN)")
    print("11. Play LunarLander DQN")
    print("12. Play LunarLander (A2C/PPO)")
    print("0. Exit")

    choice = input("Enter choice: ").strip()
    return choice

actions = {
    "1": lambda: run_script("train/train_cartpole.py"),
    "2": lambda: run_script("train/dqn_train_cartpole.py"),
    "3": lambda: run_script("play/dqn_play_cartpole.py"),
    "4": lambda: run_script("play/play_cartpole.py"),
    "5": lambda: run_script("train/train_frozenlake.py"),
    "6": lambda: run_script("train/dqn_train_frozenlake.py"),
    "7": lambda: run_script("play/dqn_play_frozenlake.py"),
    "8": lambda: run_script("play/play_frozenlake.py"),
    "9": lambda: run_script("train/Train_LunarLander.py"),
    "10": lambda: run_script("train/dqn_train_moonlanding.py"),
    "11": lambda: run_script("play/dqn_play_moonlanding.py"),
    "12": lambda: run_script("play/Play_LunarLander.py"),
}

if __name__ == "__main__":
    while True:
        c = menu()
        if c == "0":
            print("Exiting.")
            break
        action = actions.get(c)
        if action:
            action()
        else:
            print("Invalid choice.")
