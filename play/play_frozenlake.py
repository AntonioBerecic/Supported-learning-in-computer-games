import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import time

# --- Funkcija za pronalazenje najnovijeg direktorija za logove ---
def get_latest_log_dir(base_dir="./"):
    """
    Pronalazi najnoviji direktorij za logove na temelju vremenske oznake.
    """
    list_of_dirs = glob.glob(os.path.join(base_dir, "logs/frozenlake_policy_gradient_logs_*"))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

# --- Funkcija za pokretanje simulacije igranja ---
def run_simulation(model, agent_name, render_mode="human", num_episodes=1, record_video=False):
    """
    Pokrece vizualnu simulaciju agenta koji igra u FrozenLake-v1 okruzenju, s opcijom snimanja videa.
    """
    print(f"\n--- Pokrecem {agent_name} simulaciju (Epizode: {num_episodes}) ---")
    env = None 
    wrapped_env = None
    current_env = None

    try:
        # Odredi stvarni nacin prikaza za stvaranje okruzenja
        actual_render_mode = "rgb_array" if record_video else render_mode

        # Stvori FrozenLake-v1 okruzenje, osiguravajuci dosljednost s treniranjem
        env = gym.make("FrozenLake-v1", render_mode=actual_render_mode, is_slippery=True)

        if record_video:
            video_folder = os.path.join("videos", f"frozenlake_{agent_name}_{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(video_folder, exist_ok=True)
            wrapped_env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, disable_logger=True)
            print(f"Video snimanje omoguceno. Snimam u: {video_folder}")
            current_env = wrapped_env 
        else:
            current_env = env #

        for i in range(num_episodes):
            obs, info = current_env.reset() 
            total_reward = 0 
            num_steps = 0
            done = False 
            truncated = False 

            while not done and not truncated:
                # Osiguraj da je obs skalarni cijeli broj za diskretne prostore promatranja poput FrozenLakea
                # Ovo eksplicitno obradjuje slucajeve gdje env.reset() ili env.step() mogu vratiti np.array([X])
                if isinstance(obs, np.ndarray) and obs.size == 1:
                    obs = obs.item() 
                elif isinstance(obs, np.ndarray) and obs.shape == (): 
                    obs = obs.item() 

                action, _states = model.predict(obs, deterministic=True)

                # Osiguraj da je 'action' skalarni cijeli broj za diskretne prostore akcija (poput FrozenLakea)
                if isinstance(action, np.ndarray) and action.size == 1:
                    action = action.item()

                obs, reward, done, truncated, info = current_env.step(action) 
                total_reward += reward 
                num_steps += 1 

            print(f"   Epizoda {i+1} ({agent_name}): Nagrada: {total_reward:.2f}, Koraci: {num_steps}")
            if truncated:
                print("   Epizoda prekinuta zbog maksimalnog broja koraka.")

            if i < num_episodes - 1:
                print("   Cekam 3 sekunde prije sljedece simulacije...")
                time.sleep(3)

    except Exception as e:
        print(f"Greska tijekom {agent_name} simulacije: {e}")
    finally:
        if wrapped_env:
            wrapped_env.close()
        if env:
            env.close() 
        print(f"--- {agent_name} simulacija zavrsena. ---")

# --- Glavna funkcija za izvrsavanje skripte ---
def main():
    LATEST_LOG_DIR = get_latest_log_dir()
    if LATEST_LOG_DIR is None:
        print("Nema pronadjenih direktorija s logovima. Molimo prvo pokrenite train_frozenlake.py.")
        return

    print(f"Ucitavam modele iz najnovijeg direktorija: {LATEST_LOG_DIR}")

    # Ucitavanje modela
    model_a2c_loaded = None
    model_ppo_loaded = None

    try:
        model_a2c_loaded = A2C.load(os.path.join(LATEST_LOG_DIR, "a2c_frozenlake.zip"))
        print("A2C model ucitan uspjesno.")
        print(f"A2C model's expected observation space: {model_a2c_loaded.observation_space}")
        print(f"A2C model's expected action space: {model_a2c_loaded.action_space}")
    except Exception as e:
        print(f"Greska pri ucitavanju A2C modela: {e}. A2C model nece biti dostupan.")

    try:
        model_ppo_loaded = PPO.load(os.path.join(LATEST_LOG_DIR, "ppo_frozenlake.zip"))
        print("PPO model ucitan uspjesno.")
        print(f"PPO model's expected observation space: {model_ppo_loaded.observation_space}")
        print(f"PPO model's expected action space: {model_ppo_loaded.action_space}")
    except Exception as e:
        print(f"Greska pri ucitavanju PPO modela: {e}. PPO model nece biti dostupan.")

    if model_a2c_loaded is None and model_ppo_loaded is None:
        print("Nije pronadjen nijedan validan model za ucitavanje. Izlazim.")
        return

    # --- Kvantitativna evaluacija ucitanih modela ---
    print("\n--- Finalna evaluacija ucitanih modela za FrozenLake-v1 ---")
    if model_a2c_loaded:
        eval_env_a2c = Monitor(gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array"))
        # Evaluiraj politiku A2C modela na 100 epizoda
        mean_reward_a2c, std_reward_a2c = evaluate_policy(model_a2c_loaded, eval_env_a2c, n_eval_episodes=100, render=False)
        print(f"A2C FrozenLake: Prosjecna nagrada kroz 100 epizoda: {mean_reward_a2c:.2f} +/- {std_reward_a2c:.2f}")
        eval_env_a2c.close()
    else: 
        mean_reward_a2c = -np.inf

    if model_ppo_loaded:
        eval_env_ppo = Monitor(gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array"))
        # Evaluiraj politiku PPO modela na 100 epizoda
        mean_reward_ppo, std_reward_ppo = evaluate_policy(model_ppo_loaded, eval_env_ppo, n_eval_episodes=100, render=False)
        print(f"PPO FrozenLake: Prosjecna nagrada kroz 100 epizoda: {mean_reward_ppo:.2f} +/- {std_reward_ppo:.2f}")
        eval_env_ppo.close()
    else: #
        mean_reward_ppo = -np.inf

    # --- Prikaz grafikona ucenja (ponovno crtanje) ---
    performance_data_path = os.path.join(LATEST_LOG_DIR, 'performance_data.json')
    if os.path.exists(performance_data_path):
        with open(performance_data_path, 'r') as f:
            performance_data_loaded = json.load(f)

        plt.figure(figsize=(12, 6)) 
        plt.title("Performanse ucenja na FrozenLake-v1 (Policy Gradient Algoritmi)") 
        plt.xlabel("Broj koraka (Timesteps)") 
        plt.ylabel("Prosjecna nagrada") 

        colors = {'A2C': 'blue', 'PPO': 'green'} 
        labels = {'A2C': 'A2C', 'PPO': 'PPO'} 

        for algo, data in performance_data_loaded.items():
            timesteps = data['timesteps']
            mean_reward = data['mean_reward']
            std_reward = data['std_reward']

            plt.plot(timesteps, mean_reward, label=labels[algo], color=colors[algo]) 
            plt.fill_between(timesteps, np.array(mean_reward) - np.array(std_reward),
                             np.array(mean_reward) + np.array(std_reward),
                             color=colors[algo], alpha=0.2) 

        plt.axhline(y=0.7, color='r', linestyle='--', label='Prag uspjesnosti (0.7)') 

        plt.legend() 
        plt.grid(True) 
        plt.ylim(-0.05, 1.05) 
        plt.tight_layout() 
        plt.show()
    else:
        print(f"Datoteka {performance_data_path} nije pronadjena. Ne mogu prikazati grafikon ucenja.")

    # --- Prikaz simulacije igranja ---
    print("\n--- Pokretanje simulacija igranja ---")

    # Simulacija za A2C agenta
    if model_a2c_loaded:
        input_a2c = input("Zelite li pokrenuti vizualnu simulaciju za A2C agenta? (da/ne): ").lower()
        if input_a2c == 'da':
            record_a2c = input("Zelite li snimiti video A2C simulacije? (da/ne): ").lower() == 'da'
            run_simulation(model_a2c_loaded, "A2C", num_episodes=2, record_video=record_a2c)
            print("\n")
            time.sleep(2)
        else:
            print("A2C simulacija preskocena.")
    else:
        print("A2C model nije dostupan za vizualizaciju.")

    # Simulacija za PPO agenta
    if model_ppo_loaded:
        input_ppo = input("Zelite li pokrenuti vizualnu simulaciju za PPO agenta? (da/ne): ").lower()
        if input_ppo == 'da':
            record_ppo = input("Zelite li snimiti video PPO simulacije? (da/ne): ").lower() == 'da'
            run_simulation(model_ppo_loaded, "PPO", num_episodes=2, record_video=record_ppo)
        else:
            print("PPO simulacija preskocena.")
    else:
        print("PPO model nije dostupan za vizualizaciju.")

    print("\n--- Sve simulacije igranja zavrsene. ---")

if __name__ == "__main__":
    main()