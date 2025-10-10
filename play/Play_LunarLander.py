import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor # Za točnu evaluaciju
from gymnasium.wrappers import RecordVideo # Dodajemo RecordVideo wrapper
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import time # Za pauzu između simulacija

# --- Funkcija za pronalaženje najnovijeg direktorija za logove ---
def get_latest_log_dir(base_dir="./"):
    """
    Pronalazi najnoviji direktorij s logovima na temelju timestampa.
    """
    list_of_dirs = glob.glob(os.path.join(base_dir, "logs/lunarlander_policy_gradient_logs_*"))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

# --- Funkcija za pokretanje simulacije igranja ---
def run_simulation(model, agent_name, render_mode="human", num_episodes=1, record_video=False):
    """
    Pokreće vizualnu simulaciju igranja agenta u LunarLander-v3 okruženju, s opcijom snimanja videa.
    """
    print(f"\n--- Pokrećem {agent_name} simulaciju igranja (Epizode: {num_episodes}) ---")
    env = None
    wrapped_env = None 
    current_env = None 

    try:
        # Ako se snima video, postavljamo render_mode na "rgb_array" da RecordVideo wrapper može raditi
        actual_render_mode = "rgb_array" if record_video else render_mode
        
        env = gym.make("LunarLander-v3", render_mode=actual_render_mode)
        
        if record_video:
            video_folder = os.path.join("videos", f"lunarlander_{agent_name}_{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(video_folder, exist_ok=True)
            # Snimamo sve epizode
            wrapped_env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, disable_logger=True)
            print(f"Video snimanje omogućeno. Snimam u: {video_folder}")
            current_env = wrapped_env 
        else:
            current_env = env 

        for i in range(num_episodes):
            obs, info = current_env.reset()
            total_reward = 0
            num_steps = 0
            done = False
            truncated = False

            while not done and not truncated:
                action, _states = model.predict(obs, deterministic=True) 
                obs, reward, done, truncated, info = current_env.step(action)
                total_reward += reward
                num_steps += 1
                
                
            print(f"   Epizoda {i+1} ({agent_name}): Nagrada: {total_reward:.2f}, Koraci: {num_steps}")
            if truncated:
                print("   Epizoda je prekinuta zbog maksimalnog broja koraka.")
            
            if i < num_episodes - 1:
                print("   Čekam 3 sekunde prije sljedeće simulacije...")
                time.sleep(3) 

    except Exception as e:
        print(f"Greška tijekom {agent_name} simulacije: {e}")
    finally:
        if wrapped_env:
            wrapped_env.close()
        if env:
            env.close() # Zatvaranje baznog okruženja
        print(f"--- {agent_name} simulacija završena. ---")
        
# --- Glavna funkcija za izvršavanje skripte ---
def main():
    # Odaberite direktorij s logovima (automatski pronalazi najnoviji)
    LATEST_LOG_DIR = get_latest_log_dir()
    if LATEST_LOG_DIR is None:
        print("Nema pronađenih direktorija s logovima. Molimo prvo pokrenite train_lunarlander.py.")
        return

    print(f"Učitavam modele iz najnovijeg direktorija: {LATEST_LOG_DIR}")

    # Učitavanje modela
    model_a2c_loaded = None
    model_ppo_loaded = None

    try:
        model_a2c_loaded = A2C.load(os.path.join(LATEST_LOG_DIR, "a2c_lunarlander.zip"))
        print("A2C model učitan uspješno.")
    except Exception as e:
        print(f"Greška pri učitavanju A2C modela: {e}. A2C model neće biti dostupan.")

    try:
        model_ppo_loaded = PPO.load(os.path.join(LATEST_LOG_DIR, "ppo_lunarlander.zip"))
        print("PPO model učitan uspješno.")
    except Exception as e:
        print(f"Greška pri učitavanju PPO modela: {e}. PPO model neće biti dostupan.")

    if model_a2c_loaded is None and model_ppo_loaded is None:
        print("Nije pronađen nijedan validan model za učitavanje. Izlazim.")
        return

    # --- Kvantitativna evaluacija učitanih modela ---
    print("\n--- Finalna evaluacija učitanih modela za LunarLander-v3 ---")
    if model_a2c_loaded:
        eval_env_a2c = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
        mean_reward_a2c, std_reward_a2c = evaluate_policy(model_a2c_loaded, eval_env_a2c, n_eval_episodes=100, render=False)
        print(f"A2C LunarLander: Mean reward over 100 episodes: {mean_reward_a2c:.2f} +/- {std_reward_a2c:.2f}")
        eval_env_a2c.close()
    else:
        mean_reward_a2c = -np.inf 

    if model_ppo_loaded:
        eval_env_ppo = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
        mean_reward_ppo, std_reward_ppo = evaluate_policy(model_ppo_loaded, eval_env_ppo, n_eval_episodes=100, render=False)
        print(f"PPO LunarLander: Mean reward over 100 episodes: {mean_reward_ppo:.2f} +/- {std_reward_ppo:.2f}")
        eval_env_ppo.close()
    else:
        mean_reward_ppo = -np.inf 

    # --- Prikaz grafikona učenja (ponovno crtanje) ---
    performance_data_path = os.path.join(LATEST_LOG_DIR, 'performance_data.json')
    if os.path.exists(performance_data_path):
        with open(performance_data_path, 'r') as f:
            performance_data_loaded = json.load(f)

        plt.figure(figsize=(12, 6))
        plt.title("Performanse učenja na LunarLander-v3 (Policy Gradient Algoritmi)")
        plt.xlabel("Broj koraka (Timesteps)")
        plt.ylabel(f"Prosječna nagrada (20 epizoda)") 

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

        plt.axhline(y=200, color='r', linestyle='--', label='Riješeno (prosječna nagrada >= 200)')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Datoteka {performance_data_path} nije pronađena. Ne mogu prikazati grafikon učenja.")

    # --- Prikaz simulacije igranja ---
    print("\n--- Pokretanje simulacija igranja ---")
    
    # Simulacija za A2C agenta
    if model_a2c_loaded and mean_reward_a2c > -200: 
        input_a2c_sim = input("Želite li pokrenuti vizualnu simulaciju za A2C agenta? (da/ne): ").lower()
        if input_a2c_sim == 'da':
            record_a2c = input("Želite li snimiti video A2C simulacije? (da/ne): ").lower() == 'da'
            run_simulation(model_a2c_loaded, "A2C", num_episodes=2, record_video=record_a2c) 
            print("\n")
            time.sleep(2) 
        else:
            print("A2C simulacija preskočena.")
    else:
        print("A2C model nije dostupan ili nije dobro treniran za vizualizaciju.")

    # Simulacija za PPO agenta
    if model_ppo_loaded and mean_reward_ppo > -200: 
        input_ppo_sim = input("Želite li pokrenuti vizualnu simulaciju za PPO agenta? (da/ne): ").lower()
        if input_ppo_sim == 'da':
            record_ppo = input("Želite li snimiti video PPO simulacije? (da/ne): ").lower() == 'da'
            run_simulation(model_ppo_loaded, "PPO", num_episodes=2, record_video=record_ppo)
        else:
            print("PPO simulacija preskočena.")
    else:
        print("PPO model nije dostupan ili nije dobro treniran za vizualizaciju.")

    print("\n--- Sve simulacije igranja završene. ---")

if __name__ == "__main__":
    main()