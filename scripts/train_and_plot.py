"""
Entrena un agente (qlearning o dqn) sobre MountainCar-v0, guarda el historial
de recompensas en un CSV y genera una gráfica de la curva de entrenamiento.

Uso:
    uv run python scripts/train_and_plot.py qlearning --episodes 20000
    uv run python scripts/train_and_plot.py dqn --episodes 2500

Requiere matplotlib (no es dependencia del paquete, instálalo si hace falta):
    uv pip install matplotlib
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def moving_average(values: list[float], window: int) -> np.ndarray:
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def evaluate(agent, env_id: str, episodes: int = 100) -> tuple[float, int, float]:
    import gymnasium as gym

    env = gym.make(env_id)
    rewards, reached = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total, done = 0.0, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
            if terminated:
                reached += 1
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), reached, float(np.max(rewards))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["qlearning", "dqn"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--window", type=int, default=100, help="Ventana de la media móvil")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    env_id = "MountainCar-v0"

    if args.agent == "qlearning":
        from mountain_car.agents.qlearning import QLearningAgent

        episodes = args.episodes or 20_000
        agent = QLearningAgent(env_id, n_bins=20, lr=0.1, gamma=0.99, epsilon_decay=0.9995)
    else:
        from mountain_car.agents.dqn import DQNAgent

        episodes = args.episodes or 2500
        agent = DQNAgent(env_id, epsilon_decay=0.995, sticky_prob=0.9)

    print(f"Entrenando {args.agent} por {episodes} episodios...")
    history = agent.train(total_episodes=episodes, log_interval=args.log_interval)

    csv_path = RESULTS_DIR / f"{args.agent}_rewards.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(history, start=1):
            writer.writerow([i, r])
    print(f"Historial guardado en {csv_path}")

    mean_r, reached, best_r = evaluate(agent, env_id, episodes=100)
    print(f"Evaluación (100 episodios, política greedy):")
    print(f"  Recompensa media : {mean_r:.2f}")
    print(f"  Mejor episodio   : {best_r:.2f}")
    print(f"  Llegó a la meta  : {reached}/100")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        window = args.window
        avg = moving_average(history, window)

        plt.figure(figsize=(9, 5))
        plt.plot(history, alpha=0.25, color="steelblue", label="Recompensa por episodio")
        if len(avg) > 0:
            plt.plot(
                range(window - 1, window - 1 + len(avg)),
                avg,
                color="darkorange",
                linewidth=2,
                label=f"Media móvil ({window} episodios)",
            )
        plt.axhline(-110, color="green", linestyle="--", linewidth=1, label='Umbral "resuelto" (-110)')
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        title = "Q-Learning tabular" if args.agent == "qlearning" else "DQN"
        plt.title(f"{title} — MountainCar-v0 ({episodes} episodios)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        png_path = RESULTS_DIR / f"{args.agent}_training_curve.png"
        plt.savefig(png_path, dpi=150)
        print(f"Gráfica guardada en {png_path}")
    except ImportError:
        print("matplotlib no está instalado; se omitió la gráfica (el CSV sí se guardó).")


if __name__ == "__main__":
    main()
