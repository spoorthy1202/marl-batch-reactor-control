# Multi-Agent Reinforcement Learning for Batch Reactor Temperature Control

A NumPy-only implementation of independent multi-agent DDQN control for a batch
polymerization reactor, adapted from the multi-agent HVAC control framework in
Bayer & Pruckner (2022), *"Enhancing the Performance of Multi-Agent Reinforcement
Learning for Controlling HVAC Systems."*

Two agents cooperatively track a rising temperature setpoint trajectory (~45°C → ~61°C)
by controlling coolant flow and heater current, respectively.

## Problem Setup

- **Environment**: `BatchReactorEnv` — 4 coupled ODEs (initiator, monomer,
  reactor temp, jacket temp) solved with `scipy.integrate.solve_ivp`
- **Agent 1 (Coolant)**: controls coolant flow rate `u1` ∈ {0, 0.25, 0.5, 0.75, 1.0} L/min
- **Agent 2 (Heater)**: controls heater current `u2` ∈ {4, 8, 12, 16, 20} mA
- **State space**: 6-dim vector (Tr, Tj, I, M, tracking error, setpoint), normalized
- **Reward**: shared reward, `-(0.5 * |Tr - setpoint| + 0.01 * energy)`, clipped to [-1, 0]
- **Learning**: independent Q-learning, each agent has its own 2-layer NumPy Q-network,
  ε-greedy exploration decaying from 1.0 → 0.1

## Repo structure

\`\`\`
data/     Trajectory2.csv — target temperature ramp (7200 steps)
src/      batch_reactor_marl.py — full env, agents, training, eval, plotting
results/  generated output figures
\`\`\`

## Setup

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/marl-batch-reactor-control.git
cd marl-batch-reactor-control
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python src/batch_reactor_marl.py
\`\`\`

This will:
1. Train both agents for 60 episodes (200 steps/episode)
2. Run a greedy evaluation for 400 steps
3. Print a performance report to console
4. Save a 7-panel results figure to `results/batch_reactor_MARL_results.png`

### Running in Google Colab

\`\`\`python
!git clone https://github.com/YOUR_USERNAME/marl-batch-reactor-control.git
%cd marl-batch-reactor-control
!pip install -r requirements.txt -q
!python src/batch_reactor_marl.py
\`\`\`

## Reference

Bayer, D., & Pruckner, M. (2022). Enhancing the Performance of Multi-Agent
Reinforcement Learning for Controlling HVAC Systems.

## License

MIT — see [LICENSE](LICENSE).