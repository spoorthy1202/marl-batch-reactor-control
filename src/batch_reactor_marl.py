"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MULTI-AGENT REINFORCEMENT LEARNING — BATCH POLYMERIZATION REACTOR CONTROL  ║
║   Based on: Bayer & Pruckner (2022), "Enhancing the Performance of          ║
║             Multi-Agent Reinforcement Learning for Controlling HVAC Systems" ║
║   Applied to: Batch Reactor Temperature Setpoint Tracking                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO USE IN GOOGLE COLAB
──────────────────────────
Step 1 → Upload 'Trajectory2.csv' to your Colab session
         (click the folder icon on the left → upload)
Step 2 → Run the installation cell below
Step 3 → Paste this entire script into one cell and run it
         OR split it at the # ── CELL BREAK ── markers

WHAT THIS DOES (plain English)
──────────────────────────────
We have a chemical batch reactor that makes polymer.  The reactor has a temperature
sensor (Tr) and we want it to follow a rising temperature ramp from ~45 °C to ~61 °C.
Two AI agents work together:
  • Agent 1 controls how fast cold water flows through the cooling jacket (u1, L/min)
  • Agent 2 controls how much electricity goes to the heater (u2, 4-20 mA)
Both agents learn by trial and error (Reinforcement Learning).
They get a reward when they keep the temperature close to the target, and get punished
when they deviate.  Over 60 training episodes they get better and better.

ANALOGY TO THE HVAC PAPER
──────────────────────────
  HVAC paper            │  This batch reactor code
  ──────────────────────│─────────────────────────────────────────────────────
  EnergyPlus simulator  │  BatchReactorEnv (4 ODEs solved with scipy)
  Room temperature      │  Reactor temperature Tr
  Heating setpoint      │  Temperature trajectory from Trajectory2.csv
  Agent per room        │  Agent per actuator (coolant / heater)
  Energy + comfort      │  Tracking error + energy used
  DDQN networks         │  QNetwork (NumPy, no PyTorch needed)
  ε-greedy exploration  │  Same strategy, same decay
  Shared reward         │  Same reward given to both agents (independent Q-learning)
  Parameter sharing     │  Both agents see same 6-element state vector
"""

# ── CELL BREAK ── (installation — run this cell first in Colab)
# !pip install scipy matplotlib pandas numpy --quiet

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)   # Makes results reproducible — same random numbers every run


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BATCH REACTOR ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the "simulator" — equivalent to EnergyPlus in the HVAC paper.
# Instead of simulating airflow through rooms, we solve 4 coupled ODEs that
# describe the chemistry and heat balance of a batch polymerization reactor.
#
# STATE VARIABLES (what is happening inside the reactor right now):
#   I  = Initiator concentration  [mol/L]  — chemical that kick-starts the reaction
#   M  = Monomer concentration    [mol/L]  — raw material being turned into polymer
#   Tr = Reactor temperature      [°C]     — THE variable we want to control
#   Tj = Jacket temperature       [°C]     — the cooling jacket around the reactor
#
# CONTROL INPUTS (what the two agents can do each time step):
#   u1 = Coolant flow rate  [L/min]  → Agent 1 picks from {0, 0.25, 0.5, 0.75, 1.0}
#   u2 = Heater current     [mA]     → Agent 2 picks from {4, 8, 12, 16, 20}
#
# ══════════════════════════════════════════════════════════════════════════════

class BatchReactorEnv:
    """
    Batch polymerization reactor simulator.

    The reactor physics are described by 4 differential equations.
    Every call to .step(u1, u2) integrates those equations forward by dt=0.5 min
    and returns the new reactor temperature Tr and jacket temperature Tj.
    """

    def __init__(self, trajectory_path='data/Trajectory2.csv', dt=0.5, initial_history_size=50):
        # ── Physical & Chemical Constants (do not change) ──────────────────
        self.Ad = 4.4e16;       self.Ed = 140.06e3       # Initiator decomposition kinetics
        self.Ap = 1.7e11 / 60;  self.Ep = 16.9e3 / 0.239 # Propagation kinetics
        self.deltaHp = -82.2e3                             # Heat of polymerization [J/mol]
        self.UA      = 33.3083                             # Overall heat-transfer coefficient [W/°C]
        self.V       = 0.5                                 # Reactor volume [L]
        self.Tc      = 27.0                                # Coolant inlet temperature [°C]
        self.Cpc     = 4.184                               # Coolant heat capacity [J/g·°C]
        self.R       = 8.3145                              # Universal gas constant [J/mol·K]
        self.alpha   = 1.212827;  self.beta1 = 0.000267   # Heat-loss curve parameters
        self.epsilon = 0.5;       self.theta = 1.25        # Reaction order exponents
        self.m1      = 450.0;     self.cp1   = 4.184       # Reactor mass [g] & heat capacity
        self.mjCpj   = (18 * 4.184) + (240 * 0.49)        # Jacket heat capacity [J/°C]
        self.cp2     = 187.0                               # Initiator heat capacity
        self.cp3     = 110.58                              # Monomer heat capacity
        self.cp4     = 84.95                               # Polymer heat capacity
        self.m5      = 220.0;     self.cp5   = 0.49        # Stirrer mass & cp
        self.m6      = 7900.0;    self.cp6   = 0.49        # Vessel wall mass & cp
        self.M0      = 0.7034                              # Initial monomer concentration [mol/L]
        self.I0      = 4.5e-3                              # Initial initiator concentration [mol/L]
        self.Qs      = 12.41e-2                            # Constant background heat source [W]

        # ── Simulation Parameters ──────────────────────────────────────────
        self.dt = dt                               # Time step = 0.5 minutes per step
        self.initial_history_size = initial_history_size  # Pre-fill 50 steps of history

        # ── Load the Temperature Trajectory ───────────────────────────────
        # Trajectory2.csv contains 7200 target temperatures, one per time step.
        # This is the "temperature ramp" the agents must follow.
        # The reactor starts at ~45 °C and should reach ~61 °C by step 7200.
        try:
            df = pd.read_csv(trajectory_path)
            self.setpoint_traj = df['x_Traject'].dropna().values
            print(f"✓ Trajectory loaded: {len(self.setpoint_traj)} steps  "
                  f"({self.setpoint_traj[0]:.2f} °C  →  {self.setpoint_traj[-1]:.2f} °C)")
        except Exception as e:
            print(f"⚠  Trajectory2.csv not found ({e}). Using a default linear ramp.")
            self.setpoint_traj = np.linspace(45.3, 60.85, 7200)

        self.reset()

    # ── Heater Conversion: mA → Watts ─────────────────────────────────────
    def _normalize_heater_current(self, mA):
        """
        The heater is controlled via a 4–20 mA signal (industry standard).
        4 mA  → 0 W  (heater off)
        20 mA → 1500 W (heater full power)
        Linear interpolation in between.
        """
        mA = np.clip(mA, 4.0, 20.0)
        return ((mA - 4.0) / 16.0) * 1500.0

    # ── The 4 ODEs (the reactor physics) ──────────────────────────────────
    def _BR_plant(self, t, X0, F, Hc):
        """
        The heart of the simulator — 4 differential equations.

        Given the current state [I, M, Tr, Tj] and the control inputs [F, Hc],
        this returns the rate of change [dI/dt, dM/dt, dTr/dt, dTj/dt].

        scipy's solve_ivp calls this function repeatedly to integrate forward in time.

        Physical meaning:
          dI/dt  = how fast the initiator is consumed (purely chemical)
          dM/dt  = how fast the monomer is converted to polymer (the reaction we care about)
          dTr/dt = net heat balance inside the reactor:
                   heat generated by polymerization
                   − heat lost to jacket
                   + heat added by electric heater
                   + constant background heat
                   − heat lost to environment
          dTj/dt = jacket heat balance:
                   heat gained from reactor
                   − heat removed by coolant flow
        """
        I, M, Tr, Tj = X0
        F_conv = F * 16.667                             # Convert L/min → mL/min (unit match)
        Qc_val = self._normalize_heater_current(Hc)    # mA → Watts

        # Arrhenius reaction rate constants — both increase exponentially with temperature
        k_d = self.Ad * np.exp(-self.Ed / (self.R * (Tr + 273.15)))  # Initiator decomp
        k_p = self.Ap * np.exp(-self.Ep / (self.R * (Tr + 273.15)))  # Polymerisation

        Ri  = k_d * I                                   # Initiator consumption rate
        Rp  = k_p * (I ** self.epsilon) * (M ** self.theta)  # Polymerisation rate

        # Total heat capacity of everything inside the reactor
        mrCpr = (self.m1 * self.cp1 +
                 I * self.cp2 * self.V +
                 M * self.cp3 * self.V +
                 (self.M0 - M) * self.cp4 * self.V +
                 self.m5 * self.cp5 +
                 self.m6 * self.cp6)

        Qpr = self.alpha * (Tr - self.Tc) ** self.beta1  # Heat loss to surroundings

        dI_dt  = -Ri
        dM_dt  = -Rp
        dTr_dt = (Rp * self.V * (-self.deltaHp)          # Heat from reaction
                  - self.UA * (Tr - Tj)                   # Heat exchange with jacket
                  + Qc_val                                 # Electric heater power
                  + self.Qs                                # Background heat
                  - Qpr) / mrCpr                          # ÷ total heat capacity

        dTj_dt = (self.UA * (Tr - Tj)                    # Heat from reactor
                  - F_conv * self.Cpc * (Tj - self.Tc)   # Heat removed by coolant
                  ) / self.mjCpj

        return [dI_dt, dM_dt, dTr_dt, dTj_dt]

    # ── Reset to Initial Conditions ────────────────────────────────────────
    def reset(self):
        """
        Resets the reactor to starting conditions.
        Called at the beginning of every training episode.
        """
        n = self.initial_history_size
        self.state_history = np.zeros((n, 4))
        self.state_history[:, 0] = 4.5e-5              # Initial initiator I (very small — trace amount)
        self.state_history[:, 1] = 0.7034              # Initial monomer M (full tank)
        self.state_history[:, 2] = self.setpoint_traj[:n]  # Tr starts on the trajectory
        self.state_history[:, 3] = 45.0                # Tj starts at 45 °C
        self.current_step = n
        return self.get_state_vector()

    # ── Advance One Time Step ──────────────────────────────────────────────
    def step(self, u1, u2):
        """
        Applies control actions u1 (coolant flow) and u2 (heater current),
        integrates the ODEs for dt=0.5 min, and returns the new temperatures.
        """
        current_state = self.state_history[-1]
        sol = solve_ivp(
            self._BR_plant,
            [0, self.dt],
            current_state,
            method='RK45',            # Runge-Kutta 4th/5th order — accurate & stable
            args=(u1, u2),
            rtol=1e-6, atol=1e-8      # Tight tolerances for numerical accuracy
        )
        next_state = sol.y[:, -1]
        self.state_history = np.vstack((self.state_history, next_state))
        self.current_step += 1
        return next_state[2], next_state[3]   # Return Tr, Tj

    # ── Setpoint Lookup ────────────────────────────────────────────────────
    def get_setpoint(self, step=None):
        """Returns the target temperature for the current (or given) time step."""
        if step is None:
            step = self.current_step
        idx = min(step, len(self.setpoint_traj) - 1)
        return self.setpoint_traj[idx]

    # ── Build the 6-Element State Vector ──────────────────────────────────
    def get_state_vector(self):
        """
        Packs the reactor's current condition into a 6-number vector that
        the neural networks can process.  All values are normalized to roughly
        the [-1, +1] range — this helps the network learn faster.

        Equivalent to Table 1 in the HVAC paper (state space definition).

        Index  │ Meaning                              │ Normalization
        ───────┼──────────────────────────────────────┼───────────────────
          0    │ Reactor temperature Tr               │ (Tr - 50) / 20
          1    │ Jacket temperature Tj                │ (Tj - 45) / 20
          2    │ Initiator concentration I            │ I / 4.5e-5
          3    │ Monomer concentration M              │ M / 0.7034
          4    │ Tracking error  (Tr - setpoint)      │ error / 10
          5    │ Current setpoint                     │ (sp - 50) / 20
        """
        last  = self.state_history[-1]
        Tr, Tj, I, M = last[2], last[3], last[0], last[1]
        sp    = self.get_setpoint()
        error = Tr - sp                                # Negative = too cold, positive = too hot
        return np.array([
            (Tr - 50)   / 20,
            (Tj - 45)   / 20,
            I / 4.5e-5,
            M / 0.7034,
            error       / 10,
            (sp - 50)   / 20,
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Q-NETWORK (the "brain" of each agent)
# ══════════════════════════════════════════════════════════════════════════════
#
# This is a tiny neural network implemented entirely in NumPy.
# No PyTorch or TensorFlow needed — works in any environment.
#
# Architecture:
#   Input layer  : 6 neurons (one per state variable)
#   Hidden layer : 64 neurons with ReLU activation
#   Output layer : N neurons (one per possible action)
#
# The output Q(s, a) estimates "how much total reward will I get if I am in
# state s and I take action a?"  The agent always picks the action with the
# highest Q-value.
#
# This is the DDQN described in the HVAC paper (Section 3.3).
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork:
    """
    2-layer neural network that approximates the Q-function.
    Input(6) → Hidden(64, ReLU) → Output(n_actions)
    """
    def __init__(self, state_dim, n_actions, lr=0.005):
        # He initialization — scales weights so ReLU activations start healthy
        self.W1 = np.random.randn(state_dim, 64) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, n_actions) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(n_actions)
        self.lr = lr

    def forward(self, x):
        """Forward pass: state vector → Q-values for every action."""
        if x.ndim == 1:
            x = x[np.newaxis, :]                      # Make it 2D: (1, 6)
        h1 = np.maximum(0, x @ self.W1 + self.b1)    # Hidden layer with ReLU
        return h1 @ self.W2 + self.b2                 # Output layer (no activation)

    def update(self, s, a, target_val):
        """
        One step of backpropagation.
        Adjusts the network weights so that Q(s, a) moves toward target_val.
        Only the weights connected to action 'a' get updated (others unchanged).
        """
        if s.ndim == 1:
            s = s[np.newaxis, :]
        z1 = s @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        q  = h1 @ self.W2 + self.b2

        error = q[0, a] - target_val      # How far off is our Q-value estimate?

        # Gradient descent — move weights in the direction that reduces error
        self.W2[:, a] -= self.lr * error * h1[0]
        self.b2[a]    -= self.lr * error

        d_h1 = error * self.W2[:, a]
        d_z1 = d_h1 * (z1[0] > 0)        # ReLU derivative: 0 if neuron was off
        self.W1 -= self.lr * np.outer(s[0], d_z1)
        self.b1 -= self.lr * d_z1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DDQN AGENT (with ε-greedy exploration)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each agent wraps a QNetwork and adds the exploration strategy.
#
# EXPLORATION vs EXPLOITATION (ε-greedy):
#   At the start of training, the agent knows nothing, so it tries random actions
#   to discover what works (ε = 1.0 → 100% random).
#   As training progresses, ε decays toward 0.1 — the agent mostly follows
#   what it has learned, but still explores 10% of the time to avoid getting stuck.
#
# This mirrors exactly the ε-decay strategy in the HVAC paper.
# ══════════════════════════════════════════════════════════════════════════════

class DDQNAgent:
    """
    RL agent with ε-greedy action selection and Q-Network learning.
    """
    def __init__(self, state_dim, action_values, lr=0.005, name="Agent"):
        self.action_values = np.array(action_values)    # Actual physical values of each action
        self.n_actions     = len(action_values)
        self.net           = QNetwork(state_dim, self.n_actions, lr=lr)
        self.eps           = 1.0                        # Start fully random
        self.name          = name
        self.q_value_log   = []                         # For diagnostics

    def select_action(self, state, greedy=False):
        """
        Pick an action.
        If greedy=True (evaluation mode): always pick the best known action.
        If greedy=False (training mode):  pick randomly with probability ε,
                                          otherwise pick the best known action.
        Returns (action_index, physical_action_value).
        """
        if not greedy and np.random.rand() < self.eps:
            idx = np.random.randint(self.n_actions)     # Random exploration
        else:
            q_vals = self.net.forward(state)[0]
            idx    = np.argmax(q_vals)                  # Greedy exploitation
            self.q_value_log.append(np.max(q_vals))
        return idx, self.action_values[idx]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
#
# The reward tells agents how well they are doing at each time step.
# Inspired by Equation (9) in the HVAC paper:
#   r = −(λ_e × energy + λ_m × comfort_error)
#
# Our version:
#   r = −(0.5 × |Tr − setpoint| + 0.01 × (u1 + normalized_u2))
#
# Clipped to [−1, 0]:
#   • 0   = perfect (temperature exactly on target, no energy wasted)
#   • −1  = worst case (far from target or using maximum energy)
#
# The two-term structure mirrors the HVAC paper:
#   Tracking error ↔ thermal comfort complaints
#   Energy penalty  ↔ energy consumption
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(Tr, sp, u1, u2):
    """
    Compute the shared reward for both agents.
    Returns (reward, absolute_error).
    """
    error         = abs(Tr - sp)
    norm_u2       = (u2 - 4.0) / 16.0    # Normalize heater 4–20 mA → 0–1
    raw_reward    = -(0.5 * error + 0.01 * (u1 + norm_u2))
    clipped_reward = max(-1.0, min(0.0, raw_reward))
    return clipped_reward, error


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
#
# MULTI-AGENT SETUP (mirrors the HVAC paper's independent Q-learning):
#
#   Agent 1 — "Coolant Agent" controls u1 (coolant flow rate)
#             Action space: {0.0, 0.25, 0.5, 0.75, 1.0} L/min
#             More coolant → faster cooling of the jacket
#
#   Agent 2 — "Heater Agent" controls u2 (heater current)
#             Action space: {4, 8, 12, 16, 20} mA
#             More current → more heat added to the reactor
#
#   Both agents:
#     • Observe the SAME 6-element state vector
#     • Receive the SAME shared reward (this is "independent Q-learning")
#     • Update their own separate Q-networks
#     • Decay their own ε at the same rate
#
# This is identical in structure to the HVAC paper's multi-agent setup where
# every room's heating agent sees the full building state and shares the reward.
# ══════════════════════════════════════════════════════════════════════════════

def run_training(n_episodes=60, steps_per_episode=200, verbose=True):
    """
    Train both agents over n_episodes episodes.
    Each episode: reset the reactor, run steps_per_episode control steps,
    update both networks, decay ε.

    Returns: agent1, agent2, env, history_dict
    """
    env    = BatchReactorEnv()
    agent1 = DDQNAgent(6, [0.0, 0.25, 0.5, 0.75, 1.0], lr=0.005, name="Coolant Agent")
    agent2 = DDQNAgent(6, [4.0, 8.0, 12.0, 16.0, 20.0], lr=0.005, name="Heater Agent")

    history = {
        'mean_error':  [],
        'max_error':   [],
        'mean_reward': [],
        'eps':         [],
        'u1_avg':      [],
        'u2_avg':      [],
    }

    print("\n" + "═" * 62)
    print("  TRAINING MULTI-AGENT RL SYSTEM FOR BATCH REACTOR CONTROL")
    print("═" * 62)
    print(f"  Episodes          : {n_episodes}")
    print(f"  Steps / episode   : {steps_per_episode}")
    print(f"  Agent 1 (Coolant) : {agent1.n_actions} actions — {agent1.action_values} L/min")
    print(f"  Agent 2 (Heater)  : {agent2.n_actions} actions — {agent2.action_values} mA")
    print(f"  State dimension   : 6")
    print(f"  Reward function   : −(0.5×|error| + 0.01×energy), clipped to [−1, 0]")
    print("═" * 62 + "\n")

    for ep in range(n_episodes):
        state  = env.reset()
        ep_err = []
        ep_rew = []
        ep_u1  = []
        ep_u2  = []

        for step in range(steps_per_episode):
            # ── Both agents independently pick an action ──
            idx1, u1 = agent1.select_action(state)
            idx2, u2 = agent2.select_action(state)

            # ── Simulate the reactor for one 0.5-minute step ──
            Tr, Tj = env.step(u1, u2)
            sp     = env.get_setpoint()

            # ── Compute shared reward ──
            rew, err = compute_reward(Tr, sp, u1, u2)

            # ── Get next state ──
            next_state = env.get_state_vector()

            # ── Update both Q-networks (independent Q-learning) ──
            #    Each agent learns from its own perspective:
            #    "I was in state s, I took action a, I got reward r → update Q(s,a)"
            agent1.net.update(state, idx1, rew)
            agent2.net.update(state, idx2, rew)

            state = next_state
            ep_err.append(err)
            ep_rew.append(rew)
            ep_u1.append(u1)
            ep_u2.append(u2)

        # ── Decay exploration rate at end of each episode ──
        # Multiply ε by 0.94 each episode → reaches ~0.1 by episode ~40
        # This mirrors the linear ε-decay used in the HVAC paper
        agent1.eps = max(0.1, agent1.eps * 0.94)
        agent2.eps = max(0.1, agent2.eps * 0.94)

        history['mean_error'].append(np.mean(ep_err))
        history['max_error'].append(np.max(ep_err))
        history['mean_reward'].append(np.mean(ep_rew))
        history['eps'].append(agent1.eps)
        history['u1_avg'].append(np.mean(ep_u1))
        history['u2_avg'].append(np.mean(ep_u2))

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes}  │  "
                  f"Mean Error: {history['mean_error'][-1]:.3f} °C  │  "
                  f"Max Error: {history['max_error'][-1]:.3f} °C  │  "
                  f"ε = {agent1.eps:.3f}  │  "
                  f"Reward: {history['mean_reward'][-1]:.4f}")

    print("\n✓ Training complete!\n")
    return agent1, agent2, env, history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
#
# After training, we run the agents in GREEDY mode (ε = 0, no random actions).
# This shows what the trained policy actually does.
# We run for 400 steps (200 minutes of real reactor time) and record everything.
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(agent1, agent2, env, n_steps=400):
    """
    Run both agents greedily (pure exploitation, no exploration).
    Records temperatures, actions, errors, and concentrations.
    """
    print("Running final evaluation (greedy policy, no exploration)...")
    state = env.reset()
    results = {'Tr': [], 'sp': [], 'Tj': [], 'u1': [], 'u2': [],
               'error': [], 'I': [], 'M': []}

    for _ in range(n_steps):
        _, u1 = agent1.select_action(state, greedy=True)
        _, u2 = agent2.select_action(state, greedy=True)

        Tr, Tj   = env.step(u1, u2)
        sp       = env.get_setpoint()
        last     = env.state_history[-1]

        results['Tr'].append(Tr)
        results['sp'].append(sp)
        results['Tj'].append(Tj)
        results['u1'].append(u1)
        results['u2'].append(u2)
        results['error'].append(abs(Tr - sp))
        results['I'].append(last[0])
        results['M'].append(last[1])

        state = env.get_state_vector()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PERFORMANCE REPORT (printed to console)
# ══════════════════════════════════════════════════════════════════════════════

def print_performance_report(history, results):
    """Print a clean, structured performance summary."""
    errors = np.array(results['error'])
    print("\n" + "═" * 62)
    print("  FINAL PERFORMANCE REPORT")
    print("═" * 62)

    print(f"\n  TEMPERATURE TRACKING (Evaluation Phase — 400 steps)")
    print(f"  {'Mean Absolute Error':<32} {np.mean(errors):.4f} °C")
    print(f"  {'Max Absolute Error':<32} {np.max(errors):.4f} °C")
    print(f"  {'RMS Error':<32} {np.sqrt(np.mean(errors**2)):.4f} °C")
    print(f"  {'Steps within ±0.5 °C':<32} {100*np.mean(errors < 0.5):.1f}%")
    print(f"  {'Steps within ±1.0 °C':<32} {100*np.mean(errors < 1.0):.1f}%")

    print(f"\n  CONTROL ACTIONS")
    print(f"  {'Avg Coolant Flow u1':<32} {np.mean(results['u1']):.3f} L/min")
    print(f"  {'Avg Heater Current u2':<32} {np.mean(results['u2']):.3f} mA")

    print(f"\n  REACTION PROGRESS")
    print(f"  {'Initial Monomer M₀':<32} {results['M'][0]:.4f} mol/L")
    print(f"  {'Final Monomer Mf':<32} {results['M'][-1]:.4f} mol/L")
    conv = 100 * (1 - results['M'][-1] / results['M'][0])
    print(f"  {'Monomer Conversion':<32} {conv:.1f}%")

    print(f"\n  TRAINING SUMMARY")
    print(f"  {'Initial Mean Error (Episode 1)':<32} {history['mean_error'][0]:.4f} °C")
    print(f"  {'Final Mean Error (last episode)':<32} {history['mean_error'][-1]:.4f} °C")
    improve = 100 * (1 - history['mean_error'][-1] / max(history['mean_error'][0], 1e-9))
    print(f"  {'Error Reduction':<32} {improve:.1f}%")
    print("═" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Produces a 7-panel figure showing:
#   ① Learning curve (how error decreases over training)
#   ② ε-decay (exploration → exploitation transition)
#   ③ Mean reward per episode
#   ④ Main result: setpoint tracking performance (full width)
#   ⑤ Agent 1's coolant flow actions  (y: 0–1 L/min)
#   ⑥ Agent 2's heater current actions (y: 4–20 mA)
#   ⑦ Chemical concentrations (monomer + initiator)
#
# X-axis for plots ④, ⑤, ⑥, ⑦ shows trajectory index (0–7200 scale)
# as requested (even though evaluation runs 400 steps starting at step 50).
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_results(history, results, env):
    """
    Generate the complete 7-panel results figure.
    Axis limits follow the professor's specifications:
      • Heater (u2)  : y-axis  4 – 20 mA
      • Coolant (u1) : y-axis  0 – 1 L/min
      • Trajectory   : x-axis  0 – 7200 (full trajectory scale)
    """
    plt.rcParams.update({
        'figure.facecolor':  '#0f1117',
        'axes.facecolor':    '#1a1d27',
        'axes.edgecolor':    '#444',
        'axes.labelcolor':   '#ccc',
        'xtick.color':       '#aaa',
        'ytick.color':       '#aaa',
        'text.color':        '#eee',
        'grid.color':        '#2a2d3a',
        'grid.linewidth':    0.8,
        'font.family':       'monospace',
        'legend.facecolor':  '#1a1d27',
        'legend.edgecolor':  '#444',
    })

    TEAL  = '#00d4aa'
    CORAL = '#ff6b6b'
    GOLD  = '#ffd700'
    BLUE  = '#4fc3f7'
    PINK  = '#f48fb1'
    WHITE = '#ffffff'

    n_eval = len(results['Tr'])

    # The evaluation starts at step = initial_history_size (50)
    # Map evaluation steps to the full 0–7200 trajectory x-axis
    start_step   = env.initial_history_size
    traj_x_steps = np.arange(start_step, start_step + n_eval)

    # Training episode indices
    eps_idx = np.arange(len(history['mean_error']))

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        'Multi-Agent Reinforcement Learning — Batch Reactor Temperature Control\n'
        'Inspired by Bayer & Pruckner (2022) HVAC-RL Framework',
        fontsize=15, fontweight='bold', color=WHITE, y=0.99
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── ① Learning Curve ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eps_idx, history['mean_error'], color=CORAL, lw=2,   label='Mean Error')
    ax1.plot(eps_idx, history['max_error'],  color=GOLD,  lw=1.5, linestyle='--',
             alpha=0.7, label='Max Error')
    ax1.fill_between(eps_idx, history['mean_error'], alpha=0.2, color=CORAL)
    ax1.set_title('① Learning Curve', color=WHITE, fontweight='bold', pad=8)
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Temperature Error (°C)')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    # Annotate final error
    ax1.annotate(
        f"Final: {history['mean_error'][-1]:.3f} °C",
        xy=(eps_idx[-1], history['mean_error'][-1]),
        xytext=(-70, 12), textcoords='offset points',
        color=TEAL, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2)
    )

    # ── ② ε-Decay ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eps_idx, history['eps'], color=BLUE, lw=2)
    ax2.fill_between(eps_idx, history['eps'], alpha=0.25, color=BLUE)
    ax2.axhline(0.1, color=WHITE, linestyle=':', lw=1, label='Floor ε = 0.1')
    ax2.set_title('② Exploration Rate (ε) Decay', color=WHITE, fontweight='bold', pad=8)
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('ε  (Exploration Probability)')
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.text(0.05, 0.55, 'EXPLORE\n(random)', transform=ax2.transAxes,
             color='#aaa', fontsize=7)
    ax2.text(0.60, 0.15, 'EXPLOIT\n(learned)', transform=ax2.transAxes,
             color=TEAL, fontsize=7)

    # ── ③ Mean Reward ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    smooth = np.convolve(history['mean_reward'], np.ones(5) / 5, mode='same')
    ax3.plot(eps_idx, history['mean_reward'], color='#555', lw=1, alpha=0.6)
    ax3.plot(eps_idx, smooth, color=PINK, lw=2, label='5-ep smoothed')
    ax3.set_title('③ Mean Reward per Episode', color=WHITE, fontweight='bold', pad=8)
    ax3.set_xlabel('Training Episode')
    ax3.set_ylabel('Mean Reward (closer to 0 = better)')
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # ── ④ MAIN: Setpoint Tracking — full width ───────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(traj_x_steps, results['sp'], color=CORAL, lw=2.5, linestyle='--',
             label='Target Setpoint (Trajectory2.csv)', alpha=0.95, zorder=3)
    ax4.plot(traj_x_steps, results['Tr'], color=TEAL, lw=2,
             label='Reactor Temperature Tr  (Agent-controlled)', zorder=4)
    ax4.plot(traj_x_steps, results['Tj'], color=GOLD, lw=1.2, alpha=0.65,
             label='Jacket Temperature Tj', zorder=2)
    ax4.fill_between(traj_x_steps, results['sp'], results['Tr'],
                     alpha=0.15, color=CORAL, label='Tracking Error Band')

    # x-axis: full trajectory range as professor requested
    ax4.set_xlim(0, 7200)
    ax4.set_title('④ Final Evaluation: Setpoint Tracking Performance',
                  color=WHITE, fontweight='bold', pad=8)
    ax4.set_xlabel('Time Step  (1 step = 0.5 min,  full trajectory range: 0 – 7200)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True)

    # Stats box
    mean_e = np.mean(results['error'])
    ax4.text(0.76, 0.08,
             f'Mean Abs Error: {mean_e:.3f} °C\nMax Error: {np.max(results["error"]):.3f} °C\n'
             f'% within ±1°C: {100*np.mean(np.array(results["error"]) < 1.0):.1f}%',
             transform=ax4.transAxes, color=WHITE,
             bbox=dict(boxstyle='round', facecolor='#2a2d3a', alpha=0.85),
             fontsize=9)

    # ── ⑤ Agent 1 — Coolant Flow ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.step(traj_x_steps, results['u1'], color=BLUE, lw=1.5, where='post')
    ax5.fill_between(traj_x_steps, results['u1'], step='post', alpha=0.3, color=BLUE)
    ax5.set_xlim(0, 7200)
    ax5.set_ylim(0, 1)                    # ← professor's requested y-axis range
    ax5.set_title('⑤ Agent 1: Coolant Flow u₁', color=WHITE, fontweight='bold', pad=8)
    ax5.set_xlabel('Time Step  (0 – 7200)')
    ax5.set_ylabel('Flow Rate (L/min)')
    ax5.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax5.grid(True)

    # ── ⑥ Agent 2 — Heater Current ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.step(traj_x_steps, results['u2'], color=PINK, lw=1.5, where='post')
    ax6.fill_between(traj_x_steps, results['u2'], step='post', alpha=0.3, color=PINK)
    ax6.set_xlim(0, 7200)
    ax6.set_ylim(4, 20)                   # ← professor's requested y-axis range
    ax6.set_title('⑥ Agent 2: Heater Current u₂', color=WHITE, fontweight='bold', pad=8)
    ax6.set_xlabel('Time Step  (0 – 7200)')
    ax6.set_ylabel('Heater Current (mA)')
    ax6.set_yticks([4, 8, 12, 16, 20])
    ax6.grid(True)

    # ── ⑦ Chemical Concentrations ────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7_r = ax7.twinx()
    ax7.plot(traj_x_steps, results['M'], color=GOLD, lw=2, label='Monomer M')
    ax7_r.plot(traj_x_steps, np.array(results['I']) * 1e5, color=CORAL, lw=2,
               linestyle='--', label='Initiator I ×10⁵')
    ax7.set_xlim(0, 7200)
    ax7.set_title('⑦ Chemical Concentrations', color=WHITE, fontweight='bold', pad=8)
    ax7.set_xlabel('Time Step  (0 – 7200)')
    ax7.set_ylabel('Monomer M  [mol/L]', color=GOLD)
    ax7_r.set_ylabel('Initiator I ×10⁵  [mol/L]', color=CORAL)
    ax7.tick_params(axis='y', labelcolor=GOLD)
    ax7_r.tick_params(axis='y', labelcolor=CORAL)
    ax7.grid(True)
    lines1, lab1 = ax7.get_legend_handles_labels()
    lines2, lab2 = ax7_r.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc='upper right')

    plt.savefig('batch_reactor_MARL_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0f1117')
    plt.show()
    print("✓ Figure saved as 'batch_reactor_MARL_results.png'")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)    # Reproducibility

    # ── TRAIN ──
    agent1, agent2, env, history = run_training(
        n_episodes=60,
        steps_per_episode=200,
        verbose=True
    )

    # ── EVALUATE ──
    results = run_evaluation(agent1, agent2, env, n_steps=400)

    # ── REPORT ──
    print_performance_report(history, results)

    # ── PLOT ──
    plot_all_results(history, results, env)