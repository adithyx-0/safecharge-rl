<img src="media/040e85f86d52cf9c1f349fefff376b1f65568493.png" title="Logo" style="width:2.70833in;height:0.70833in" alt="Institution logo" />

**Term Project**

Project Documentation: RL Formulation, Solution Design and Evaluation Plan

**SAFECHARGE-RL: REINFORCEMENT LEARNING FOR EV SMART CHARGING**

**Team 32**

**Adithyan R \| AM.SC.U4AIE23065**

**Adithya Rajesh \| AM.SC.U4AIE23007**

**Balu M Krishna \| AM.SC.U4AIE23026**

**Guide**

**Dr. Geetha M**

Department of Computer Science & Engineering (AI)

**VIII. Initial RL Formulation**

This section turns the high-level constrained Markov decision process (CMDP) from Section II into a concrete mathematical model. It defines the states, actions, rewards, cost signals, network structure and training goals. Any specific numerical parameters that rely on dataset tuning are finalized after the first test runs.

**A. Setting and Notation**

Each episode is modelled as a single day, divided into T = 96 time steps of 15 minutes (Δ = 0.25 h). The charging station has N ports, each delivering up to P<sub>port</sub> kW, with an overall grid cap of P<sub>grid</sub> kW that the combined power draw cannot exceed.

When vehicle i arrives at t<sub>i</sub><sup>arr</sup>, it requests e<sub>i</sub> kWh of energy and reports an expected departure time t<sub>i</sub><sup>rep</sup>. However, its actual departure time is t<sub>i</sub><sup>dep</sup>, which might differ from what the driver reported. Regardless of what was reported, the vehicle unplugs and leaves at t<sub>i</sub><sup>dep</sup>.

If p<sub>i</sub>(t) represents the power delivered to vehicle i during step t, the remaining energy requirement updates as:

r<sub>i</sub>(t+1) = r<sub>i</sub>(t) − p<sub>i</sub>(t)·Δ, r<sub>i</sub>(t<sub>i</sub><sup>arr</sup>) = e<sub>i</sub>, 0 ≤ p<sub>i</sub>(t) ≤ min(P<sub>port</sub>, r<sub>i</sub>(t)/Δ).

Any energy left unserved when the vehicle departs is recorded as m<sub>i</sub> = max(0, r<sub>i</sub>(t<sub>i</sub><sup>dep</sup>)). The policy never gets to see the true departure time t<sub>i</sub><sup>dep</sup>.

**B. State Representation**

The state vector combines individual port details with global station data. Inactive or empty ports are zero-padded and masked out.

- **Per-port features (i):** an occupancy flag o<sub>i</sub>(t); the remaining energy needed r<sub>i</sub>(t); the time remaining until reported departure τ<sub>i</sub>(t) = max(0, t<sub>i</sub><sup>rep</sup> − t)·Δ; and laxity ℓ<sub>i</sub>(t) = τ<sub>i</sub>(t) − r<sub>i</sub>(t)/P<sub>port</sub>. Laxity indicates how much leeway the vehicle has if charged at maximum power; a negative value means the vehicle cannot hit its target by the reported deadline even at full rate.

- **Global features:** the current electricity price c(t), time of day encoded as sin(2πt/T) and cos(2πt/T), and the previous step's total power usage ratio P(t−1)/P<sub>grid</sub>.

This gives 4N + 4 input features in total. Neither the true departure time nor a driver-honesty flag is included in the state space.

**C. Action Space and Safety Projection**

At each step, the agent outputs a continuous action vector u(t) ∈ \[0,1\]<sup>N</sup> representing raw charging requests for each port. Let h<sub>i</sub>(t) = o<sub>i</sub>(t)·min(P<sub>port</sub>, r<sub>i</sub>(t)/Δ) be the maximum power port i can physically draw. The action is first scaled to a target power vector:

p̃<sub>i</sub>(t) = u<sub>i</sub>(t)·h<sub>i</sub>(t),

and then projected into the feasible grid space:

p<sub>i</sub>(t) = clip( p̃<sub>i</sub>(t) − μ, 0, h<sub>i</sub>(t) ), where μ ≥ 0 is the smallest scalar satisfying Σ<sub>i</sub> p<sub>i</sub>(t) ≤ P<sub>grid</sub>.

If the raw request already satisfies the grid limit, μ = 0 and the actions remain untouched. If it exceeds the limit, μ is found by a fast bisection search, usually taking around twenty iterations. This operation performs an exact Euclidean projection of p̃ onto the feasible set {0 ≤ p ≤ h, Σp ≤ P<sub>grid</sub>}. Because this safety layer lives inside the environment step, grid constraints are guaranteed to hold on every step without modifying the core RL algorithm. The environment then feeds the actual applied power p(t) back to the agent in the next observation.

**D. Reward and Cost Structure**

- **Primary reward:** the negative electricity cost for the step, R<sub>t</sub> = −c(t)·Σ<sub>i</sub> p<sub>i</sub>(t)·Δ, normalized by a constant factor to keep values near unit scale.

- **Cost 1 (unmet-energy penalty):** c<sup>(1)</sup><sub>t</sub> = (Σ m<sub>i</sub>) / e<sub>ref</sub>, evaluated whenever vehicles depart at step t and scaled by a reference energy value.

- **Cost 2 (battery-wear proxy):** c<sup>(2)</sup><sub>t</sub> = (1/N)·Σ<sub>i</sub> 1\[ p<sub>i</sub>(t) \> θ·P<sub>port</sub> \] with θ = 0.8. This tracks the proportion of ports running at high power output. Since ACN-Data lacks internal battery diagnostics, this serves as a baseline proxy for stress.

- **Grid limits and fairness:** grid caps are enforced directly via projection rather than cost terms. Fairness metrics are logged for analysis rather than optimized directly in the loss function.

Including Cost 1 is essential — without a penalty for unserved energy, the policy would simply minimize cost by turning off all chargers.

**E. Policy Architecture**

The policy π<sub>θ</sub>(u \| s) is stochastic and built on a Deep-Sets architecture, allowing the network to handle varying numbers of active vehicles without needing retraining. Each port feature vector is processed by a shared feature-extractor network. The outputs from active ports are averaged to form a global station context vector. A second shared network then takes each port's features, the global context, and the station-wide features to output the mean and standard deviation for u<sub>i</sub>. Because parameters are shared across ports, the model remains invariant to port numbering order. The reward critic and cost critics share a similar structure using pooled representations. The full execution loop follows s<sub>t</sub> → u<sub>t</sub> ~ π<sub>θ</sub> → p<sub>t</sub> = Π(u<sub>t</sub>), where Π represents the safety projection.

Mean-pooling summarizes overall station demand but cannot capture fine-grained priority differences between specific vehicles; replacing it with cross-port self-attention, so the model can compare vehicle laxity directly, is a possible later extension.

**F. Optimization Objective**

Using discount factor γ = 0.99 and cost thresholds d<sub>1</sub>, d<sub>2</sub>, the training problem is:

max<sub>θ</sub> J<sub>R</sub>(θ) = E\[ Σ<sub>t=0</sub><sup>T−1</sup> γ<sup>t</sup> R<sub>t</sub> \] s.t. J<sub>C,k</sub>(θ) = E\[ Σ<sub>t</sub> γ<sup>t</sup> c<sup>(k)</sup><sub>t</sub> \] ≤ d<sub>k</sub>, k = 1, 2,

with Σ<sub>i</sub> p<sub>i</sub>(t) ≤ P<sub>grid</sub> guaranteed by the projection layer. This is optimized using PPO-Lagrangian (and SAC-Lagrangian if training time permits), updating the Lagrange multipliers λ<sub>k</sub> via dual ascent after each policy update:

L(θ, λ) = J<sub>R</sub>(θ) − Σ<sub>k</sub> λ<sub>k</sub>·( J<sub>C,k</sub>(θ) − d<sub>k</sub> ), λ<sub>k</sub> ← max(0, λ<sub>k</sub> + α<sub>λ</sub>·( Ĵ<sub>C,k</sub> − d<sub>k</sub> )).

The unmet-energy budget d<sub>1</sub> is set to 2% of total requested energy, while d<sub>2</sub> is established from baseline wear values observed in the initial rule-based runs.

**G. Key Modeling Assumptions**

- **Partial observability:** hidden true departure times make this a partially observable CMDP. The starting point is a memoryless policy treating observations as states, with recurrent policies planned as a potential upgrade.

- **Price visibility:** electricity tariffs follow a known time-of-use schedule. System uncertainty primarily stems from dynamic vehicle arrivals, unannounced departures and varying energy demands rather than price fluctuation.

- **Simplifications:** for this stage, uniform port hardware, no charging conversion losses or tapering curves, a single-point grid limit, and no local solar or vehicle-to-grid capability are assumed.

**IX. High-Level Solution Diagram**

*\[Figure 1 — SafeCharge-RL high-level system architecture, to be inserted here.\]*

Real ACN-Data sessions (1) pass through the report model (2), which turns each vehicle's true departure into the time the policy is shown — truthful or misreported. The environment (3) tracks port state, the live price and the shared grid limit. The RL agent (4) proposes a charging action for each port, and the safety projection (5) enforces the grid limit as a hard constraint before the power is applied. The loop closes as the applied power steps the environment forward, which returns the next state and reward to the agent.

**X. Dataset and Simulation Plan**

**A. Data**

We use ACN-Data (Lee, Li & Low, 2019), accessed through its public API with a registered token. The Caltech site is used for training, validation and testing, and the JPL site as a second, cross-site test set. The fields we use are listed below.

| **ACN-Data field** | **Used as**                                | **Note**                                                                                                                           |
|--------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| connectionTime     | Arrival t<sub>i</sub><sup>arr</sup>        | Rounded to the 15-minute grid.                                                                                                     |
| disconnectTime     | True departure t<sub>i</sub><sup>dep</sup> | The vehicle always leaves here.                                                                                                    |
| kWhDelivered       | Requested energy e<sub>i</sub>             | Used as the energy the vehicle needed. Driver-entered energy is not used, so that departure reports are the only unreliable input. |
| siteID             | Site split                                 | Caltech versus JPL.                                                                                                                |

*Dataset and registration: ev.caltech.edu/dataset (ev.caltech.edu/register for the API token). Python client and ACN-Sim documentation: acnportal.readthedocs.io. Fallback simulator (EV2Gym): github.com/StavrosOrf/EV2Gym.*

*The userInputs.requestedDeparture field, which would give a real driver-reported departure time, was checked directly against the API: of a 500-session sample from Caltech, only 0.2% had it populated. This is too sparse to use, so it is not in the table above, and the report model (Section X.D) is built entirely from synthetic reports rather than from recorded ones.*

**B. Preprocessing and splits**

- Sessions with zero energy or an invalid time range are dropped. If e<sub>i</sub> \> P<sub>port</sub>·(t<sub>i</sub><sup>dep</sup> − t<sub>i</sub><sup>arr</sup>), it is clipped so that every request is feasible on its own.

- P<sub>port</sub> and N are taken from the site itself rather than estimated: ACN-Sim ships the exact EVSE list for each site (acnportal.acnsim.network.sites.caltech<sub>a</sub>cn / jpl<sub>a</sub>cn). Every port at both sites is rated 32 A at 208 V, giving P<sub>port</sub> = 6.656 kW. Caltech has N = 54 ports and JPL has N = 52 — the real physical count, not an estimate from occupancy, so a test day can never exceed it.

- Only weekdays are used in the main experiments. The split is chronological within one stable period, for example a pre-2020 year with the first part for training, then validation, then test, so that the COVID-era change reported by Li & Sun does not mix with the misreporting effect. The exact dates are fixed after checking session counts. A post-shift test set is kept as an extra robustness check.

**C. Tariff and grid limit**

- **Tariff.** A named, real tariff: SCE TOU-EV-4, effective March 2019, shipped with ACN-Sim (acnportal.signals.tariffs) and linked to Southern California Edison's published rate sheet. Summer weekdays have five periods — off-peak \$0.05623/kWh (00:00–08:00, 23:00–24:00), mid-peak \$0.0925/kWh (08:00–12:00, 18:00–23:00), peak \$0.26668/kWh (12:00–18:00). Winter weekdays use the same period boundaries at \$0.06087 / \$0.07492 / \$0.0869/kWh — a much flatter price signal. Weekends are flat-rate. Main experiments use the summer schedule, since its sharper contrast is a more informative test of whether the policy shifts load off-peak.

- **Grid limit.** P<sub>grid</sub> = κ·P<sub>peak</sub>, where P<sub>peak</sub> is the peak total power on a typical training day if every vehicle charged at full rate on arrival. κ is chosen so that the limit binds during a substantial part of the occupied time, as in Ferragut et al., and results are also shown for a looser and a tighter κ. As a reference point, the real infrastructure is rated well below what full simultaneous charging would draw: Caltech's transformer defaults to 150 kW against a theoretical 54 × 6.656 ≈ 359 kW if every port ran at once, and JPL splits its ports across a 45 kW and a 150 kW transformer group rather than one shared limit. The single scalar P<sub>grid</sub> used here is a deliberate simplification of that multi-constraint reality (already noted as an assumption in Section VIII.G), and these real figures are what κ is calibrated against rather than an arbitrary choice.

**D. Report model**

Each test or training day is replayed under one of the following ways of generating t<sub>i</sub><sup>rep</sup>. A recorded-report mode (driver-entered requestedDeparture, as in Li & Sun) was planned but dropped after checking the field directly: only 0.2 % of a 500-session Caltech sample had it populated, too sparse to train or test on. The synthetic modes below are therefore the entire basis of the uncertainty experiments, not a supplement to real reports. The same random draws are reused for all methods.

| **Mode**        | **How the report is generated**                                                                                                  | **Source of the idea** |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------|
| M0 Truthful     | t<sup>rep</sup> = t<sup>dep</sup>                                                                                                | Reference case.        |
| M2 Noisy        | t<sup>rep</sup> = t<sup>dep</sup> + ε, ε ~ U(−w, w), w ∈ {0.5, 1, 2} h                                                           | Ferragut et al.        |
| M3 Late reports | t<sup>rep</sup> = t<sup>dep</sup> + δ, δ ~ U(0, w) for a share q of vehicles (vehicle leaves earlier than announced)             | Ting et al.            |
| M4 Strategic    | A share ρ ∈ {0.1, 0.25, 0.5} of vehicles reports t<sup>rep</sup> = t<sup>dep</sup> − s, s ∈ {0.5, 1, 2} h; the rest are truthful | Ferragut et al.        |

M1 (recorded driver reports) is omitted from the table above, for the reason given in Section X.D. Reports are clipped so that they stay after the arrival time. Since M2–M4 are simulated, the results show sensitivity and exploitability of the policy, not how often real drivers misreport (see Limits in Section VII).

**E. Simulator**

We build on ACN-Sim for data access, session replay and its Gym interface. The environment only needs energy bookkeeping, so if ACN-Sim's interface proves too restrictive for the projection layer and the report model, a thin Gymnasium wrapper around the same session data is used instead, and EV2Gym remains the fallback. As a sanity check, an uncontrolled replay (each vehicle at full rate on arrival, no grid limit) must reproduce the recorded delivered energy within a small tolerance.

**F. Training and experiments**

PPO-Lagrangian is the main learner, using an existing implementation of constrained PPO. SAC-Lagrangian is added if time allows. Each configuration is trained with at least five seeds. The planned experiments are:

| **Exp.** | **Question**                                                                      | **Setup**                                                                                                                                                                                         |
|----------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| E1       | How does the learned policy compare with the baselines when reports are truthful? | Train and test under M0.                                                                                                                                                                          |
| E2       | How fast does performance degrade with report noise?                              | Test under M2, M3 with increasing w and q.                                                                                                                                                        |
| E3       | How much extra service does a misreporting driver gain? (H1, H2)                  | Test under M4; paired replay described in Section XI.                                                                                                                                             |
| E4       | Do curtailment and randomised training reduce the gain? (H3)                      | Compare the standard policy, the policy trained on randomised reports (mix of M2–M4), and the variant that stops serving a vehicle after its reported departure.                                  |
| E5       | Does the projection scheme matter?                                                | Compare the uniform-cut projection (Section VIII.C) against a priority-weighted variant that cuts low-laxity vehicles less, and against RL with a soft grid-limit cost instead of any projection. |
| E6       | Does it transfer?                                                                 | Test on JPL and on the post-shift period.                                                                                                                                                         |

**XI. Evaluation Metrics and Baselines**

**A. Performance Metrics**

Let E<sub>i</sub> be the total energy delivered to vehicle i by its actual departure time, and e<sub>i</sub> be its requested amount. The satisfaction ratio is defined as s<sub>i</sub> = E<sub>i</sub> / e<sub>i</sub>.

| **Metric**         | **Calculation**                                                                                                               | **Primary purpose**                                                                                                                |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Cost               | C = Σ<sub>t</sub> c(t)·Σ<sub>i</sub> p<sub>i</sub>(t)·Δ per day; also reported per kWh delivered (Cost / ΣE<sub>i</sub>)      | Total financial expenditure. The per-kWh figure ensures a fair comparison across algorithms that deliver different total energies. |
| Unmet-energy share | U = Σ<sub>i</sub> max(0, e<sub>i</sub> − E<sub>i</sub>) / Σ<sub>i</sub> e<sub>i</sub>                                         | Share of total requested demand left unserved across all vehicles.                                                                 |
| Deadline-miss rate | Proportion of vehicles where E<sub>i</sub> \< 0.95·e<sub>i</sub>                                                              | Share of drivers receiving less than 95% of their requested charge.                                                                |
| Grid violations    | Total time steps where Σ<sub>i</sub> p<sub>i</sub>(t) \> P<sub>grid</sub>, plus the maximum excess power                      | Verifies constraint safety. Should be strictly zero for projected policies; evaluated explicitly for soft-constraint variants.     |
| High-power ratio   | W = Σ<sub>t,i</sub> 1\[p<sub>i</sub>(t) \> 0.8·P<sub>port</sub>\] / Σ<sub>t,i</sub> 1\[p<sub>i</sub>(t) \> 0\]                | Share of active charging time spent at high power output (an intensity proxy, not an exact battery-health measure).                |
| Fairness           | Jain's fairness index F = (Σ<sub>i</sub> s<sub>i</sub>)<sup>2</sup> / (n·Σ<sub>i</sub> s<sub>i</sub><sup>2</sup>)             | Whether service deficits are spread evenly across users or concentrated on a few.                                                  |
| Gap to optimal     | G<sub>C</sub> = (C<sub>kWh</sub> − C\*<sub>kWh</sub>)/C\*<sub>kWh</sub> for cost; G<sub>U</sub> = U − U\* for unserved energy | Distance from the theoretical full-information offline benchmark.                                                                  |
| Misreporting gain  | G<sub>i</sub>(s) = E<sub>i</sub>(early report by s hours) − E<sub>i</sub>(honest report)                                      | Extra energy a driver gains by intentionally reporting an earlier departure time.                                                  |
| Decision time      | Average wall-clock execution time (ms) per time step                                                                          | Evaluates real-time computational overhead.                                                                                        |

Note on battery wear: the high-power ratio W serves purely as a rough proxy for charging intensity. Because ACN-Data does not include battery thermals, chemistry profiles or degradation rates, W should not be read as a physical battery-health calculation.

**Paired Replay Protocol for Misreporting**

To isolate the exact advantage gained by misreporting departure times, a paired replay methodology is used:

- Select a focal vehicle on a test day.

- Run the simulation twice under identical initial conditions, arrival sequences and background driver behaviour.

- In run 1, the focal vehicle reports its departure honestly (t<sub>rep</sub> = t<sub>dep</sub>).

- In run 2, the focal vehicle lies and claims it will leave s hours earlier (t<sub>rep</sub> = t<sub>dep</sub> − s).

- The resulting difference in delivered energy (E<sub>i</sub>) yields the exact misreporting gain G<sub>i</sub>(s), in both absolute kWh and percentage gain.

**B. Baseline Algorithms**

| **Baseline**                    | **Strategy description**                                                                                            | **Benchmarking role**                                                               |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| FCFS (First-Come, First-Served) | Charges vehicles in order of arrival time at maximum rate until P<sub>grid</sub> is reached.                        | Simple non-deadline-aware heuristic baseline.                                       |
| EDF (Earliest Deadline First)   | Prioritizes vehicles based on earliest reported departure time (t<sub>rep</sub>).                                   | Standard deadline-aware rule; directly tests vulnerability to early-departure lies. |
| LLF (Least Laxity First)        | Prioritizes vehicles with the lowest laxity, based on reported deadline and remaining demand.                       | Slack-aware rule widely used in dynamic scheduling.                                 |
| Unprojected RL                  | Identical PPO-Lagrangian setup, but replaces the projection layer with a soft penalty cost for grid-limit breaches. | Isolates the practical value of the hard safety projection.                         |
| MPC (Model Predictive Control)  | Receding-horizon linear program (LP) solving for currently connected vehicles based on declared deadlines.          | Standard optimization-based online benchmark.                                       |
| Offline optimum                 | Full-information LP with complete hindsight of future arrivals, actual departures and energy demands.               | Establishes the full-information benchmark for performance.                         |

**Offline Optimal Formulation**

The offline benchmark solves a global linear program minimizing energy purchase costs plus unserved-energy penalties:

minimize Σ<sub>t</sub> c(t)·Σ<sub>i</sub> p<sub>i</sub>(t)·Δ + M·Σ<sub>i</sub> m<sub>i</sub> s.t. 0 ≤ p<sub>i</sub>(t) ≤ P<sub>port</sub>, Σ<sub>i</sub> p<sub>i</sub>(t) ≤ P<sub>grid</sub>, and the individual vehicle energy constraints.

Setting the penalty multiplier M significantly higher than peak electricity prices ensures the solver prioritizes fulfilling energy needs over saving money whenever physical capacity permits. Because power allocation is continuous, this problem reduces to a linear program; it only becomes a mixed-integer linear program (MILP) if discrete charging steps or hardware switching constraints are introduced. Note that the heuristic baselines (FCFS, EDF, LLF) do not explicitly optimize for electricity tariffs, so their cost efficiency must be analyzed alongside unserved-energy rates rather than in isolation.

**C. Testing Protocol**

- **Identical test conditions:** every algorithm is evaluated over the exact same held-out test days, using matching arrival sequences, price profiles and misreporting behaviour.

- **Statistical rigor:** RL results are averaged across a minimum of five random seeds. Final figures report mean values alongside 95% confidence intervals generated via test-day bootstrapping.

- **Comparative variants:** experiments evaluate two main policy variants — one trained exclusively on honest reporting (M0), and one trained with randomized report noise (M2–M4) — to test overall robustness.

- **Safety mandate:** projected RL policies must achieve zero grid-constraint violations across all runs. Unprojected variants report raw violation counts and peak power excesses.

- **Core target requirements:** success requires strict grid compliance and keeping unserved energy within the target service budget (d<sub>1</sub>). All other metrics highlight secondary trade-offs across cost, fairness, compute speed and system stability.
