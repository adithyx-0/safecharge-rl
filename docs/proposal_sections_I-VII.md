<img src="media/040e85f86d52cf9c1f349fefff376b1f65568493.png" title="Logo" style="width:2.70833in;height:0.70833in" alt="Institution logo" />

**Term Project**

Project Proposal Documentation

**SAFECHARGE-RL: REINFORCEMENT LEARNING FOR EV SMART CHARGING**

**Team 32**

**Adithyan R \| AM.SC.U4AIE23065**

**Adithya Rajesh \| AM.SC.U4AIE23007**

**Balu M Krishna \| AM.SC.U4AIE23026**

**Guide**

**Dr. Geetha M**

Department of Computer Science & Engineering (AI)

**I. Problem Background**

More and more people are buying electric vehicles, and most of them get charged either at home or at a shared charging point at work or in public. The trouble starts when a lot of these vehicles are plugged in at the same time. If nobody is coordinating them, the total power they draw can go past what the local transformer or wiring was built to handle. This can result in equipment damage, blackouts, or expensive upgrades of the distribution infrastructure.

At the same time, electricity is not priced the same throughout the day anymore. Many places now use time-of-use rates or real-time pricing, so charging at the wrong hour can cost a lot more than charging at the right one. On top of that, every EV owner has a deadline — they need the car ready by a certain time — so charging cannot just be pushed off indefinitely to wait for a cheaper hour.

Putting this together: at every moment, someone (or something) has to decide how much power to give each connected vehicle, while keeping the total under the grid limit, without missing anyone's deadline, and while keeping the cost down. Future arrivals and future prices are not fully known ahead of time, so this has to be handled as an ongoing, moment-to-moment decision problem, not something that can be solved once and forgotten.

**II. Problem Statement**

The paper considers a common charging station with a fixed number of charging ports and a hard power constraint, which cannot be exceeded at any point in time. The electric vehicles (EVs) arrive at random times, and upon arrival, their reported deadlines and required charge levels become known (the reported deadline may differ from the real departure time). The problem is to find out a policy that at any point in time allocates power to the connected EVs in such a way that:

- The total power drawn by all vehicles together never goes above the grid limit — this is a hard rule that cannot be broken.

- As many vehicles as possible reach their target charge level before they leave — this should be respected almost always, but a rare miss is tolerable.

- The total money spent on electricity, given that prices change through the day, is kept as low as possible.

- The battery is not degraded more than necessary by charging too aggressively. ACN-Data has no battery state, so this is measured with a simple proxy, such as the share of charging time spent at high power.

This is a constrained Markov decision process (CMDP) where aside from the standard optimization objective function, there are additional objectives that have to be taken into account with prescribed levels of satisfaction. The paper provides an overview of several research directions, including vehicle-to-grid discharging opportunities, photovoltaic power generation, and other multi-station scenarios, as possible extensions of the basic model described in the paper.

The CMDP is defined by the following elements:

- **State:** for every connected vehicle, the energy still needed and the time left before its (reported) departure, together with the current electricity price, the time of day and the total power currently drawn.

- **Action:** the charging power assigned to each connected port, between zero and the maximum of that port.

- **Reward:** the negative of the electricity cost of the energy delivered in that time step.

- **Constraints:** the energy left unmet at departure must stay within a small budget (soft constraint, handled by the Lagrangian objective), and the total power must never exceed the grid limit (hard constraint, enforced by the projection layer).

- **Episode:** one day of operation divided into fixed time steps (for example 15 minutes), with vehicles arriving and leaving according to real charging sessions.

**Key challenges**

- **Unknown future:** arrivals, departures and prices are not known in advance, so every decision has to be made online.

- **Hard limit versus deadlines:** when many vehicles are present, the power limit can make it impossible to meet every deadline, so the policy has to decide who is shortchanged.

- **Competing objectives:** cost, deadlines and battery wear pull in different directions.

- **Unreliable departure information:** drivers enter their own departure time and these inputs are often wrong (Li & Sun, 2023, observe such errors in ACN-Data), so the deadline the policy sees may differ from the real one.

**III. Why Reinforcement Learning?**

There are three reasons why this problem calls for reinforcement learning rather than a more traditional method.

**The problem has to be solved online, one step at a time**

Methods like Linear or Mixed-Integer Programming can find the mathematically best charging schedule, but only if everything about the future is already known — when every car will arrive, when it will leave, and what the price will be at every future minute. In real life none of that is known ahead of time. Reinforcement learning is built for exactly this kind of situation: it makes a decision now, using only what is known so far, and updates as new information comes in.

**Simple rules do not balance competing goals well**

Rules like First-Come-First-Served or Earliest-Deadline-First are easy to code but they only look at one thing at a time. They do not notice that electricity is cheap right now, and First-Come-First-Served does not notice that one car is in far more of a hurry than another. A learned policy can weigh cost, urgency, fairness and battery wear together, because it is trained to do well on all of them at once rather than following one fixed rule.

**Model Predictive Control is a fair comparison, and it has a real trade-off**

Model Predictive Control (MPC) is a viable alternative that is worth considering for discussion rather than dismissal. MPC uses a new optimisation problem at every step that uses a forecast of prices and arrivals. It can be very effective if the forecast is accurate, but it is usually slower to execute at each step, and its effectiveness depends on the accuracy of the forecast. Reinforcement learning, by contrast, uses most of its compute budget to train an agent, and once trained, the agent can make decisions at decision time quickly. This is a fundamental difference, rather than an alternative with an obvious answer. Reinforcement learning, however, has no inherent safety guarantee, hence the need for the separate safety mechanism from Section VII, rather than an optional add-on. Reinforcement learning is not automatically robust either: Li & Sun (2023) report that RL trained on pre-COVID ACN-Data performs worse on later data, once charging behaviour has changed.

**IV. Literature Review**

**1. Li, H., Wan, Z., He, H. — Constrained EV Charging Scheduling Based on Safe Deep Reinforcement Learning**

*IEEE Transactions on Smart Grid, vol. 11, no. 3, pp. 2427–2439, 2020. DOI: 10.1109/TSG.2019.2955437.*

**What it does:** treats single-EV charging as a CMDP and uses a safe deep RL method to keep the charging within limits while keeping cost low, even though future prices and the departure time are uncertain.

**Where it falls short:** it only looks at one charger at a time, so there is no shared grid-capacity limit across many vehicles, and the safety it offers is learned rather than guaranteed by design. SafeCharge-RL is meant to close exactly this gap.

**2. Chen, G., Shi, X. — A Deep Reinforcement Learning-Based Charging Scheduling Approach with Augmented Lagrangian for Electric Vehicle**

*arXiv:2209.09772, 2022.*

**What it does:** also treats EV charging as a CMDP, and solves it by mixing the augmented Lagrangian method with Soft Actor-Critic (SAC). The actor network is updated using a Lagrangian value function, and a double-critic setup is used to avoid overestimating how good an action is.

**Where it falls short:** this is the closest match to the soft-constraint half of SafeCharge-RL's design, but it relies only on the augmented Lagrangian, with no separate hard-constraint layer underneath it. It also covers a single EV with hourly steps, real German electricity prices and arrival and departure times drawn from truncated normal distributions, so there is no shared grid limit and no real charging sessions. It does compare its cost against an MPC that knows the future exactly (its “theoretical optimum”), and we reuse that benchmarking idea.

**3. Ting, L. P.-Y., Şenol, A., Wang, H.-Y., Lai, H.-C., Chuang, K.-T., Liu, H. — Uncertainty-Aware Critic Augmentation for Hierarchical Multi-Agent EV Charging Control**

*arXiv:2412.18047, 2024.*

**What it does:** targets workplace charging with a station power limit, dynamic electricity prices and EVs that may leave earlier than their planned departure time. It uses hierarchical multi-agent actor-critic RL with a critic augmentation that accounts for departure uncertainty. It is tested on real electricity and building-load data with simulated EV behaviour, and compared with RL baselines and with a linear-programming oracle that knows the whole future; its total cost is comparable to that oracle.

**Where it falls short:** it models only early departure, using simulated EV arrivals and departures (truncated normal distributions) instead of real charging sessions, and the building’s contracted capacity is handled through a penalty cost rather than a hard guarantee. It shows that departure uncertainty matters, but it is not compared against simple rule-based schedulers such as Earliest-Deadline-First or Least-Laxity-First.

**4. Li, T., Sun, C. — Out-of-Distribution-Aware Electric Vehicle Charging**

*arXiv:2311.05941, 2023.*

**What it does:** studies EV charging under distribution shift using real Caltech ACN-Data, where charging behaviour changed after a pricing change and after COVID-19. It argues that MPC is often too conservative while RL tends to be overly aggressive and trusts its data too much, and proposes OOD-Charging, which blends a DDPG policy with the MPC scheduler used at Caltech through an “awareness radius” updated from the temporal-difference error. It also shows that the departure times and battery capacities entered by users in the app differ significantly from the true values.

**Where it falls short:** it uses a small station (two chargers in the experiments), a quadratic cost instead of electricity prices, and DDPG as the learned policy. Its experiments compare against an MPC baseline and RL variants, not against rule-based schedulers or an offline-optimal schedule, and its contribution is the robustness mechanism, not constrained RL for deadlines. It does show, on real ACN-Data, that RL trained before COVID degrades after the shift, and that user-entered departure times cannot be trusted blindly.

**5. Ferragut, A., Narbondo, L., Paganini, F. — Scheduling EV Charging with Uncertain Departure Times**

*IFIP Performance 2021, Milan, Italy, November 2021.*

**What it does:** analyses Earliest-Deadline-First and Least-Laxity-First when the departure times declared by users are uncertain, using a mean-field model of an overloaded charging facility with limited power. It shows that vehicles that under-report their deadline tend to receive more service, so users have an incentive to under-report, and it proposes a simple modification (serving a vehicle only up to its declared departure time) that removes this incentive. It ends with simulations on real traces from a Silicon Valley parking lot.

**Where it falls short:** it analyses rule-based policies with a mathematical model, not learned policies, and it does not consider electricity prices. Whether a learned charging policy can be gamed in the same way, and whether the curtailment idea carries over to RL, is left open; this is the question we add to SafeCharge-RL.

**6. Gabriele, G., Pavirani, F., Karimi Madahi, S. S., Develder, C. — Forecasting what Matters: Decision-Focused RL for Controlled EV Charging with Unknown Departure Times**

*Proc. ACM e-Energy ’26, Banff, Canada, June 2026. DOI: 10.1145/3744255.3811736. arXiv:2606.19199.*

**What it does:** addresses the case where the departure time is not available to the RL controller. A forecaster predicts it, but forecasters trained only for accuracy can pass their errors on to the controller, so the paper trains the forecaster end-to-end with feedback from the charging agent’s actions (decision-focused RL). It reports up to 14% higher total reward and 55% less unsupplied energy than RL without departure forecasting.

**Where it falls short:** it is a short (five-page) paper that covers the case where the departure time is missing altogether, rather than one that is reported but wrong or strategic. It is still a useful recent example of departure uncertainty in RL, and a forecast could later replace the reported time in our setting.

**7. Lee, Z. J., Li, T., Low, S. H. — ACN-Data: Analysis and Applications of an Open EV Charging Dataset**

*ACM e-Energy '19, pp. 139–149, 2019. DOI: 10.1145/3307772.3328313.*

**What it does:** introduces and studies ACN-Data, a real dataset of EV charging sessions collected at Caltech and JPL.

**Where it falls short:** it is not an RL paper at all — it is the dataset that SafeCharge-RL will actually train and test on, so it forms the empirical base of the whole project rather than a method to build on.

**What this tells us**

Taken together, the seven papers above show three things. First, safe and constrained RL for EV charging is already established: CMDP formulations (Li, Wan & He; Chen & Shi) and projection onto the allowed states and actions (Li & Sun), so a safety layer on its own is not new. Second, comparisons against a full-information optimum exist (Chen & Shi; Ting et al.), but on synthetic or simulated EV behaviour. Third, uncertain departure times are treated as random noise or missing information (Ting et al.; Gabriele et al.), Li & Sun show that driver-entered values are unreliable, and Ferragut et al. show that drivers can gain by under-reporting under rule-based schedulers. Based on our search, we did not find a study that treats reported departure times as something drivers may misreport to a learned policy, tested on real sessions from ACN-Data (Lee, Li & Low). SafeCharge-RL aims to study this.

**V. Existing Systems and Products**

- ChargePoint's Power Management software shares the available power among stations so that a maximum load set by the site is never exceeded, and its plans also offer time-of-use power sharing, scheduled charging and support for Automated Demand Response. These are configured limits, schedules and sharing rules; the published feature descriptions do not describe a learned, per-session policy.

- Enel X's JuiceNet was a platform that matches drivers' historical charging patterns, real-time input and signals from grid operators and utilities to manage charging demand, with support for OpenADR demand response. Its algorithms are not published, so we cannot say whether it used a learned policy. Enel X Way closed its North American EV charging business in October 2024, so it is listed only as a historical example.

- Caltech's Adaptive Charging Network (ACN) is a real, working multi-port charging system, and the source of the ACN-Data dataset and the ACN-Sim simulator used in this project. It schedules charging with model predictive control, in use at Caltech since 2018, using departure times and battery capacities entered by drivers through a mobile app, not reinforcement learning. Li & Sun report that these entered values differ significantly from the true ones.

- The Open Charge Point Protocol (OCPP) lets a central system send charging profiles, that is, power limits over time, to a station, including a maximum power for the whole charge point. It defines the message format but not how the limits are computed, so a learned policy such as ours could sit behind it.

- Open-source simulators provide ready-made charging environments. ACN-Sim (Lee et al.) is built around ACN-Data, includes a simple network model with a single capacity constraint and connects to OpenAI Gym for RL; EV2Gym (Orfanoudakis et al.) supports heuristics, mathematical programming and RL in one platform. We plan to build on such existing frameworks instead of writing a simulator from scratch.

**VI. Gap Analysis**

- Deployed schedulers depend on inputs that can be wrong. The MPC at Caltech uses driver-entered departure times that Li & Sun show to be unreliable, while Li & Sun describe MPC as conservative and independent of data. Commercial platforms (ChargePoint, JuiceNet) document configured limits, schedules and grid signals, but their algorithms are not public, so we cannot compare against them.

- The RL papers reviewed in Section IV each cover part of the problem: constrained formulations for a single EV (Li, Wan & He; Chen & Shi), a power-limited station with uncertain departures (Ting et al.), and robustness to distribution shift with projection onto allowed actions (Li & Sun). Recent work outside this review may combine a safety projection with a Lagrangian objective, so we do not claim that combination as new.

- Where safety-aware RL has been tried, the safety is often learned through constraint costs or Lagrangian multipliers (Li, Wan & He; Chen & Shi), which can still allow rare violations of the grid limit. Projection of actions has been used (Li & Sun), but in a small two-charger simulation with a quadratic cost. We did not find it evaluated together with deadline constraints on real session data where many vehicles share one hard limit. It therefore remains untested on real multi-EV charging data under a shared power constraint.

- Comparisons with a full-information optimum do exist (Chen & Shi use an MPC with perfect information; Ting et al. use a linear-programming oracle), but on synthetic or simulated EV behaviour. We did not find a comparison against a hindsight-optimal schedule on real session data with a shared power limit, so it is unclear how much performance RL leaves on the table in that setting.

- Uncertain departure times are recognised in the literature, but as random noise or missing information: Ting et al. sample early departures, Gabriele et al. forecast unavailable departure times, and Li & Sun show that driver-entered values are wrong. Ferragut et al. show that, under rule-based schedulers, vehicles that under-report their deadline receive more service. Based on our search of publicly available sources, we did not find work that treats a reported departure time as something a driver may deliberately misreport to a learned policy, or that measures how much extra service a misreporting driver gains from such a policy.

**VII. Proposed Research Direction**

The SafeCharge-RL does not introduce a novel algorithm. Instead, it models a shared-station charging problem with an additional difficulty – inaccurate departure reporting and analyzes it using existing RL approaches on actual data. The research question is: what is the difference between losses of cost and missed deadlines of a learning policy and of traditional rule-based policies under drivers’ self-reported departure times where some reports are erroneous or intentionally early, how much advantage can be obtained by a misreporting driver, and whether it is possible to mitigate the effect with a minor modification of the policy or training procedure?

We are planning to test three hypotheses. H1: a policy trained on drivers' self-reported departure times allocates more charge to drivers who report their departure times as earlier, as demonstrated by Ferragut et al. in relation to Earliest-Deadline-First and Least-Laxity-First. H2: cost and missed deadlines increase as the noise level or misreporting drivers' proportion increases. H3: servicing a vehicle only up until the declared departure time (suggested fix by Ferragut et al.) and randomizing the departure times during training mitigate the misreporting advantage at the expense of inefficiency.

To keep the learned policy safe at a shared station, we combine two mechanisms that have each been studied for EV charging (similar combinations may already exist in recent work, so the mechanism itself is not our claim):

- A hard-constraint Safety Projection Layer. Rather than using a penalty or a learned, probabilistic guarantee to discourage crossing the grid limit (as in Li, Wan & He, 2020, and Chen & Shi, 2022), the action proposed by the RL policy is passed through a quick optimisation step that projects it onto the set of allowed actions before it is actually applied. This makes it structurally impossible for the total power drawn to exceed the grid limit, no matter what the policy outputs — safety baked into the system, not just encouraged by a penalty. For a single shared power limit, this projection lowers all proposed powers by a common amount, keeping each between zero and the port maximum, until their sum fits under the limit; the amount is found with a simple one-dimensional search, so it is cheap to run.

- A constrained RL objective, using either PPO-Lagrangian or SAC-Lagrangian, for the softer goals of meeting deadlines and keeping battery wear within a budget; fairness across vehicles will be reported as an evaluation metric. This builds on the augmented-Lagrangian-plus-SAC idea of Chen & Shi (2022), adding the hard-constraint layer that their version does not have.

**Data.** We will use real sessions from Caltech's ACN-Data (Lee, Li & Low, 2019), which is public after registering for an access token: arrival and disconnect times, energy delivered or requested and, where available, the departure time entered by the driver. Each day is one episode, with earlier days for training, later days for testing and a second site as a cross-site test. ACN-Data has no electricity prices or battery state, so we use a time-of-use tariff and model each vehicle by the energy it still needs. The grid limit is set low enough for the station to be overloaded part of the time, as Ferragut et al. do. Driver-entered departure times are used where they exist; otherwise reports are generated artificially (uniform noise as in Ferragut et al., early departures as in Ting et al., and a share of drivers who report earlier on purpose), so the results show sensitivity and exploitability, not how often real drivers misreport.

**Comparison.** The learned policy will be compared against simple rule-based baselines (First-Come-First-Served, Earliest-Deadline-First, and a Least-Laxity-First rule that ranks vehicles by how little slack they have), an RL baseline with no hard safety layer, and an LP/MILP-computed offline-optimal schedule that knows the true departure times — thus providing a meaningful numerical assessment of how close the learned policy gets to the best possible schedule. Results will be reported as total cost, deadline-miss rate, grid-limit violations, the gap to the offline optimum, and the gain from misreporting (the extra energy a vehicle receives when it reports an earlier departure than when it reports truthfully). We will build on existing tools (ACN-Sim or EV2Gym, and standard RL libraries).

**Limits.** The misreporting is simulated, so we make no claim about how often real drivers misreport. ACN-Data has no battery state and comes from workplace and campus sites, so the results may not carry over to home or public charging.
