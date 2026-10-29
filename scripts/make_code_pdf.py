"""Code snippets + explanations PDF. Code is read verbatim from the repo (with real line numbers), so it cannot drift from the source.
    python scripts/make_code_pdf.py      -> docs/briefing/SafeCharge_Code_Snippets.pdf   (needs reportlab; reuses make_briefing_pdfs helpers)
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_briefing_pdfs import ACCENT, B, BASE, BOX, H1, H2, OUT, P, ROOT, SMALL, T, Spacer, build, cover  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import PageBreak, Preformatted  # noqa: E402

CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.9, leading=8.5, backColor=colors.HexColor("#F4F6F9"), borderColor=colors.HexColor("#D5DDE8"),
                      borderWidth=0.5, borderPadding=4, leftIndent=2, rightIndent=2, spaceAfter=6, spaceBefore=2)
WIDTH = 116


def code(rel, a, b, expect=None):
    """Verbatim lines a..b (1-based, inclusive) of rel, with line numbers; long lines wrapped for print."""
    lines = (ROOT / rel).read_text().splitlines()
    if expect:
        assert expect in lines[a - 1], f"{rel}:{a} expected {expect!r}, found {lines[a - 1]!r}"
    out = []
    for n in range(a, b + 1):
        raw = lines[n - 1].rstrip()
        pre = f"{n:>4}  "
        if len(raw) + len(pre) <= WIDTH:
            out.append(pre + raw)
        else:
            ind = len(raw) - len(raw.lstrip())
            parts = textwrap.wrap(raw.strip(), WIDTH - len(pre) - ind - 4, break_long_words=False)
            out.append(pre + " " * ind + parts[0])
            out += [" " * len(pre) + " " * (ind + 4) + p for p in parts[1:]]
    return Preformatted("\n".join(out), CODE)


def block(title, rel, a, b, expect, notes, key=None):
    s = [P(f"{title} <font size='8.5' color='#666666'>({rel}, lines {a}-{b})</font>", H2), code(rel, a, b, expect)]
    s.append(T([["Lines", "What it does"]] + [[x, y.replace("<=", "&lt;=")] for x, y in notes], [20, 154]))
    if key:
        s += [Spacer(1, 1.5 * mm), BOX("<b>Remember:</b> " + key)]
    s.append(Spacer(1, 2 * mm))
    return s


def main():
    s = cover("Code snippets with explanations: the RL core and everything set up around it. Code is copied verbatim from the repo, with real line numbers.")
    s += [BOX("<b>How to read this.</b> Each block shows real code, then a table that explains it line by line (the numbers on the left are real line numbers in the file, so you can open the file and follow). "
              "Long lines are wrapped for print. Read in order: the setup (data, config) first, then the environment, the safety layer, the network, the algorithm, and last the evaluation code."),
          P("Contents", H2),
          B(["A. Setup: data cleaning, price curve, configs, report model", "B. The environment (the MDP): reset, view, observation, step", "C. The safety projection",
             "D. The baselines, the offline optimum and MPC", "E. The neural network (Deep Sets actor and critic)", "F. PPO-Lagrangian: advantages, dual step, rollout, update, warm start",
             "G. Training loop and evaluation code (train.py, metrics, paired replay)"])]

    # ------------------------------------------------------------------ A
    s += [PageBreak(), P("A. Setup", H1)]
    s += block("A1. Price curve", "src/safecharge/data.py", 25, 31, "def price_curve", [
        ("27", "summer is True if tariff is 'summer' (used everywhere in our experiments) or a real June-September date"),
        ("28", "pick the 5 summer or winter prices"), ("29", "hour of each of the 96 steps: 0, 0.25, 0.5 ..."),
        ("30", "searchsorted maps each hour to its price period (periods start at hours 0, 8, 12, 18, 23)"), ("31", "returns a 96-long array of $/kWh")],
        "Step 50 is hour 12.5, which falls in the 12-18 period, so its price is $0.26668 (the peak).")
    s += block("A2. Cleaning sessions into days (the core of build_days)", "src/safecharge/data.py", 97, 131, "d = df[(df", [
        ("97-101", "keep sessions with energy above 0 and disconnect after connect"),
        ("102-106", "day = local date of arrival; a session whose disconnect date is later crosses midnight, and with midnight='drop' it is removed (6.3% Caltech)"),
        ("107", "arrival step = ceil(minutes since midnight / 15): rounded UP"), ("108-109", "departure step = floor(...): rounded DOWN, capped at 96"),
        ("110-112", "drop sessions whose window became empty"), ("113-114", "weekdays only"),
        ("115", "cap = most energy a port can deliver in the stay = 6.656 kW x steps x 0.25 h"), ("119", "e = min(recorded kWh, cap): the energy the car needs"),
        ("121-130", "group by date, skip clock-change days, sort by arrival, and build one Day object per date (arrival, departure, energy, price, ids, recorded kWh, recorded driver report)")],
        "Rounding arrival up and departure down means the simulated stay is never longer than the real one. Every exclusion is counted in the stats dict so the share affected can be reported.")
    s += block("A3. Splits and cache", "src/safecharge/config.py", 16, 32, "def get_days", [
        ("19-20", "cache file name depends on site, midnight rule, tariff and excluded stations"), ("21-23", "if cached, load the saved list of days"),
        ("24-30", "else read the raw JSON, build the days, save them"), ("31-32", "return the days whose date lies inside the split's date range (train, val, test or post)")],
        "Splits are chronological. Data come from data/raw (not in GitHub) and the cache goes to data/processed.")
    s += [P("A4. Settings: base.yaml, derived yaml and e1.yaml", H2),
          code("configs/base.yaml", 3, 13, "site: caltech"),
          code("configs/derived_caltech_kappa0.2.yaml", 1, 11, "# generated"),
          T([["Setting", "Meaning"],
             ["exclude_stations / midnight / tariff", "The data choices (fast station out, midnight crossers dropped, summer tariff on all weekdays)"],
             ["splits", "Chronological train, validation, test, and a post-COVID set"],
             ["P_grid 34.61, P_peak 173.06", "Grid cap = 0.20 x typical full-rate peak, computed from TRAIN days only"],
             ["e_ref 8.359", "Mean energy per car; unmet energy is divided by it (cost 1)"],
             ["d1 0.7133, d2 2.923", "Budgets. d1 = 2% of requested energy per day in cost units; d2 = mean wear proxy of the FCFS/EDF/LLF baselines"],
             ["reward_scale 4.78", "Divides the dollar cost so a typical day's return is about -10"],
             ["offline_penalty_M 5.0", "Penalty per kWh unmet in the LP (about 19x the peak price)"]], [58, 116])]
    s += [code("configs/e1.yaml", 8, 33, "agent:"),
          T([["Setting", "Meaning"],
             ["hid 64, use_count true, pool mean", "Network width; add the occupied-port count to the pooled context; masked mean-pool (the design)"],
             ["lr 3e-4, lr_final 0.1", "Learning rate decays linearly to 10%"], ["n_envs 48, epochs 10, minibatch 256, clip 0.2", "48 days per update, 10 passes, PPO clip range"],
             ["gamma 0.999, gae_lambda 0.97", "Discount (design says 0.99; changed and disclosed) and GAE smoothing"], ["ent_coef 0.001, target_kl 0.02", "Small exploration bonus; stop an update early if the policy moves too far"],
             ["dual_lr 0.05, dual_clip, lam_max 10, j_ema 0.7", "Multiplier step size, clip of the relative violation, cap on lambda, smoothing of the cost estimate"],
             ["critic_warmup 30, bc {LLF, 200 epochs}", "Train only the critic for 30 updates; behaviour-cloning warm start from LLF (not in the design)"],
             ["scheme uniform", "Projection scheme (E5 would use priority)"], ["train_reports [M0]", "Train with truthful reports (E4 configs use mixtures)"],
             ["n_updates 1500, eval_every 20", "Training length; validation check every 20 updates"]], [58, 116])]
    s += block("A5. Report model", "src/safecharge/report_model.py", 12, 15, "def base_draws", [
        ("13", "a random generator seeded by (seed, day index): the same for every method and every mode"), ("14", "four arrays of uniform numbers: eps (error), late, sel (who is selected), imp (imputation)")],
        "These fixed draws are why every method sees identical reports, and why larger w or rho reuse the same numbers.")
    s += block("A6. Making the reported departure", "src/safecharge/report_model.py", 31, 49, "n = len(day.arr)", [
        ("32", "true departure as float steps"), ("34-35", "M0: report = true departure"), ("36-37", "M2: add eps x w/DELTA (w in hours, DELTA = 0.25 h per step)"),
        ("38-39", "M3: for a share q of cars (sel below q) add a late amount up to w hours"), ("40-41", "M4: for a share rho of cars subtract s hours (cheating early)"),
        ("42-44", "M1r: recorded driver report where it exists, else true departure plus an error from the real error pool"), ("47-48", "paired replay hook: force one focal car to report dep minus focal_s"),
        ("49", "never earlier than arrival")],
        "The vehicle always leaves at its true time; only the reported time changes.")

    # ------------------------------------------------------------------ B
    s += [PageBreak(), P("B. The environment (the MDP)", H1)]
    s += block("B1. reset", "src/safecharge/env.py", 27, 42, "def reset", [
        ("29", "the day (true times) and the reported departures come in through options"), ("32", "vid: which vehicle is on each port (-1 = empty)"), ("33", "r: remaining energy per port"),
        ("34-36", "E: delivered energy per vehicle; m: unmet energy per vehicle; dropped: no free port on arrival"), ("37-40", "bookkeeping: next arrival, last total power, power log, port power log"),
        ("41-42", "plug in cars arriving at step 0 and return the first observation")])
    s += block("B2. view: the only information a policy may use", "src/safecharge/env.py", 58, 69, "def view", [
        ("60-61", "occupied ports; vehicle index (0 for empty so indexing works)"), ("62", "tau = time left until the REPORTED departure, floored at 0, in hours"),
        ("63-64", "remaining energy r and laxity = tau - r / P_port (spare time at full power)"), ("65", "h = max useful power: min(port max, energy that would just finish the car in one step)"),
        ("66-67", "curtail mode (E4): no power once the reported departure has passed"), ("69", "returns only reported-time quantities; the true departure is never here")],
        "A test checks that changing only the true departure does not change what the policy sees.")
    s += block("B3. Observation", "src/safecharge/env.py", 71, 77, "def _obs", [
        ("73", "per port: [occupied, r/30, tau/24, laxity/24]: four features, scaled to about unit size"), ("74-76", "globals: price/0.27, sin and cos of time of day, last total power / grid cap"),
        ("77", "flattened: N x 4 port numbers + 4 globals = 4N + 4 (220 numbers for 54 ports)")])
    s += block("B4. step: one 15-minute decision", "src/safecharge/env.py", 80, 116, "def step", [
        ("83", "p_tilde = u x h: the requested power per port"), ("85", "project(): cut the requests so the total fits the grid cap"),
        ("88-91", "remaining energy r falls by p x 0.25 h; delivered energy E rises"), ("92-95", "log total power and per-port power (used by metrics)"), ("96-97", "dollars = price x total power x 0.25 h"),
        ("101-106", "cars whose TRUE departure is this step leave: unmet energy m = remaining r; cost 1 adds it; the port is freed"), ("107", "cost 1 divided by e_ref"),
        ("108", "cost 2 = share of ports above 80% of max power"), ("110-114", "advance time; plug in new arrivals; done after step 96"), ("116", "returns observation, reward = -dollars / reward_scale, done flag, info with costs")],
        "Everything the design defines is here: state, action, reward, two costs, episode. The projection sits inside step(), so the grid cap cannot be broken.")

    # ------------------------------------------------------------------ C
    s += [PageBreak(), P("C. The safety projection", H1)]
    s += block("C1. project", "src/safecharge/projection.py", 10, 30, "def project", [
        ("13", "clip requests to [0, h]"), ("14-15", "if the total is already within the cap, return unchanged (identity)"),
        ("16-21", "uniform: every port is cut by the same mu (w = 1). priority: cut mu x weight, so urgent cars are cut less"),
        ("23", "mu is searched between 0 and the largest request, so the total at the top end is 0"), ("24-29", "bisection, 60 halvings: if the total at mid still exceeds the cap, raise the lower end; else lower the upper end"),
        ("30", "return the result at the upper end, which is always feasible")],
        "The total is continuous and decreasing in mu, so bisection always finds the smallest mu that fits. Example: cap 20, requests 12/10/8 (total 30) become 8.67/6.67/4.67 with mu = 3.33.")

    # ------------------------------------------------------------------ D
    s += [PageBreak(), P("D. Baselines, offline optimum and MPC", H1)]
    s += block("D1. Rule baselines", "src/safecharge/baselines.py", 8, 27, "def rule_action", [
        ("9-10", "the same view the policy gets: reported times only"), ("11-17", "sort key: arrival (FCFS), reported departure (EDF) or laxity (LLF)"),
        ("19", "order occupied ports by the key (smallest first)"), ("20-26", "greedy: give each port its maximum power in that order until the grid cap is used up"),
        ("27", "return u = p / h so the environment treats it like any action")],
        "Rules see exactly what the learned policy sees, so the comparison is fair.")
    s += block("D2. Offline optimum (hindsight LP)", "src/safecharge/offline_lp.py", 15, 41, "def solve_offline", [
        ("16-24", "one variable x[i,t] for each car and each step between its arrival and TRUE departure"), ("25", "cost vector: price x 0.25 for charging, plus M for each kWh unmet"),
        ("26-28", "grid constraint per step: sum of x at step t at most P_grid"), ("29-34", "energy balance per car: sum of x x 0.25 + unmet = needed energy"),
        ("35", "bounds: 0 to port max for x, unmet at least 0"), ("36-38", "solve with HiGHS (about 3 ms per day)"), ("39-41", "return cost, unmet energy, the objective and the schedule")],
        "It knows the true departures and the whole day, so it is a lower bound. The experiment script asserts that no method beats it on any day.")
    s += block("D3. MPC (re-plan every step)", "src/safecharge/mpc.py", 16, 41, "def mpc_action", [
        ("18-20", "only the cars plugged in now"), ("23", "last allowed step for each car = its REPORTED departure (at least the current step)"), ("24-27", "variables and costs as in the offline LP, but only from now on"),
        ("28-32", "grid constraint per remaining step, energy balance with the car's remaining energy, solve"), ("33-41", "take only the first step of the plan and convert it to u = p / h")],
        "MPC trusts the declared deadlines. With wrong reports it postpones charging and runs out of time (about 30x more unmet energy at w = 2 h).")

    # ------------------------------------------------------------------ E
    s += [PageBreak(), P("E. The neural network (Deep Sets)", H1)]
    s += block("E1. Masked pooling", "src/safecharge/agents/deepsets.py", 30, 35, "def pool", [
        ("31", "cnt: number of occupied ports"), ("32", "context = average of the port embeddings over occupied ports only (empty ports are masked out)"),
        ("33-34", "optional ablation: also take the max over ports"), ("35", "append the occupied count / N so the network knows how many cars compete")],
        "Averaging makes the network independent of port order and of how many cars there are.")
    s += block("E2. Actor (the policy)", "src/safecharge/agents/deepsets.py", 52, 86, "class Actor", [
        ("56", "shared encoder: 4 features to 64 to 64, applied to every port"), ("59", "shared head: input = port embedding + context + 4 globals; output = 2 numbers per port; tiny last layer so the policy starts near 'half power'"),
        ("63-67", "split the observation, encode each port, zero out empty ports, pool into a context"), ("68-69", "concatenate (port embedding, context, globals) for every port and run the head"),
        ("70-71", "mean of the Gaussian and a log standard deviation (start -0.5, clamped to [-4, 1])"), ("74-77", "act: sample a = mean + noise x std (or the mean when testing); the applied action is sigmoid(a) in [0,1]"),
        ("80-82", "log-probability summed over occupied ports only"), ("85-86", "entropy over occupied ports")],
        "Parameters are shared across ports, so the policy works for 5 or 45 cars. A test checks it is permutation-equivariant and ignores empty ports.")
    s += block("E3. Critic (the value function)", "src/safecharge/agents/deepsets.py", 89, 105, "class Critic", [
        ("95-98", "same encoder; head takes the pooled context + globals and outputs 3 values"), ("100-105", "the 3 values = expected discounted reward, expected unmet-energy cost, expected wear cost")])

    # ------------------------------------------------------------------ F
    s += [PageBreak(), P("F. PPO-Lagrangian", H1)]
    s += block("F1. GAE and the dual step", "src/safecharge/agents/ppo_lag.py", 21, 36, "def gae", [
        ("22-29", "advantage = weighted sum of future TD errors, computed backwards through the 96 steps; returns = advantage + value"),
        ("32-36", "dual step: relative violation J/d - 1, clipped to [-1, 1]; lambda moves by alpha x that, never below 0 and never above lam_max")],
        "Lambda goes up when a cost is above its budget and down when it is below.")
    s += block("F2. Rollout (collect)", "src/safecharge/agents/ppo_lag.py", 65, 84, "def collect", [
        ("66-67", "K = 48 episodes in lockstep"), ("67", "choose 48 random training days (with replacement only if there are fewer than 48)"), ("68-69", "reset each env with its day and a report draw from the training mixture"),
        ("70-71", "buffers for observations, actions, log-probs, rewards and the 2 costs"), ("73-77", "for each of 96 steps: the actor samples a and its log-prob, u = sigmoid(a)"),
        ("78-83", "step every env, store reward and both costs"), ("84", "return the batch")])
    s += block("F3. Update, part 1: advantages for three signals", "src/safecharge/agents/ppo_lag.py", 94, 115, "def update", [
        ("98-99", "critic values for every state, multiplied by running scales"), ("100-105", "GAE separately for reward, cost 1 and cost 2"),
        ("107-108", "running scale of each signal (moving average of the return std)"), ("109", "normalise each advantage to mean 0, std 1"),
        ("110-111", "combined advantage = (A_reward - lam1 x A_cost1 - lam2 x A_cost2) / (1 + lam1 + lam2)"), ("114", "critic targets in normalised units")],
        "A large lambda makes the policy treat that cost like a big penalty.")
    s += block("F4. Update, part 2: PPO step and dual ascent", "src/safecharge/agents/ppo_lag.py", 117, 156, "for _ in range", [
        ("117-121", "10 epochs of shuffled minibatches of 256 samples"), ("122-125", "new log-prob, probability ratio, PPO clipped objective (clip 0.2)"), ("126", "entropy bonus"),
        ("127-131", "actor step with gradient clipping 0.5, only after the critic warm-up"), ("132-136", "critic regression to the targets"), ("137-145", "if the policy moved too far (KL), stop early"),
        ("147-150", "average undiscounted cost per day, smoothed, then dual_step updates the two lambdas"), ("151-156", "increment counter; return statistics for the log (costs, lambdas, return)")],
        "This is the whole algorithm: collect days, compute three advantages, take a PPO step on the lambda-weighted advantage, then update lambda from the measured costs.")
    s += block("F5. Behaviour-cloning warm start (not in the submitted design)", "src/safecharge/agents/ppo_lag.py", 159, 181, "def bc_pretrain", [
        ("161-167", "play the training days with LLF (or MPC) and record (observation, action) pairs"), ("168-169", "convert expert actions to logits (clamped to 0.05-0.95)"),
        ("170-181", "regress the actor's mean onto those logits, ignoring empty ports")],
        "Without this the policy stayed worse than every rule after 2,000 updates. Say it is an addition when asked.")

    # ------------------------------------------------------------------ G
    s += [PageBreak(), P("G. Training loop and evaluation code", H1)]
    s += block("G1. train.py: setup and loop", "scripts/train.py", 38, 50, "agent = PPOLag", [
        ("38", "create the agent with P_grid, e_ref, reward_scale and the two budgets from the derived config"), ("39", "pool of real report errors (for M1r)"),
        ("44-48", "resume from last.pt if it exists"), ("49-50", "otherwise run the behaviour-cloning warm start")])
    s += block("G2. train.py: main loop", "scripts/train.py", 61, 78, "t0 = time.time()", [
        ("63", "decay the learning rate"), ("64-65", "collect a batch and update"), ("66-72", "every 20 updates: evaluate on validation days, save best.pt if feasible and cheapest, save last.pt"),
        ("73-77", "print progress and write the CSV log")])
    s += block("G3. Metrics for one day", "src/safecharge/metrics.py", 34, 59, "def day_metrics", [
        ("40", "dollars = sum price x power x 0.25 h"), ("41", "s = delivered / requested per car"), ("49", "unmet share = total unmet / total requested"),
        ("50", "miss rate = share of cars with under 95% of their request"), ("51-52", "grid violations: count of steps above the cap, and the largest excess"),
        ("53", "high-power ratio: share of charging port-steps above 80% of max power (wear proxy)"), ("54", "Jain fairness (sum s)^2 / (n x sum s^2): 1 means everyone is served equally")])
    s += block("G4. Paired replay: the cheating measure", "src/safecharge/metrics.py", 70, 81, "def paired_gain", [
        ("73-74", "run 1: the focal car reports its true departure"), ("75-76", "run 2: it reports s hours earlier"), ("77-80", "same day, same policy, same other reports; gain = energy in run 2 minus run 1"),
        ("81", "returned in kWh and as % of the car's request")],
        "With s = 0 the gain is exactly 0 (a test). Everything else is held fixed, so any difference is caused by the report alone.")
    s += block("G5. Running a day with any policy", "src/safecharge/metrics.py", 11, 22, "def rollout", [
        ("13", "reset with the day and its reports"), ("15-20", "loop: act, step, add the two costs"), ("21-22", "time per step and the day's metrics")],
        "Rules, MPC and the learned policy all go through this one function.")
    build(OUT / "SafeCharge_Code_Snippets.pdf", s, "Code snippets")


if __name__ == "__main__":
    main()
    print("wrote", OUT / "SafeCharge_Code_Snippets.pdf")
