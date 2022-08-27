from abc import ABC, abstractmethod

import numpy as np


class AbstractArm(ABC):
    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError


class ArmWithBetaPosterior(AbstractArm):
    def __init__(self, seed: int):
        self.alpha = 0.0
        self.beta = 0.0
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float):
        self.alpha += new_val
        self.beta += 1 - new_val


class ArmWithAdaptiveBetaPosterior(AbstractArm):
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.alpha = 0.0
        self.beta = 0.0
        self.propensity_scores = []
        self.sample_averages = []
        self.all_rewards = []
        self.n_pulls = []
        self.startup_complete = False
        self.doubly_summed = []
        self.was_pulled = False
        self.this_reward = np.nan
        self.this_propensity = np.nan
        self.prev_avg = np.nan
        self.this_avg = np.nan
        self.prev_doubly_total = 0
        self.pulls = 0
        self.t = 0

    def __len__(self):
        return 0 if len(self.n_pulls) == 0 else self.n_pulls[-1]

    def sample(self):
        return self.rng.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float, propensity_score: float, was_pulled: bool):
        self.this_propensity = propensity_score
        self.this_reward = new_val
        self.t += 1
        self.was_pulled = was_pulled
        if not self.startup_complete:
            self.prev_avg = new_val
            self.was_pulled = True
            self.pulls += 1
            self.startup_complete = True
            return
        if was_pulled:
            new_avg = (self.pulls * self.prev_avg + new_val) / (self.pulls + 1)
            self.this_avg = new_avg
            self.pulls += 1
        alpha, beta = self.compute_alpha_beta_3()
        self.set(alpha, beta)

    def compute_ipw_estimator(self):
        indicator = np.diff(self.n_pulls, prepend=0)
        rewards = self.all_rewards * indicator
        frac = rewards / np.array(self.propensity_scores)
        avg = np.average(frac)
        norm_avg = avg
        return norm_avg

    def compute_doubly_estimator(self):
        identity = self.was_pulled  # was_pulled
        this_reward = self.this_reward  # this_reward
        this_prop_score = self.this_propensity  # this_prop_score
        prev_avg = self.prev_avg  # prev_avg
        prev_total = self.prev_doubly_total  # self.doubly_summed[-1] if len(self.doubly_summed) else 0  # prev_total
        new_addend = prev_avg + (this_reward - prev_avg) * identity / this_prop_score
        total = prev_total + new_addend
        new_estimate = total / (self.t - 1)
        self.prev_doubly_total = total
        return new_estimate

    def compute_alpha_beta_3(self):
        corrected_avg_reward = self.compute_doubly_estimator()
        corrected_avg_reward = min(1.0, corrected_avg_reward)
        corrected_avg_reward = max(0.0, corrected_avg_reward)
        alpha = self.pulls * corrected_avg_reward
        beta = self.pulls * (1 - corrected_avg_reward)
        return alpha, beta

    def compute_gamma(self, new_val):
        r_t = np.array(self.all_rewards)
        avg_r_t = np.array(self.sample_averages)
        indicator_func = np.diff(self.n_pulls, prepend=0)
        pi_t = np.array(self.propensity_scores)
        Gamma = []
        for t in range(1, len(self.n_pulls)):
            g = avg_r_t[t - 1] + indicator_func[t] * (r_t[t] - avg_r_t[t - 1]) / pi_t[t]
            Gamma.append(g)
        return np.array(Gamma)

    def compute_mean(self, Gamma: np.ndarray):
        pi_sqrt = np.sqrt(self.propensity_scores)
        pi_sqrt_gamma = pi_sqrt * Gamma
        numerator = np.sum(pi_sqrt_gamma)
        pi_cum = np.sum(pi_sqrt)
        return numerator / pi_cum

    def compute_alpha_beta(self, Gamma: np.ndarray):
        n = self.n_pulls[-1]
        mu = self.compute_mean(Gamma)
        alpha = mu * (n + 2) - 1
        beta = n - alpha
        return alpha, beta

    def compute_alpha_beta_2(self, Gamma: np.ndarray):
        pi_sqrt = np.sqrt(self.propensity_scores)
        pi_sqrt_gamma = pi_sqrt * Gamma
        numerator = np.sum(pi_sqrt_gamma)
        pi_cum = np.sum(pi_sqrt)
        alpha = numerator / pi_cum * self.n_pulls[-1]
        beta = self.n_pulls[-1] - alpha
        return alpha, beta


class AbstractBandit(ABC):
    def __init__(self, k: int, name: str, seed: int):
        self.name = name
        self.k = k
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def set(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        self.arms = [ArmWithBetaPosterior(seed + arm_index) for arm_index in range(k)]
        super(ThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        samples = [a.sample() for a in self.arms]
        return int(np.argmax(samples))

    def set(self, arm: int, alpha: float, beta: float):
        self.arms[arm].set(alpha, beta)

    def update(self, arm: int, reward: float):
        if reward == 1 or reward == 0:
            # Bernoulli reward
            self.arms[arm].update(reward)
        else:
            bernoulli_reward = int(self.rng.uniform() < reward)
            self.arms[arm].update(bernoulli_reward)

    def alphas(self):
        return np.array([a.alpha for a in self.arms])

    def betas(self):
        return np.array([a.beta for a in self.arms])

    def __len__(self):
        return len(self.arms)


class BudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        self.reward_arms = [ArmWithBetaPosterior(seed + arm_index) for arm_index in range(k)]
        self.cost_arms = [ArmWithBetaPosterior(seed + k + arm_index) for arm_index in range(k)]
        super(BudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        reward_samples = [a.sample() for a in self.reward_arms]
        cost_samples = [a.sample() for a in self.cost_arms]
        reward_cost_ratio = np.array(reward_samples) / np.array(cost_samples)
        return int(np.argmax(reward_cost_ratio))

    def set(self, arm: int, alpha_r: float, beta_r: float, alpha_c: float, beta_c: float):
        self.reward_arms[arm].set(alpha_r, beta_r)
        self.cost_arms[arm].set(alpha_c, beta_c)

    def update(self, arm: int, reward: float, cost: float):
        if reward == 1 or reward == 0:
            # Bernoulli reward
            self.reward_arms[arm].update(reward)
        else:
            bernoulli_reward = int(self.rng.uniform() < reward)
            self.reward_arms[arm].update(bernoulli_reward)
        if cost == 1 or cost == 0:
            # Bernoulli cost
            self.cost_arms[arm].update(cost)
        else:
            bernoulli_cost = int(self.rng.uniform() < cost)
            self.cost_arms[arm].update(bernoulli_cost)

    def __len__(self):
        return len(self.cost_arms)


class AdaptiveBudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str, seed: int):
        self.reward_arms = [ArmWithAdaptiveBetaPosterior(arm_index) for arm_index in range(k)]
        self.cost_arms = [ArmWithAdaptiveBetaPosterior(k + arm_index) for arm_index in range(k)]
        super(AdaptiveBudgetedThompsonSampling, self).__init__(k, name, seed)

    def sample(self) -> int:
        arm_lengths = [len(a) == 0 for a in self.reward_arms]
        if np.any(arm_lengths):
            return [i for i in range(len(arm_lengths)) if arm_lengths[i]][0]
        reward_samples = [a.sample() for a in self.reward_arms]
        cost_samples = [a.sample() for a in self.cost_arms]
        reward_cost_ratio = np.array(reward_samples) / np.array(cost_samples)
        return int(np.argmax(reward_cost_ratio))

    def set(self, arm: int, alpha_r: float, beta_r: float, alpha_c: float, beta_c: float):
        self.reward_arms[arm].set(alpha_r, beta_r)
        self.cost_arms[arm].set(alpha_c, beta_c)

    def update(self, arm: int, reward: float, cost: float):
        prop_scores = self.estimate_propensity_scores()
        for i in range(len(self.cost_arms)):
            if reward == 1 or reward == 0:
                # Bernoulli reward
                self.reward_arms[i].update(reward, prop_scores[i], was_pulled=arm == i)
            else:
                bernoulli_reward = int(self.rng.uniform() < reward)
                self.reward_arms[i].update(bernoulli_reward, prop_scores[i], was_pulled=arm == i)
            if cost == 1 or cost == 0:
                # Bernoulli cost
                self.cost_arms[i].update(cost, prop_scores[i], was_pulled=arm == i)
            else:
                bernoulli_cost = int(self.rng.uniform() < cost)
                self.cost_arms[i].update(bernoulli_cost, prop_scores[i], was_pulled=arm == i)

    def estimate_propensity_scores(self):
        alpha_reward = [a.alpha + 1 for a in self.reward_arms]
        beta_reward = [a.beta + 1 for a in self.reward_arms]
        alpha_cost = [a.alpha + 1 for a in self.cost_arms]
        beta_cost = [a.beta + 1 for a in self.cost_arms]
        scores_reward = self.rng.beta(alpha_reward, beta_reward, size=(1000, len(alpha_cost)))
        scores_cost = self.rng.beta(alpha_cost, beta_cost, size=(1000, len(alpha_cost)))
        scores = scores_reward / scores_cost
        scores = np.sum(scores, axis=0)
        # scores = np.array([
        #     np.sum(self.rng.beta(ra.alpha + 1, ra.beta + 1, size=1000) / self.rng.beta(ca.alpha + 1, ca.beta + 1, size=1000))
        #     for ra, ca in zip(self.reward_arms, self.cost_arms)
        # ])
        return scores / np.sum(scores)

    def __len__(self):
        return len(self.cost_arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.cost_arms])
