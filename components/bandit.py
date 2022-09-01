from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


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
    def __init__(self, seed: int, ci: str):
        self.rng = np.random.default_rng(seed)
        self.alpha = 0.0
        self.beta = 0.0
        self.startup_complete = False
        self.doubly_summed = []
        self.was_pulled = False
        self.this_reward = np.nan
        self.this_propensity = np.nan
        self.prev_avg = np.nan
        self.this_avg = np.nan
        self.prev_doubly_total = 0
        self.prop_scores_sqrt_total = 0
        self.pulls = 0
        self.t = 0
        self._ci = ci

    def __len__(self):
        return self.t

    def sample(self):
        return self.rng.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float, propensity_score: float, was_pulled: bool):
        # Set values that must be updated continuously
        self.this_propensity = propensity_score
        self.this_reward = new_val
        self.t += 1
        self.was_pulled = was_pulled
        # Handle startup
        if not self.startup_complete:
            self.prev_avg = new_val
            self.this_avg = new_val
            self.pulls += 1
            self.startup_complete = True
            return
        # Adjust values after the arm was pulled
        # 1. update previous average
        # 2. update current_average
        # 3. increment number of pulls
        if was_pulled:
            self.prev_avg = self.this_avg
            new_avg = (self.pulls * self.prev_avg + new_val) / (self.pulls + 1)
            self.this_avg = new_avg
            self.pulls += 1
        # Update alpha and beta
        alpha, beta = self._compute_alpha_beta_uncorrected()
        self.set(alpha, beta)

    def hoeffding_ci(self, alpha=0.05):
        return np.sqrt(-1 / (2 * self.pulls) * np.log(alpha / 2))

    def wilson_ci(self, alpha=0.05):
        n = self.pulls
        ns = self.this_avg * n
        nf = n - ns
        z = 1.96 if alpha == 0.05 else stats.norm.interval(1 - alpha)[1]
        z2 = np.power(z, 2)
        return 2 / (n + z2) * np.sqrt((ns * nf) / n + z2 / 4)

    def combined_ci(self, alpha=0.01):
        return min(self.wilson_ci(alpha), self.hoeffding_ci(alpha))

    def get_ci(self):
        if self._ci == "hoeffding":
            return self.hoeffding_ci()
        elif self._ci == "wilson":
            return self.wilson_ci()
        elif self._ci == "combined":
            return self.combined_ci()
        elif self._ci is None:
            return 0.0
        else:
            raise ValueError

    def _compute_sample_average_estimator(self):
        ci = self.get_ci()
        avg = self.this_avg + ci
        return min(1.0, max(0.0, avg))

    def _compute_doubly_estimator(self):
        identity = self.was_pulled
        this_reward = self.this_reward
        this_prop_score = self.this_propensity + 1e-5
        this_prop_score_sqrt = np.sqrt(this_prop_score)
        self.prop_scores_sqrt_total += this_prop_score_sqrt
        prev_avg = self.prev_avg
        prev_total = self.prev_doubly_total
        Gamma = prev_avg + (this_reward - prev_avg) * identity / this_prop_score
        use_adaptive_weights = False
        if use_adaptive_weights:
            new_addend = this_prop_score_sqrt * Gamma
            total = prev_total + new_addend
            new_estimate = total / self.prop_scores_sqrt_total
        else:
            new_addend = Gamma
            total = prev_total + new_addend
            new_estimate = total / self.t
        self.prev_doubly_total = total
        return new_estimate

    def _compute_alpha_beta_doubly_corrected(self):
        corrected_avg_reward = self._compute_doubly_estimator()
        corrected_avg_reward = min(1.0, max(0.0, corrected_avg_reward))  # averages out of [0, 1] are not possible
        current_alpha = self.alpha
        current_beta = self.beta
        alpha = current_alpha + corrected_avg_reward
        beta = current_beta + (1 - corrected_avg_reward)
        # alpha = self.pulls * corrected_avg_reward
        # beta = self.pulls * (1 - corrected_avg_reward)
        return alpha, beta

    def _compute_alpha_beta_uncorrected(self):
        avg = self._compute_sample_average_estimator()
        alpha = avg * self.pulls
        beta = self.pulls * (1 - avg)
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
    def __init__(self, k: int, name: str, seed: int, ci_reward: str, ci_cost: str):
        self.ci_reward = ci_reward
        self.ci_cost = ci_cost
        self.reward_arms = [ArmWithAdaptiveBetaPosterior(arm_index, ci=ci_reward) for arm_index in range(k)]
        self.cost_arms = [ArmWithAdaptiveBetaPosterior(k + arm_index, ci=ci_cost) for arm_index in range(k)]
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

    def update(self, arm: int, reward: float, cost: float, estimate_props: bool = False):
        prop_scores = self.estimate_propensity_scores() if estimate_props else np.zeros(len(self.reward_arms), dtype=float)
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
        selected_arms = np.argmax([
            self.rng.beta(ra.alpha + 1, ra.beta + 1, size=1000) / self.rng.beta(ca.alpha + 1, ca.beta + 1, size=1000)
            for ra, ca in zip(self.reward_arms, self.cost_arms)
        ], axis=0)
        arms, counts = np.unique(selected_arms, return_counts=True)
        result = np.zeros(len(self.reward_arms))
        result[arms] = counts
        result = result / np.sum(result)
        return result

    def __len__(self):
        return len(self.cost_arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.cost_arms])
