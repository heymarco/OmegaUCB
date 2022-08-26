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
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0

    def sample(self):
        return np.random.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float):
        self.alpha += new_val
        self.beta += 1 - new_val


class ArmWithAdaptiveBetaPosterior(AbstractArm):
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.propensity_scores = []
        self.sample_averages = []
        self.all_rewards = []
        self.n_pulls = []
        self.startup_complete = False
        self.doubly_summed = []

    def __len__(self):
        return 0 if len(self.n_pulls) == 0 else self.n_pulls[-1]

    def sample(self):
        return np.random.beta(a=self.alpha + 1, b=self.beta + 1)

    def mean(self):
        return (self.alpha + 1) / (self.alpha + self.beta + 2)

    def set(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, new_val: float, propensity_score: float, was_pulled: bool):
        self.propensity_scores.append(propensity_score)
        self.all_rewards.append(new_val)
        if not self.startup_complete:
            self.sample_averages.append(new_val)
            self.n_pulls.append(1)
            self.startup_complete = True
            return
        if not was_pulled:
            self.n_pulls.append(self.n_pulls[-1])
            self.sample_averages.append(self.sample_averages[-1])
        else:
            new_avg = (self.n_pulls[-1] * self.sample_averages[-1] + new_val) / (self.n_pulls[-1] + 1)
            self.sample_averages.append(new_avg)
            self.n_pulls.append(self.n_pulls[-1] + 1)
        # Gamma = self.compute_gamma(new_val)
        # alpha, beta = self.compute_alpha_beta_2(Gamma)
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
        identity = self.n_pulls[-1] - self.n_pulls[-2]  # was_pulled
        this_reward = self.all_rewards[-1]  # this_reward
        this_prop_score = self.propensity_scores[-1]  # this_prop_score
        prev_avg = self.sample_averages[-2]  # prev_avg
        prev_total = self.doubly_summed[-1] if len(self.doubly_summed) else 0  # prev_total
        new_addend = prev_avg + (this_reward - prev_avg) * identity / this_prop_score
        total = prev_total + new_addend
        new_estimate = total / (len(self.n_pulls) - 1)
        self.doubly_summed.append(total)
        return new_estimate

    def compute_alpha_beta_3(self):
        corrected_avg_reward = self.compute_doubly_estimator()
        corrected_avg_reward = min(1.0, corrected_avg_reward)
        alpha = self.n_pulls[-1] * corrected_avg_reward
        beta = self.n_pulls[-1] * (1 - corrected_avg_reward)
        if beta < 0:
            beta = 0
        if alpha < 0:
            alpha = 0
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
    def __init__(self, k: int, name: str):
        self.name = name
        self.k = k

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
    def __init__(self, k: int, name: str):
        self.arms = [ArmWithBetaPosterior() for _ in range(k)]
        super(ThompsonSampling, self).__init__(k, name)

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
            bernoulli_reward = int(np.random.uniform() < reward)
            self.arms[arm].update(bernoulli_reward)

    def alphas(self):
        return np.array([a.alpha for a in self.arms])

    def betas(self):
        return np.array([a.beta for a in self.arms])

    def __len__(self):
        return len(self.arms)


class BudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str):
        self.reward_arms = [ArmWithBetaPosterior() for _ in range(k)]
        self.cost_arms = [ArmWithBetaPosterior() for _ in range(k)]
        super(BudgetedThompsonSampling, self).__init__(k, name)

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
            bernoulli_reward = int(np.random.uniform() < reward)
            self.reward_arms[arm].update(bernoulli_reward)
        if cost == 1 or cost == 0:
            # Bernoulli cost
            self.cost_arms[arm].update(cost)
        else:
            bernoulli_cost = int(np.random.uniform() < cost)
            self.cost_arms[arm].update(bernoulli_cost)

    def __len__(self):
        return len(self.cost_arms)


class AdaptiveBudgetedThompsonSampling(AbstractBandit):
    def __init__(self, k: int, name: str):
        self.reward_arms = [ArmWithAdaptiveBetaPosterior() for _ in range(k)]
        self.cost_arms = [ArmWithAdaptiveBetaPosterior() for _ in range(k)]
        super(AdaptiveBudgetedThompsonSampling, self).__init__(k, name)

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
                bernoulli_reward = int(np.random.uniform() < reward)
                self.reward_arms[i].update(bernoulli_reward, prop_scores[i], was_pulled=arm == i)
            if cost == 1 or cost == 0:
                # Bernoulli cost
                self.cost_arms[i].update(cost, prop_scores[i], was_pulled=arm == i)
            else:
                bernoulli_cost = int(np.random.uniform() < cost)
                self.cost_arms[i].update(bernoulli_cost, prop_scores[i], was_pulled=arm == i)

    def estimate_propensity_scores(self):
        scores = []
        for ra, ca in zip(self.reward_arms, self.cost_arms):
            ratio = [ra.sample() / ca.sample() for _ in range(1000)]
            scores.append(np.sum(ratio))
        return np.array(scores) / np.sum(scores)

    def __len__(self):
        return len(self.cost_arms)

    def startup_complete(self):
        return not np.any([not a.startup_complete for a in self.cost_arms])
