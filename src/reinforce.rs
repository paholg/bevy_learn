//! Vanilla policy gradient
//! See https://arxiv.org/abs/1604.06778

use std::collections::VecDeque;

use tch::{
    nn::{self, Module, OptimizerConfig},
    COptimizer, Device, Kind, Tensor,
};
use tch_distr::{Categorical, Distribution};
use tracing::info;

use crate::{Env, Obs, Trainer};

#[derive(Debug)]
struct PolicyNet {
    seq: nn::Sequential,
    device: Device,
}

impl PolicyNet {
    pub fn new<E: Env>(env: &E, n_hidden: i64) -> Self {
        let device = env.path().device();
        let hidden = nn::linear(
            env.path() / "reinforce-lhidden",
            E::NUM_OBSERVATIONS,
            n_hidden,
            nn::LinearConfig::default(),
        );
        let out = nn::linear(
            env.path() / "reinfoce-lout",
            n_hidden,
            E::NUM_ACTIONS,
            nn::LinearConfig::default(),
        );
        let seq = nn::seq()
            .add(hidden)
            .add_fn(|xs| xs.relu())
            .add(out)
            .add_fn(|xs| xs.softmax(1, Kind::Float));
        Self { seq, device }
    }
}

impl nn::Module for PolicyNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.to_device(self.device).apply(&self.seq)
    }
}

pub struct ReinforceTrainer<E: Env> {
    env: E,
    policy_net: PolicyNet,
    optimizer: COptimizer,
    returns: VecDeque<f32>,
    rewards: Vec<f32>,
    actions: Vec<i64>,
    states: Vec<Obs>,
    state: Obs,

    // gamma will be used in a better reward function.
    #[allow(unused)]
    gamma: f32,

    n_episodes: usize,
}

impl<E: Env> ReinforceTrainer<E> {
    pub fn new(env: E) -> Self {
        let policy_net = PolicyNet::new(&env, 20);
        let optimizer = nn::Adam::default().build_copt(1e-4).unwrap();
        let gamma = 0.99;
        let returns = VecDeque::with_capacity(100);

        let state = env.init();

        Self {
            env,
            policy_net,
            optimizer,
            returns,
            rewards: Vec::new(),
            actions: Vec::new(),
            states: Vec::new(),
            state,
            gamma,
            n_episodes: 0,
        }
    }
}

impl<E: Env> Trainer for ReinforceTrainer<E> {
    type Param = E::Param;

    fn train_one_step(&mut self, param: Self::Param) {
        // Calculate the probabilities of taking each action.
        let probs = self
            .policy_net
            .forward(&Tensor::of_slice(&self.state))
            .unsqueeze(0);
        let action = probs.multinomial(1, true).into();

        let step = self.env.step(action, param);
        self.states.push(step.obs);
        self.actions.push(action);
        self.rewards.push(step.reward);

        if step.is_done {
            // TODO: calculate rewards betterer
            let reward: f32 = self.rewards.iter().sum();
            let states = Tensor::of_slice2(&self.states);
            let actions = Tensor::of_slice(&self.actions);
            let probs = self.policy_net.forward(&states);

            let sampler = Categorical::from_probs(probs);
            let log_probs = -sampler.log_prob(&actions);
            let pseudo_loss = (log_probs * reward as f64).sum(Kind::Float);

            self.optimizer.zero_grad().unwrap();
            pseudo_loss.backward();
            self.optimizer.step().unwrap();

            let sum_rewards = self.rewards.iter().sum();
            self.returns.push_back(sum_rewards);
            let avg_return = self.returns.iter().sum::<f32>() / self.returns.len() as f32;
            info!(%self.n_episodes, %avg_return, "Episode complete.");
            self.n_episodes += 1;
        }
    }
}
