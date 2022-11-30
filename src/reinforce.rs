//! Vanilla policy gradient
//! See https://arxiv.org/abs/1604.06778

use std::{fs::OpenOptions, io::Write};

use tch::{
    nn::{self, Module, OptimizerConfig},
    COptimizer, Device, Kind, Tensor,
};
use tch_distr::{Categorical, Distribution};

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
            env.path() / "reinforce-lout",
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
    rewards: Vec<f32>,
    actions: Vec<i64>,
    states: Vec<Obs>,
    state: Obs,

    gamma: f32,

    n_episodes: usize,
    n_steps: usize,
}

impl<E: Env> ReinforceTrainer<E> {
    pub fn new(env: E) -> Self {
        let policy_net = PolicyNet::new(&env, 20);
        let optimizer = nn::Adam::default().build_copt(1e-4).unwrap();
        let gamma = 0.99;

        let state = env.init();

        Self {
            env,
            policy_net,
            optimizer,
            rewards: Vec::new(),
            actions: Vec::new(),
            states: Vec::new(),
            state,
            gamma,
            n_episodes: 0,
            n_steps: 0,
        }
    }
}

impl<E: Env> Trainer<E> for ReinforceTrainer<E> {
    fn train_one_step<'w, 's>(&mut self, mut param: E::Param<'w, 's>) {
        if self.n_episodes == 0 && self.n_steps == 0 {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("data.csv")
                .unwrap();
            write!(&mut file, "episode,reward\n").unwrap();
        }
        self.n_steps += 1;
        // Calculate the probabilities of taking each action.
        let state = Tensor::of_slice(&self.state).unsqueeze(0);
        let probs = self.policy_net.forward(&state);
        let sampler = Categorical::from_probs(probs);
        let action = sampler.sample(&[]).into();

        let step = self.env.step(action, &mut param);
        self.states.push(self.state.clone());
        self.actions.push(action);
        self.rewards.push(step.reward);
        self.state = step.obs;

        if step.is_done {
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

            self.n_episodes += 1;

            self.rewards = Vec::new();
            self.actions = Vec::new();
            self.states = Vec::new();
            self.state = self.env.reset(&mut param);

            println!(
                "Episode: {}, Return: {reward:7.2}, steps: {}",
                self.n_episodes, self.n_steps
            );
            let mut file = OpenOptions::new().append(true).open("data.csv").unwrap();
            write!(&mut file, "{},{}\n", self.n_episodes, reward).unwrap();
            self.n_steps = 0;
        }
    }
}
