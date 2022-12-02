//! Vanilla policy gradient
//! See https://arxiv.org/abs/1604.06778

use std::{fs::OpenOptions, io::Write};

use tch::{
    kind::FLOAT_CPU,
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};
use typed_builder::TypedBuilder;

use crate::{Env, Step, Trainer};

#[derive(Debug)]
struct Model {
    seq: nn::Sequential,
    device: Device,
}

impl Model {
    pub fn new(env: &Env, n_hidden: i64) -> Self {
        let num_observations = env.observation_space.iter().product();
        let device = env.vs.root().device();
        let hidden = nn::linear(
            env.vs.root() / "reinforce-linear-hidden",
            num_observations,
            n_hidden,
            nn::LinearConfig::default(),
        );
        let out = nn::linear(
            env.vs.root() / "reinforce-linear-out",
            n_hidden,
            env.num_actions,
            nn::LinearConfig::default(),
        );
        let seq = nn::seq().add(hidden).add_fn(|xs| xs.tanh()).add(out);
        Self { seq, device }
    }
}

impl nn::Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.to_device(self.device).apply(&self.seq)
    }
}

#[derive(TypedBuilder, Debug)]
pub struct ReinforceConfig {
    /// The learning rate of the optimizer.
    #[builder(default = 1e-4)]
    learning_rate: f64,

    /// The number of neurons in the hidden layer.
    #[builder(default = 32)]
    n_hidden: i64,
}

pub struct ReinforceTrainer {
    env: Env,
    model: Model,
    optimizer: nn::Optimizer,
    rewards: Vec<f32>,
    actions: Vec<i64>,
    observations: Vec<Tensor>,
    obs: Tensor,

    n_epochs: usize,
    n_steps: i64,
}

impl ReinforceTrainer {
    pub fn new(config: ReinforceConfig, env: Env, initial_obs: Tensor) -> Self {
        let model = Model::new(&env, config.n_hidden);
        let optimizer = nn::Adam::default()
            .build(&env.vs, config.learning_rate)
            .unwrap();
        let obs = initial_obs;

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("data.csv")
            .unwrap();
        write!(&mut file, "epoch,reward\n").unwrap();

        Self {
            env,
            model,
            optimizer,
            rewards: Vec::new(),
            actions: Vec::new(),
            observations: Vec::new(),
            obs,
            n_epochs: 0,
            n_steps: 0,
        }
    }
}

fn accumulate_rewards(mut rewards: Vec<f32>) -> Tensor {
    let mut acc_reward = 0.0;
    for reward in rewards.iter_mut().rev() {
        acc_reward += *reward;
        *reward = acc_reward;
    }
    Tensor::of_slice(&rewards)
}

impl Trainer for ReinforceTrainer {
    fn env(&self) -> &Env {
        &self.env
    }

    fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    fn pick_action(&mut self) -> i64 {
        let action = tch::no_grad(|| {
            self.obs
                .unsqueeze(0)
                .apply(&self.model)
                .softmax(1, Kind::Float)
                .multinomial(1, true)
        })
        .into();

        self.actions.push(action);
        action
    }

    fn train(&mut self, step: Step) {
        self.rewards.push(step.reward);
        self.observations.push(self.obs.shallow_clone());
        self.obs = step.obs;

        self.n_steps += 1;

        if let Some(initial_obs) = step.is_done {
            let sum_reward: f32 = self.rewards.iter().sum();
            let actions = Tensor::of_slice(&self.actions).unsqueeze(1);
            let rewards = accumulate_rewards(self.rewards.drain(..).collect());
            let action_mask =
                Tensor::zeros(&[self.n_steps, 2], FLOAT_CPU).scatter_value(1, &actions, 1.0);
            let logits = Tensor::stack(&self.observations, 0).apply(&self.model);
            let log_probs = (action_mask * logits.log_softmax(1, Kind::Float)).sum_dim_intlist(
                Some([1].as_ref()),
                false,
                Kind::Float,
            );
            let loss = -(rewards * log_probs).mean(Kind::Float);
            self.optimizer.backward_step(&loss);

            println!(
                "Epoch: {}, Return: {sum_reward:7.2}, steps: {}",
                self.n_epochs, self.n_steps
            );
            let mut file = OpenOptions::new().append(true).open("data.csv").unwrap();
            write!(&mut file, "{},{}\n", self.n_epochs, sum_reward).unwrap();

            self.n_epochs += 1;
            self.n_steps = 0;

            self.rewards = Vec::new();
            self.actions = Vec::new();
            self.observations = Vec::new();
            self.obs = initial_obs;
        }
    }
}
