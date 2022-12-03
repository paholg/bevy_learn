use tch::{nn, Device, Tensor};

pub mod reinforce;

#[derive(Debug)]
pub struct Step {
    /// Observation: The state of the world as exposed to the AI.
    pub obs: Tensor,
    /// Reward for the AI for this step. Can be negative to be a punishment.
    pub reward: f32,
    /// If done, indicates the starting observation for the next epoch.
    pub is_done: Option<Tensor>,
}

impl Step {
    fn copy_with_obs(&self, obs: &Tensor) -> Self {
        Self {
            obs: obs.copy(),
            reward: self.reward.clone(),
            is_done: self.is_done.as_ref().map(|t| t.copy()),
        }
    }
}

pub struct Env {
    num_actions: i64,
    observation_space: Vec<i64>,
    vs: nn::VarStore,
}

impl Env {
    pub fn new(num_actions: i64, observation_space: impl Into<Vec<i64>>, device: Device) -> Self {
        Self {
            num_actions,
            observation_space: observation_space.into(),
            vs: nn::VarStore::new(device),
        }
    }
}

pub trait Trainer {
    fn pick_action(&mut self) -> i64;
    fn train(&mut self, step: Step);
}
