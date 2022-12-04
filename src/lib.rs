use tch::{nn, Device, Tensor};

pub mod ppo;
pub mod reinforce;

pub struct Env {
    num_actions: i64,
    num_observations: i64,
    vs: nn::VarStore,
}

impl Env {
    pub fn new(num_actions: i64, num_observations: i64, device: Device) -> Self {
        Self {
            num_actions,
            num_observations,
            vs: nn::VarStore::new(device),
        }
    }

    pub fn path(&self) -> nn::Path {
        self.vs.root()
    }
}

pub trait Trainer {
    fn pick_action(&mut self) -> i64;
    fn train(&mut self, obs: &[f32], reward: f32);
    fn reset(&mut self, obs: &[f32]);
}

#[derive(Debug)]
pub struct Model {
    seq: nn::Sequential,
    device: Device,
}

impl Model {
    pub fn new_custom(seq: nn::Sequential, device: Device) -> Self {
        Self { seq, device }
    }
}

impl nn::Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.to_device(self.device).apply(&self.seq)
    }
}

#[derive(Debug)]
pub struct ActorCriticModel {
    seq: nn::Sequential,
    critic: nn::Linear,
    actor: nn::Linear,
    device: Device,
}

impl ActorCriticModel {
    pub fn forward(&self, xs: &Tensor) -> ActorCritic {
        let xs = xs.to_device(self.device).apply(&self.seq);
        ActorCritic {
            critic: xs.apply(&self.critic),
            actor: xs.apply(&self.actor),
        }
    }
}

pub struct ActorCritic {
    pub actor: Tensor,
    pub critic: Tensor,
}
