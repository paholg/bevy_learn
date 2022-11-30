use bevy::prelude::NonSendMut;
use tch::{nn, Tensor};

pub mod reinforce;

#[derive(Debug)]
pub struct Step {
    /// Observation: The state of the world as exposed to the AI.
    pub obs: Tensor,
    /// Reward for the AI for this step. Can be negative to be a punishment.
    pub reward: f32,
    pub is_done: bool,
}

impl Step {
    fn copy_with_obs(&self, obs: &Tensor) -> Self {
        Self {
            obs: obs.copy(),
            reward: self.reward.clone(),
            is_done: self.is_done.clone(),
        }
    }
}

pub trait Env {
    type Param<'w, 's>;

    const NUM_ACTIONS: i64;

    const OBSERVATION_SPACE: &'static [i64];

    fn vs(&self) -> &nn::VarStore;

    fn path(&self) -> nn::Path;

    /// Return the initial observation of the world state.
    fn init(&self) -> Tensor;

    /// Reset the environment, returning the observation of the world state.
    fn reset<'w, 's>(&mut self, param: &mut Self::Param<'w, 's>) -> Tensor;

    fn step<'w, 's>(&mut self, action: i64, param: &mut Self::Param<'w, 's>) -> Step;
}

pub trait Trainer<E: Env> {
    fn train_one_step<'w, 's>(&mut self, param: E::Param<'w, 's>);
}

pub fn train_one_step<'w, 's, T: Trainer<E>, E: Env>(
    mut trainer: NonSendMut<T>,
    param: E::Param<'w, 's>,
) {
    trainer.train_one_step(param)
}

pub trait Sampler {
    type Param<'w, 's>;

    fn sample_one_step<'w, 's>(&mut self, param: Self::Param<'w, 's>);
}
