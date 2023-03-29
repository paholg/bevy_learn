#![doc = include_str!("../README.md")]

pub mod ppo;
pub mod reinforce;
mod save;

pub use reinforce::{ReinforceConfig, ReinforceTrainer};
// pub use ppo::{PpoConfig, PpoTrainer};

pub trait Trainer<const OBS: usize> {
    fn pick_action(&mut self) -> i64;
    fn train(&mut self, obs: [f32; OBS], reward: f32);
    fn reset(&mut self, obs: [f32; OBS]);
}
