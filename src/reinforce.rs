//! Vanilla policy gradient
//! See <https://arxiv.org/abs/1604.06778>

use std::path::PathBuf;

use dfdx::{
    optim::{Adam, AdamConfig},
    prelude::{Linear, ModuleBuilder, ModuleMut, OwnedTape, ReLU, ResetParams, Softmax},
    shapes::{Axis, Rank1},
    tensor::{AsArray, Cpu, DeviceStorage, Tensor, TensorFromArray},
    tensor_ops::Device,
};
use rand::{distributions::WeightedIndex, prelude::Distribution};
use typed_builder::TypedBuilder;

use crate::{save::Saver, Trainer};

type Model<const OBS: usize, const ACT: usize> = ((Linear<OBS, 128>, ReLU), Linear<128, ACT>);

type Observation<const OBS: usize, DEV> = Tensor<Rank1<OBS>, f32, DEV, OwnedTape<DEV>>;

#[derive(TypedBuilder, Debug)]
pub struct ReinforceConfig {
    /// The learning rate of the optimizer.
    #[builder(default = 2e-4)]
    learning_rate: f32,

    #[builder(default = 0.98)]
    gamma: f32,

    /// If set, the path where progress data will be stored as CSV.
    #[builder(default, setter(strip_option))]
    save_progress: Option<PathBuf>,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

pub struct ReinforceTrainer<const OBS: usize, const ACT: usize, DEV: Device<f32>> {
    device: DEV,
    model: Model<OBS, ACT>,
    optimizer: Adam<Model<OBS, ACT>>,
    rewards: Vec<f32>,
    actions: Vec<i64>,
    observations: Vec<[f32; OBS]>,
    obs: [f32; OBS],

    n_episodes: usize,
    n_steps: usize,
    saver: Saver,
}

impl<const OBS: usize, const ACT: usize, DEV> ReinforceTrainer<OBS, ACT, DEV>
where
    DEV: Device<f32> + TensorFromArray<[f32; OBS], Rank1<OBS>, f32>,
{
    pub fn new(config: ReinforceConfig, initial_obs: [f32; OBS]) -> Self
    where
        Model<OBS, ACT>: ResetParams<DEV, f32>,
    {
        let device = DEV::default();
        let model = device.build_module();
        let mut opt_config = AdamConfig::default();
        opt_config.lr = config.learning_rate;
        let optimizer = Adam::new(opt_config);

        let saver = Saver::init(config.save_progress);

        Self {
            saver,
            device,
            model,
            optimizer,
            rewards: Vec::new(),
            actions: Vec::new(),
            observations: Vec::new(),
            obs: initial_obs,
            n_episodes: 0,
            n_steps: 0,
        }
    }
}

impl<const OBS: usize, const ACT: usize> Trainer<OBS> for ReinforceTrainer<OBS, ACT, Cpu> {
    fn pick_action(&mut self) -> i64 {
        let obs = self.device.tensor(self.obs).traced();
        let weights = self.model.forward_mut(obs).softmax::<Axis<0>>();
        let dist = WeightedIndex::new(weights.array()).unwrap();
        let mut rng = rand::thread_rng();
        let action = dist.sample(&mut rng) as i64;

        self.actions.push(action);
        action
    }

    fn train(&mut self, mut obs: [f32; OBS], reward: f32) {
        // self.rewards.push(reward);

        // std::mem::swap(&mut self.obs, &mut obs);
        // self.observations.push(obs);

        // self.n_steps += 1;
    }

    fn reset(&mut self, obs: [f32; OBS]) {}
}

fn categorical<const OBS: usize, DEV: Device<f32>>(obs: &Observation<OBS, DEV>) -> [f32; OBS] {
    todo!()
}
