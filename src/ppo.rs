use std::path::PathBuf;

use dfdx::{
    prelude::{Linear, Tanh},
    tensor_ops::Device,
};
use typed_builder::TypedBuilder;

use crate::Trainer;

#[derive(TypedBuilder, Debug)]
pub struct PpoConfig {
    #[builder(default = 1e-4)]
    learning_rate: f64,

    #[builder(default = 128)]
    n_steps: i64,

    #[builder(default = 0.99)]
    gamma: f64,

    #[builder(default = 0.95)]
    gae_lambda: f64,

    #[builder(default = 4)]
    n_minibatches: i64,

    #[builder(default = 4)]
    update_epochs: i64,

    #[builder(default = true)]
    norm_adv: bool,

    #[builder(default = 0.2)]
    clip_coef: f64,

    #[builder(default = true)]
    clip_vloss: bool,

    #[builder(default = 0.01)]
    ent_coef: f64,

    #[builder(default = 0.5)]
    vf_coef: f64,

    #[builder(default = 0.5)]
    max_grad_norm: f64,

    #[builder(default, setter(strip_option))]
    save_progress: Option<PathBuf>,
}

type Critic<const OBS: usize> = (
    (Linear<OBS, 64>, Tanh),
    (Linear<64, 64>, Tanh),
    Linear<64, 1>,
);

type Actor<const OBS: usize, const ACT: usize> = (
    (Linear<OBS, 64>, Tanh),
    (Linear<64, 64>, Tanh),
    Linear<64, ACT>,
);

// pub struct PpoTrainer<const OBS: usize, const ACT: usize, DEV: Device<f32>> {
//     device: DEV,
//     model: Model<OBS, ACT>,
//     optimizer: Adam<Model<OBS, ACT>>,

//     config: PpoConfig,
//     minibatch_size: usize,

//     obs: Tensor,
//     actions: Tensor,
//     logprobs: Tensor,
//     rewards: Tensor,
//     dones: Tensor,
//     values: Tensor,
//     global_step: i64,
//     next_obs: Tensor,
//     next_done: Tensor,

//     just_reset: bool,

//     step: usize,
//     update: usize,
//     episode: usize,
//     episode_step: usize,
//     sum_reward: f32,
// }

// impl<const OBS: usize, const ACT: usize, DEV> PpoTrainer<OBS, ACT, DEV>
// where
//     DEV: Device<f32> + TensorFromArray<[f32; OBS], Rank1<OBS>, f32>,
// {
//     pub fn new(config: PpoConfig, env: Env, initial_obs: &[f32]) -> Self {
//         let minibatch_size = config.n_steps / config.n_minibatches;

//         let model = ActorCriticModel::new_linear(&env);
//         let optimizer = nn::Adam::default()
//             .build(&env.vs, config.learning_rate)
//             .unwrap();

//         let device = env.path().device();

//         let obs = Tensor::zeros(
//             &[config.n_steps, env.num_observations],
//             (Kind::Float, device),
//         );

//         let actions = Tensor::zeros(&[config.n_steps, 1], (Kind::Int64, device));
//         let logprobs = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
//         let rewards = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
//         let dones = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
//         let values = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));

//         let global_step = 0;
//         let next_obs = Tensor::of_slice(initial_obs).to(device);
//         let next_done = Tensor::from(0.0);

//         Self {
//             env,
//             model,
//             optimizer,
//             config,
//             minibatch_size,
//             obs,
//             actions,
//             logprobs,
//             rewards,
//             dones,
//             values,
//             global_step,
//             next_obs,
//             next_done,

//             just_reset: false,

//             step: 0,
//             update: 0,
//             episode: 0,
//             episode_step: 0,
//             sum_reward: 0.0,
//         }
//     }
// }

// impl Trainer for PpoTrainer {}
