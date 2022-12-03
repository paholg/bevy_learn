use tch::{kind::FLOAT_CPU, nn, Device, Tensor};

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

    pub fn path(&self) -> nn::Path {
        self.vs.root()
    }
}

pub trait Trainer {
    fn pick_action(&mut self) -> i64;
    fn train(&mut self, step: Step);
}

#[derive(Debug)]
pub(crate) struct FrameStack {
    data: Tensor,
    nprocs: i64,
    nstack: i64,
}

impl FrameStack {
    pub fn new(nprocs: i64, nstack: i64, obs_space: &[i64]) -> FrameStack {
        if obs_space.len() != 2 {
            panic!("Currently only 2d observation spaces are supported for this trainer");
        }
        FrameStack {
            data: Tensor::zeros(&[nprocs, nstack, obs_space[0], obs_space[1]], FLOAT_CPU),
            nprocs,
            nstack,
        }
    }

    pub fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
        if let Some(masks) = masks {
            self.data *= masks.view([self.nprocs, 1, 1, 1])
        };
        let slice = |i| self.data.narrow(1, i, 1);
        for i in 1..self.nstack {
            slice(i - 1).copy_(&slice(i))
        }
        slice(self.nstack - 1).copy_(img);
        &self.data
    }
}
