use tch::nn;

pub mod reinforce;

/// The type used for the observation of the state of the world.
/// It is currently a 1-d vector, but will be a Tensor in the future.
pub type Obs = Vec<f32>;

pub struct Step {
    /// Observation: The state of the world as exposed to the AI.
    pub obs: Obs,
    /// Reward for the AI for this step. Can be negative to be a punishment.
    pub reward: f32,
    pub is_done: bool,
}

pub trait Env {
    type Param<'w, 's>;

    const NUM_ACTIONS: i64;

    // TODO: Allow observation_space to be more than one dimensional.
    const NUM_OBSERVATIONS: i64;

    fn path(&self) -> nn::Path;

    /// Return the initial observation of the world state.
    fn init(&self) -> Obs;

    /// Reset the environment, returning the observation of the world state.
    fn reset<'w, 's>(&mut self, param: &mut Self::Param<'w, 's>) -> Obs;

    fn step<'w, 's>(&mut self, action: i64, param: &mut Self::Param<'w, 's>) -> Step;
}

pub trait Trainer {
    type Param<'w, 's>;
    fn train_one_step<'w, 's>(&mut self, param: Self::Param<'w, 's>);
}

// pub trait Sampler {
//     type Param<'w, 's>;
//     fn sample_one_step<'w, 's>(&mut self, param: Self::Param<'w, 's>);
// }
