use std::{fs::OpenOptions, io::Write};

use rand::seq::SliceRandom;
use tch::{
    nn::{self, OptimizerConfig},
    Device, IndexOp, Kind, Tensor,
};
use tch_distr::{Categorical, Distribution};
use typed_builder::TypedBuilder;

use crate::{ActorCriticModel, Env, Trainer};

fn layer_init(mut layer: nn::Linear, std: f64) -> nn::Linear {
    // TODO: fixme
    // nn::Init::Orthogonal { gain: std }.set(&mut layer.ws);
    // nn::Init::Const(0.0).set(layer.bs.as_mut().unwrap());
    layer
}

impl ActorCriticModel {
    pub fn new_linear(env: &Env) -> Self {
        let p = &env.path();
        let device = p.device();

        let seq = nn::seq()
            .add(layer_init(
                nn::linear(p / "l1", env.num_observations, 64, Default::default()),
                2f64.sqrt(),
            ))
            .add_fn(|xs| xs.tanh())
            .add(layer_init(
                nn::linear(p / "l2", 64, 64, Default::default()),
                2f64.sqrt(),
            ))
            .add_fn(|xs| xs.tanh());

        let critic = layer_init(nn::linear(p / "cl", 64, 1, Default::default()), 1.0);
        let actor = layer_init(
            nn::linear(p / "al", 64, env.num_actions, Default::default()),
            0.01,
        );

        Self {
            seq,
            critic,
            actor,
            device,
        }
    }
}

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
}

struct ActionValue {
    action: Tensor,
    logprob: Tensor,
    entropy: Tensor,
    value: Tensor,
}

impl ActionValue {
    fn get(model: &ActorCriticModel, xs: &Tensor, action: Option<Tensor>) -> Self {
        let ac = model.forward(xs);
        let probs = Categorical::from_logits(ac.actor);
        let action = match action {
            Some(a) => a,
            None => probs.sample(&[]).shallow_clone(),
        };
        let logprob = probs.log_prob(&action);

        Self {
            action,
            logprob,
            entropy: probs.entropy(),
            value: ac.critic,
        }
    }
}

pub struct PpoTrainer {
    env: Env,
    model: ActorCriticModel,
    optimizer: nn::Optimizer,

    config: PpoConfig,
    minibatch_size: i64,

    obs: Tensor,
    actions: Tensor,
    logprobs: Tensor,
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    global_step: i64,
    next_obs: Tensor,
    next_done: Tensor,

    just_reset: bool,

    step: i64,
    update: i64,
    episode: i64,
    episode_step: i64,
    sum_reward: f32,
}

impl PpoTrainer {
    pub fn new(config: PpoConfig, env: Env, initial_obs: &[f32]) -> Self {
        let minibatch_size = config.n_steps / config.n_minibatches;

        let model = ActorCriticModel::new_linear(&env);
        let optimizer = nn::Adam::default()
            .build(&env.vs, config.learning_rate)
            .unwrap();

        let device = env.path().device();

        let obs = Tensor::zeros(
            &[config.n_steps, env.num_observations],
            (Kind::Float, device),
        );

        let actions = Tensor::zeros(&[config.n_steps, 1], (Kind::Int64, device));
        let logprobs = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
        let rewards = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
        let dones = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));
        let values = Tensor::zeros(&[config.n_steps, 1], (Kind::Float, device));

        let global_step = 0;
        let next_obs = Tensor::of_slice(initial_obs).to(device);
        let next_done = Tensor::from(0.0);

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("data.csv")
            .unwrap();
        write!(&mut file, "epoch,reward,steps\n").unwrap();

        Self {
            env,
            model,
            optimizer,
            config,
            minibatch_size,
            obs,
            actions,
            logprobs,
            rewards,
            dones,
            values,
            global_step,
            next_obs,
            next_done,

            just_reset: false,

            step: 0,
            update: 0,
            episode: 0,
            episode_step: 0,
            sum_reward: 0.0,
        }
    }

    fn device(&self) -> Device {
        self.env.path().device()
    }
}

impl Trainer for PpoTrainer {
    fn pick_action(&mut self) -> i64 {
        self.global_step += 1;

        if self.just_reset {
            self.just_reset = false;
            self.next_done = Tensor::from(1.0);
        } else {
            self.next_done = Tensor::from(0.0);
        }

        self.obs.get(self.step).copy_(&self.next_obs);
        self.dones.get(self.step).copy_(&self.next_done);

        let av = tch::no_grad(|| {
            let av = ActionValue::get(&self.model, &self.next_obs, None);
            self.values.get(self.step).copy_(&av.value.flatten(0, -1));
            av
        });
        self.actions.get(self.step).copy_(&av.action);
        self.logprobs.get(self.step).copy_(&av.logprob);

        av.action.into()
    }

    fn train(&mut self, obs: &[f32], reward: f32) {
        self.next_obs = Tensor::of_slice(obs).to(self.device());
        self.sum_reward += reward;
        self.rewards
            .get(self.step)
            .copy_(&Tensor::from(reward).view(-1).to(self.device()));
        self.step += 1;
        self.episode_step += 1;

        if self.step >= self.config.n_steps {
            self.step = 0;
            let (returns, advantages) = tch::no_grad(|| {
                let next_value = self.model.forward(&self.next_obs).critic.reshape(&[1, -1]);
                let advantages = Tensor::zeros_like(&self.rewards).to(self.device());
                let mut lastgaelam = Tensor::from(0.0);

                for t in (0..self.config.n_steps).rev() {
                    let tmp;
                    let (nextnonterminal, nextvalues) = if t == self.config.n_steps - 1 {
                        (1.0 - &self.next_done, &next_value)
                    } else {
                        tmp = self.values.get(t + 1);
                        (1.0 - &self.dones.get(t + 1), &tmp)
                    };
                    let delta = self.rewards.get(t)
                        + self.config.gamma * nextvalues * &nextnonterminal
                        - self.values.get(t);

                    lastgaelam = delta
                        + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam;

                    advantages.get(t).copy_(&lastgaelam.squeeze());
                }
                let returns = &advantages + &self.values;
                (returns, advantages)
            });

            let b_obs = self.obs.reshape(&[-1, self.env.num_observations]);
            let b_logprobs = self.logprobs.reshape(&[-1]);
            let b_actions = self.actions.reshape(&[-1]);
            let b_advantages = advantages.reshape(&[-1]);
            let b_returns = returns.reshape(&[-1]);
            let b_values = self.values.reshape(&[-1]);

            let mut b_inds: Vec<i64> = (0..self.config.n_steps).collect();
            // let mut clipfracs = Vec::new();

            let mut rng = rand::thread_rng();
            for _epoch in 0..self.config.update_epochs {
                b_inds.shuffle(&mut rng);
                for start in (0..self.config.n_steps).step_by(self.minibatch_size as usize) {
                    let end = start + self.minibatch_size;
                    let mb_inds = &b_inds[start as usize..end as usize];

                    let av = ActionValue::get(
                        &self.model,
                        &b_obs.i(mb_inds),
                        Some(b_actions.i(mb_inds)),
                    );
                    let logratio = av.logprob - b_logprobs.i(mb_inds);
                    let ratio = logratio.exp();
                    // tch::no_grad(|| {
                    //     // let old_approx_kl = (-&logratio).mean(Kind::Float);
                    //     // let approx_kl = ((&ratio - 1) - logratio).mean(Kind::Float);
                    //     let clipfrac = (f64::from(&ratio) - 1.0).abs() > self.config.clip_coef;
                    //     clipfracs.push(clipfrac);
                    // });
                    let mut mb_advantages = b_advantages.i(mb_inds);
                    if self.config.norm_adv {
                        mb_advantages = (&mb_advantages - mb_advantages.mean(Kind::Float))
                            / (mb_advantages.std(true) + 1e-8);
                    }

                    let pg_loss1 = -&mb_advantages * &ratio;
                    let pg_loss2 = -&mb_advantages
                        * ratio.clamp(1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef);
                    let pg_loss = pg_loss1.maximum(&pg_loss2).mean(Kind::Float);

                    let newvalue = av.value.view(-1);
                    let v_loss = if self.config.clip_vloss {
                        let v_loss_unclipped =
                            (&newvalue - b_returns.i(mb_inds)).pow_tensor_scalar(2);
                        let v_clipped = b_values.i(mb_inds)
                            + (&newvalue - b_values.i(mb_inds))
                                .clamp(-self.config.clip_coef, self.config.clip_coef);
                        let v_loss_clipped =
                            (&v_clipped - b_returns.i(mb_inds)).pow_tensor_scalar(2);
                        let v_loss_max = v_loss_unclipped.maximum(&v_loss_clipped);
                        0.5 * v_loss_max.mean(Kind::Float)
                    } else {
                        0.5 * ((newvalue - b_returns.i(mb_inds)).pow_tensor_scalar(2))
                            .mean(Kind::Float)
                    };

                    let entropy_loss = av.entropy.mean(Kind::Float);
                    let loss: Tensor = pg_loss - self.config.ent_coef * entropy_loss
                        + v_loss * self.config.vf_coef;

                    self.optimizer.zero_grad();
                    loss.backward();
                    self.optimizer.clip_grad_norm(self.config.max_grad_norm);
                    self.optimizer.step();
                }
            }
            self.update += 1;
        }
    }

    fn reset(&mut self, _obs: &[f32]) {
        self.just_reset = true;

        println!(
            "Episode: {}, Return: {:7.2}, steps: {}",
            self.episode, self.sum_reward, self.episode_step
        );
        let mut file = OpenOptions::new().append(true).open("data.csv").unwrap();
        write!(
            &mut file,
            "{},{},{}\n",
            self.episode, self.sum_reward, self.episode_step
        )
        .unwrap();

        self.episode += 1;
        self.episode_step = 0;
        self.sum_reward = 0.0;
    }
}
