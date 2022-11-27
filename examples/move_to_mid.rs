//! A very basic example where the AI is rewarded for moving to the center of
//! a 1d grid.

use bevy::{
    ecs::system::BoxedSystem,
    prelude::{
        default, shape, App, Assets, Camera2dBundle, Color, Commands, Component, Mesh, NonSendMut,
        Query, ResMut, System, Transform, Vec2, Vec3, With,
    },
    sprite::{ColorMaterial, MaterialMesh2dBundle, Sprite, SpriteBundle},
    DefaultPlugins,
};
use bevy_learn::{reinforce::ReinforceTrainer, Env, Trainer};
use tch::nn;

const GRID_SIZE: f32 = 1000.0;
const START_X: f32 = -GRID_SIZE * 0.5;

#[derive(Component)]
struct Ai;

fn main() {
    // Ai
    let env = MoveEnv::new();
    // let trainer = ReinforceTrainer::new(env);
    // let system: BoxedSystem = Box::new(train_one_step);
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        // .insert_non_send_resource(trainer)
        .add_system(train_one_step)
        .run();
}

fn train_one_step<'w, 's, 'a>(
    // mut trainer: NonSendMut<ReinforceTrainer<MoveEnv>>,
    query: Query<'w, 's, &'a mut Transform, With<Ai>>,
) {
    // trainer.train_one_step(query)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    // Grid
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(1.0, 0.4, 0.4),
            custom_size: Some(Vec2::new(GRID_SIZE, GRID_SIZE)),
            ..default()
        },
        ..default()
    });

    // Entity
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(20.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::PURPLE)),
            transform: Transform::from_translation(Vec3::new(START_X, 0.0, 1.0)),
            ..default()
        },
        Ai,
    ));
}

struct MoveEnv {
    vs: nn::VarStore,
}

impl MoveEnv {
    fn new() -> Self {
        let device = tch::Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        Self { vs }
    }
}

const ACTIONS: [Vec2; 2] = [Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0)];

impl Env for MoveEnv {
    type Param<'w, 's, 'a> = Query<'w, 's, &'a mut Transform, With<Ai>>;

    const NUM_ACTIONS: i64 = ACTIONS.len() as i64;

    const NUM_OBSERVATIONS: i64 = 1;

    fn path(&self) -> tch::nn::Path {
        self.vs.root()
    }

    fn init(&self) -> bevy_learn::Obs {
        vec![START_X]
    }

    fn step<'w, 's, 'a>(
        &mut self,
        action: i64,
        mut param: Self::Param<'w, 's, 'a>,
    ) -> bevy_learn::Step {
        let mut transform = param.single_mut();
        let action = ACTIONS[action as usize];
        transform.translation += action.extend(0.0);

        if transform.translation.x > GRID_SIZE * 0.5 {
            transform.translation.x = GRID_SIZE * 0.5
        } else if transform.translation.x < -GRID_SIZE * 0.5 {
            transform.translation.x = -GRID_SIZE * 0.5
        }

        bevy_learn::Step {
            obs: vec![transform.translation.x],
            reward: transform.translation.x.abs() / (GRID_SIZE * 0.5),
            is_done: true,
        }
    }
}
