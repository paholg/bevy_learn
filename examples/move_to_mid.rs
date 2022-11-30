//! A very basic example where the AI is rewarded for moving to the center of
//! a 1d grid.

use bevy::{
    app::ScheduleRunnerSettings,
    prelude::{
        default, shape, App, Assets, Camera2dBundle, Color, Commands, Component, Mesh,
        OrthographicProjection, PluginGroup, Query, ResMut, Transform, Vec2, Vec3, With,
    },
    sprite::{ColorMaterial, MaterialMesh2dBundle, Sprite, SpriteBundle},
    window::{PresentMode, WindowDescriptor, WindowPlugin},
    DefaultPlugins, MinimalPlugins,
};
use bevy_learn::{reinforce::ReinforceTrainer, train_one_step, Env};
use clap::Parser;
use tch::nn;

const GRID_SIZE: f32 = 10.0;
const START_X: f32 = 0.0;
const START_Y: f32 = 0.0;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    graphics: bool,
}

#[derive(Component)]
struct Ai;

fn main() {
    let args = Args::parse();
    // Ai
    let env = MoveEnv::new();
    let trainer = ReinforceTrainer::new(env);

    let mut app = App::new();

    if args.graphics {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            },
            ..default()
        }))
        .add_startup_system(setup_graphics);
    } else {
        app.insert_resource(ScheduleRunnerSettings {
            run_mode: bevy::app::RunMode::Loop { wait: None },
        })
        .add_plugins(MinimalPlugins)
        .add_startup_system(setup);
    }
    app.insert_non_send_resource(trainer)
        .add_system(train_one_step::<ReinforceTrainer<MoveEnv>, MoveEnv>)
        .run();
}

fn setup(mut commands: Commands) {
    // Entity
    commands.spawn((
        Transform::from_translation(Vec3::new(START_X, START_Y, 2.0)),
        Ai,
    ));
}

fn setup_graphics(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle {
        projection: OrthographicProjection {
            scale: 0.01,
            ..default()
        },
        ..default()
    });

    // Grid
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.4, 0.4, 1.0),
            custom_size: Some(Vec2::new(GRID_SIZE, GRID_SIZE)),
            ..default()
        },
        ..default()
    });

    // Entity
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::PURPLE)),
            transform: Transform::from_translation(Vec3::new(START_X, START_Y, 2.0)),
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
    type Param<'w, 's> = Query<'w, 's, &'static mut Transform, With<Ai>>;

    const NUM_ACTIONS: i64 = ACTIONS.len() as i64;

    const NUM_OBSERVATIONS: i64 = 1;

    fn vs(&self) -> &tch::nn::VarStore {
        &self.vs
    }

    fn path(&self) -> tch::nn::Path {
        self.vs.root()
    }

    fn init(&self) -> bevy_learn::Obs {
        vec![START_X]
    }

    fn step<'w, 's>(&mut self, action: i64, param: &mut Self::Param<'w, 's>) -> bevy_learn::Step {
        let mut transform = param.single_mut();
        let action = ACTIONS[action as usize];
        transform.translation += action.extend(0.0);

        let is_done = transform.translation.x.abs() > GRID_SIZE * 0.5;
        let reward = if transform.translation.x > GRID_SIZE * 0.5 {
            1.0
        } else if transform.translation.x < -GRID_SIZE * 0.5 {
            -1.0
        } else {
            0.0
        };

        bevy_learn::Step {
            obs: vec![transform.translation.x / (GRID_SIZE * 0.5)],
            reward,
            is_done,
        }
    }

    fn reset<'w, 's>(&mut self, param: &mut Self::Param<'w, 's>) -> bevy_learn::Obs {
        let mut transform = param.single_mut();
        transform.translation.x = START_X;
        transform.translation.y = START_Y;

        vec![transform.translation.x / (GRID_SIZE * 0.5)]
    }
}
