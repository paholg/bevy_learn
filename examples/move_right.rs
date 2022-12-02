//! A very basic example where the AI is rewarded for moving to the center of
//! a 1d grid.

use bevy::{
    app::ScheduleRunnerSettings,
    prelude::{
        default, shape, App, Assets, Camera2dBundle, Color, Commands, Component, Mesh, NonSendMut,
        OrthographicProjection, PluginGroup, Query, ResMut, Transform, Vec2, Vec3, With,
    },
    sprite::{ColorMaterial, MaterialMesh2dBundle, Sprite, SpriteBundle},
    window::{PresentMode, WindowDescriptor, WindowPlugin},
    DefaultPlugins, MinimalPlugins,
};
use bevy_learn::{
    reinforce::{ReinforceConfig, ReinforceTrainer},
    Env, Step, Trainer,
};
use clap::Parser;
use tch::{nn, Tensor};

const GRID_SIZE: f32 = 50.0;
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
    let device = tch::Device::cuda_if_available();
    let env = Env::new(ACTIONS.len() as i64, vec![1], device);
    let trainer = ReinforceTrainer::new(
        ReinforceConfig::builder().build(),
        env,
        Tensor::of_slice(&[START_X]),
    );

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
        .add_system(ai_act)
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

const ACTIONS: [Vec2; 2] = [Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0)];

fn ai_act(mut trainer: NonSendMut<ReinforceTrainer>, mut ai: Query<&mut Transform, With<Ai>>) {
    let action_id = trainer.pick_action();
    let action = ACTIONS[action_id as usize];
    let mut transform = ai.get_single_mut().unwrap();
    transform.translation += action.extend(0.0);

    let reward = if transform.translation.x > GRID_SIZE * 0.5 {
        1.0
    } else if transform.translation.x < -GRID_SIZE * 0.5 {
        -1.0
    } else {
        0.0
    };

    let is_done = if transform.translation.x.abs() > GRID_SIZE * 0.5 {
        transform.translation.x = START_X;
        Some(Tensor::of_slice(&[transform.translation.x]))
    } else {
        None
    };

    let step = Step {
        obs: Tensor::of_slice(&[transform.translation.x / (GRID_SIZE * 0.5)]),
        reward,
        is_done,
    };

    trainer.train(step);
}
