//! A very basic example where the AI is rewarded for moving to the center of
//! a 1d grid.

use bevy::{
    app::ScheduleRunnerSettings,
    prelude::{
        default, shape, App, Assets, Camera2dBundle, Color, Commands, Component, Mesh, NonSendMut,
        OrthographicProjection, PluginGroup, Query, ResMut, Transform, Vec2, Vec3, With, Without,
    },
    render::camera::ScalingMode,
    sprite::{ColorMaterial, MaterialMesh2dBundle, Sprite, SpriteBundle},
    window::{PresentMode, WindowDescriptor, WindowPlugin},
    DefaultPlugins, MinimalPlugins,
};
use bevy_learn::{
    ppo::{PpoConfig, PpoTrainer},
    Env, Trainer,
};
use clap::Parser;
use rand::Rng;

const GRID_SIZE: f32 = 50.0;
const MAX: f32 = GRID_SIZE * 0.5;

const TARGET_SIZE: f32 = 1.0;

const START_X: f32 = -20.0;
const START_Y: f32 = -20.0;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    graphics: bool,
}

#[derive(Component)]
struct Ai;

#[derive(Component)]
struct Target;

fn main() {
    let args = Args::parse();
    // Ai
    let device = tch::Device::cuda_if_available();
    let env = Env::new(ACTIONS.len() as i64, 4, device);
    let trainer = PpoTrainer::new(
        PpoConfig::builder().build(),
        env,
        &[START_X / MAX, START_Y / MAX, 0.0, 0.0],
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
        .add_system(ai_reset)
        .run();
}

fn setup(mut commands: Commands) {
    // Entity
    commands.spawn((
        Transform::from_translation(Vec3::new(START_X, START_Y, 2.0)),
        Ai,
    ));

    // Target
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
        Target,
    ));
}

fn setup_graphics(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle {
        projection: OrthographicProjection {
            scaling_mode: ScalingMode::Auto {
                min_width: GRID_SIZE,
                min_height: GRID_SIZE,
            },
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

    // Target
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(TARGET_SIZE).into()).into(),
            material: materials.add(ColorMaterial::from(Color::GREEN)),
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            ..default()
        },
        Target,
    ));

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

const ACTIONS: [Vec2; 5] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(0.0, 1.0),
];

fn out_of_bounds(ai: &Transform) -> bool {
    ai.translation.x.abs() > MAX || ai.translation.y.abs() > MAX
}

fn hit(ai: &Transform, target: &Transform) -> bool {
    let n2 = (ai.translation.truncate() - target.translation.truncate()).length_squared();

    n2 < TARGET_SIZE * TARGET_SIZE
}

fn ai_act(
    mut trainer: NonSendMut<PpoTrainer>,
    mut ai: Query<&mut Transform, With<Ai>>,
    target: Query<&Transform, (With<Target>, Without<Ai>)>,
) {
    let action_id = trainer.pick_action();
    let action = ACTIONS[action_id as usize];
    let mut transform = ai.get_single_mut().unwrap();
    let target = target.get_single().unwrap();
    transform.translation += action.extend(0.0);

    let reward = if out_of_bounds(&transform) {
        -1.0
    } else if hit(&transform, &target) {
        1.0
    } else {
        0.0
    };

    let obs = &[
        transform.translation.x / MAX,
        transform.translation.y / MAX,
        target.translation.x / MAX,
        target.translation.y / MAX,
    ];

    trainer.train(obs, reward);
}

fn ai_reset(
    mut trainer: NonSendMut<PpoTrainer>,
    mut ai: Query<&mut Transform, With<Ai>>,
    mut target: Query<&mut Transform, (With<Target>, Without<Ai>)>,
) {
    let mut target = target.get_single_mut().unwrap();
    let mut transform = ai.get_single_mut().unwrap();
    if out_of_bounds(&transform) || hit(&transform, &target) {
        let mut rng = rand::thread_rng();
        transform.translation.x = rng.gen::<f32>() * GRID_SIZE - MAX;
        transform.translation.y = rng.gen::<f32>() * GRID_SIZE - MAX;
        target.translation.x = rng.gen::<f32>() * GRID_SIZE - MAX;
        target.translation.y = rng.gen::<f32>() * GRID_SIZE - MAX;

        let obs = &[
            transform.translation.x / MAX,
            transform.translation.y / MAX,
            target.translation.x / MAX,
            target.translation.y / MAX,
        ];

        trainer.reset(obs);
    }
}
