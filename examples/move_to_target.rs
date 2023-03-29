//! A very basic example where the AI is rewarded for moving to the center of
//! a 1d grid.

use bevy::{
    app::ScheduleRunnerSettings,
    prelude::{
        default, shape, App, Assets, Camera2dBundle, Color, Commands, Component, Mesh, NonSendMut,
        OrthographicProjection, PluginGroup, Query, ResMut, Transform, Vec2, With, Without,
    },
    render::camera::ScalingMode,
    sprite::{ColorMaterial, MaterialMesh2dBundle, Sprite, SpriteBundle},
    window::{PresentMode, WindowDescriptor, WindowPlugin},
    DefaultPlugins, MinimalPlugins,
};
use clap::Parser;
use learnit::{
    ppo::{PpoConfig, PpoTrainer},
    Env, Trainer,
};
use rand::Rng;

const GRID_SIZE: f32 = 50.0;
const MAX: f32 = GRID_SIZE * 0.5;

const TARGET_SIZE: f32 = 1.0;

const START: Transform = Transform::from_xyz(-20.0, -20.0, 2.0);
const TARGET_START: Transform = Transform::from_xyz(0.0, 0.0, 1.0);

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
    let obs = obs(&START, &TARGET_START);
    let env = Env::new(ACTIONS.len() as i64, obs.len() as i64, device);
    let trainer = PpoTrainer::new(PpoConfig::builder().build(), env, &obs);

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
    commands.spawn((START, Ai));

    // Target
    commands.spawn((TARGET_START, Target));
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
            transform: TARGET_START,
            ..default()
        },
        Target,
    ));

    // Entity
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::PURPLE)),
            transform: START,
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

fn obs(ai: &Transform, target: &Transform) -> [f32; 6] {
    [
        ai.translation.x - target.translation.x,
        ai.translation.y - target.translation.y,
        0.0,
        0.0,
        0.0,
        0.0,
        // MAX - ai.translation.x,
        // ai.translation.x + MAX,
        // MAX - ai.translation.y,
        // ai.translation.y + MAX,
        // (MAX - ai.translation.x).recip(),
        // (ai.translation.x + MAX).recip(),
        // (MAX - ai.translation.y).recip(),
        // (ai.translation.y + MAX).recip(),
        // (ai.translation.x - target.translation.x).recip(),
        // (ai.translation.y - target.translation.y).recip(),
    ]
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
        -0.1
    } else if hit(&transform, &target) {
        1.0
    } else {
        -0.001
    };

    let obs = obs(&transform, target);
    trainer.train(&obs, reward);
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

        let obs = obs(&transform, &target);

        trainer.reset(&obs);
    }
}
