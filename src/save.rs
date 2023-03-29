use std::{fs::OpenOptions, io::Write, path::PathBuf};

pub struct Saver {
    path: Option<PathBuf>,
    initialized: bool,
}

impl Saver {
    pub fn init(path: Option<PathBuf>) -> Self {
        let initialized = if let Some(p) = &path {
            let result = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .and_then(|mut file| write!(&mut file, "episode,reward,steps\n"));
            match result {
                Ok(_) => true,
                Err(error) => {
                    tracing::error!(
                        path = p.display().to_string(),
                        ?error,
                        "Could not initialize output file. Data will not be saved."
                    );
                    false
                }
            }
        } else {
            false
        };

        Saver { path, initialized }
    }

    pub fn save_line(&self, n_episodes: usize, sum_reward: f32, n_steps: usize) {
        match (&self.path, self.initialized) {
            (Some(p), true) => {
                let result = OpenOptions::new()
                    .append(true)
                    .open(p)
                    .and_then(|mut file| {
                        write!(&mut file, "{},{},{}\n", n_episodes, sum_reward, n_steps)
                    });
                if let Err(error) = result {
                    tracing::error!(
                        path = p.display().to_string(),
                        ?error,
                        "Could not write data line."
                    )
                }
            }
            _ => (),
        }
    }
}
