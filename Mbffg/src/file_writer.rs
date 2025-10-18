#![allow(dead_code)]
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct PathLike {
    path: PathBuf,
}

impl PathLike {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        PathLike { path: path.into() }
    }

    pub fn stem(&self) -> Option<String> {
        self.path
            .file_stem()
            .and_then(|s| Some(s.to_string_lossy().into_owned()))
    }

    pub fn parent(&self) -> Option<&Path> {
        self.path.parent()
    }

    pub fn to_string(&self) -> String {
        self.path.to_str().unwrap_or("").to_string()
    }

    pub fn with_extension(&self, ext: &str) -> PathLike {
        let mut new_path = self.path.clone();
        if new_path.set_extension(ext) {
            PathLike { path: new_path }
        } else {
            panic!("Failed to set the extension of the path.");
        }
    }

    /// Create all parent directories if they do not exist.
    pub fn create_dir_all(&self) -> std::io::Result<()> {
        if let Some(parent) = self.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct FileWriter {
    // The file handle, protected by a standard library Mutex for thread-safe access.
    // Arc is used to allow sharing this Mutex across multiple threads if needed.
    file: Arc<Mutex<std::fs::File>>,
    path: String, // Keep path for logging/information
}

impl FileWriter {
    /// Creates a new `FileWriter` instance.
    /// If the file at the given path exists, it will be removed before creation.
    /// This method performs synchronous file operations and will block.
    pub fn new(path: impl Into<String>) -> Self {
        let path_str = path.into();

        PathLike::new(&path_str).create_dir_all().unwrap();

        // Remove the file if it exists (synchronously, as this is setup)
        if Path::new(&path_str).exists() {
            std::fs::remove_file(&path_str).unwrap_or_else(|e| {
                eprintln!("Failed to remove file {}: {}", path_str, e);
            });
        }

        let file = File::create(&path_str).unwrap();

        Self {
            file: Arc::new(Mutex::new(file)), // Wrap File in Mutex and Arc for sharing
            path: path_str,
        }
    }
    /// Writes a line (adds \n automatically) to the file synchronously.
    /// This method will block the current thread until the write is complete.
    /// It acquires a lock on the file to ensure exclusive access during the write.
    pub fn write_line(&self, line: &str) -> io::Result<()> {
        // Acquire the mutex lock. This will block if another thread holds the lock.
        let mut file_guard = self.file.lock().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to acquire file lock: {}", e),
            )
        })?;

        // Perform the blocking write operations
        file_guard.write_all(line.as_bytes())?;
        file_guard.write_all(b"\n")?; // Add newline
        Ok(())
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn dev_null() -> Self {
        let file = File::create("/dev/null").unwrap();
        Self {
            file: Arc::new(Mutex::new(file)),
            path: "/dev/null".into(),
        }
    }
}
