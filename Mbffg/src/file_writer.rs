use crate::PathLike;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

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
