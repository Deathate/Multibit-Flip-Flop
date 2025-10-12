use std::sync::Arc; // Not strictly needed for File in this pattern, but common
use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, Result};
use tokio::sync::mpsc;
// Define the type of message we'll send over the channel
enum WriteCommand {
    Write(Vec<u8>),
    WriteLine(Vec<u8>),
}

pub struct AsyncFileWriter {
    // The sender half of the MPSC channel.
    // Each call to `write` or `write_line` will send a message here.
    sender: mpsc::Sender<WriteCommand>,
    // Store the path for potential error logging or future use,
    // though the writer task already has it.
    path: String,
}

impl AsyncFileWriter {
    pub fn new(path: impl Into<String>) -> Self {
        let path_str = path.into();

        // Remove the file if it exists (synchronously, as this is setup)
        if std::path::Path::new(&path_str).exists() {
            std::fs::remove_file(&path_str).unwrap_or_else(|e| {
                eprintln!("Failed to remove file {}: {}", path_str, e);
            });
        }

        // Create an MPSC channel. The buffer size (e.g., 100)
        // helps with backpressure.
        let (tx, mut rx) = mpsc::channel::<WriteCommand>(10000);

        let writer_path = path_str.clone();

        // Spawn a single, dedicated async task to handle all writes.
        // This task owns the `File` handle and processes messages sequentially.
        tokio::spawn(async move {
            let file_result = OpenOptions::new()
                .append(true)
                .create(true)
                .open(&writer_path)
                .await;

            let mut file = match file_result {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to open file for writing {}: {}", writer_path, e);
                    return; // Exit the writer task if file can't be opened
                }
            };

            // Loop indefinitely, receiving and processing write commands
            while let Some(command) = rx.recv().await {
                let write_result: Result<()> = match command {
                    WriteCommand::Write(data) => file.write_all(&data).await,
                    WriteCommand::WriteLine(mut data) => {
                        if !data.ends_with(b"\n") {
                            data.push(b'\n');
                        }
                        file.write_all(&data).await
                    }
                };

                if let Err(e) = write_result {
                    eprintln!("Failed to write to file {}: {}", writer_path, e);
                    // Decide on your error handling strategy:
                    // - Log and continue
                    // - Try to re-establish the file
                    // - Break the loop to stop the writer task if errors are persistent
                }
            }
            println!(
                "AsyncFileWriter internal writer task finished for path: {}",
                writer_path
            );
        });

        Self {
            sender: tx,
            path: path_str,
        }
    }

    /// Sends data to the internal writer task.
    /// This method is not async, but internally spawns a task to send the message.
    /// Errors during sending are logged, not returned.
    pub fn write(&self, data: &str) {
        let bytes = data.as_bytes().to_vec();
        let sender_clone = self.sender.clone(); // Clone the sender for the spawned task
        let path_clone = self.path.clone(); // Clone path for error logging

        tokio::spawn(async move {
            if let Err(e) = sender_clone.send(WriteCommand::Write(bytes)).await {
                eprintln!(
                    "Failed to send write command to writer task for {}: {}",
                    path_clone, e
                );
            }
        });
    }

    /// Sends a line (adds \n automatically if not present) to the internal writer task.
    /// This method is not async, but internally spawns a task to send the message.
    /// Errors during sending are logged, not returned.
    pub fn write_line(&self, line: &str) {
        let bytes = line.as_bytes().to_vec();
        let sender_clone = self.sender.clone(); // Clone the sender for the spawned task
        let path_clone = self.path.clone(); // Clone path for error logging

        tokio::spawn(async move {
            if let Err(e) = sender_clone.send(WriteCommand::WriteLine(bytes)).await {
                eprintln!(
                    "Failed to send write line command to writer task for {}: {}",
                    path_clone, e
                );
            }
        });
    }

    /// This method can be called to explicitly drop the sender,
    /// which will eventually cause the internal writer task to finish
    /// once all messages in the channel are processed.
    /// This is useful for graceful shutdown.
    pub fn close(&self) {
        // Drop the sender held by this struct.
        // If there are other clones (unlikely with this public API, but possible
        // if `self.sender.clone()` was exposed), they would also need to be dropped
        // for the receiver to close.
        drop(self.sender.clone()); // Cloning and dropping ensures the one owned by self is decremented
        println!(
            "AsyncFileWriter channel sender dropped for path: {}",
            self.path
        );
    }
}
use crate::PathLike;
use std::io::{self, Write};
use std::path::Path;
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

        // Open the file synchronously.
        let file = std::fs::OpenOptions::new()
            .append(true) // Open in append mode
            .create(true) // Create the file if it doesn't exist
            .open(&path_str)
            .unwrap(); // Perform the blocking open operation

        Self {
            file: Arc::new(Mutex::new(file)), // Wrap File in Mutex and Arc for sharing
            path: path_str,
        }
    }

    /// Writes data to the file synchronously.
    /// This method will block the current thread until the write is complete.
    /// It acquires a lock on the file to ensure exclusive access during the write.
    pub fn write(&self, data: &str) -> io::Result<()> {
        // Acquire the mutex lock. This will block if another thread holds the lock.
        let mut file_guard = self.file.lock().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to acquire file lock: {}", e),
            )
        })?;

        // Perform the blocking write operation
        file_guard.write_all(data.as_bytes())?;
        Ok(())
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

    /// Flushes the file's internal buffer to disk synchronously.
    /// This method will block the current thread until the flush is complete.
    /// It acquires a lock on the file to ensure exclusive access.
    pub fn flush(&self) -> io::Result<()> {
        // Acquire the mutex lock for flushing
        let mut file_guard = self.file.lock().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to acquire file lock for flush: {}", e),
            )
        })?;

        // Perform the blocking flush operation
        file_guard.flush()?;
        Ok(())
    }
    pub fn path(&self) -> &str {
        &self.path
    }
}
