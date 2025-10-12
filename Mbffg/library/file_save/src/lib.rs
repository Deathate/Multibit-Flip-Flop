use serde;
pub use serde::*;
use serde_json;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
const FOLDER_PATH: &str = "tmp";
fn create_folder() -> Result<(), Box<dyn Error>> {
    if !Path::new(FOLDER_PATH).exists() {
        std::fs::create_dir(FOLDER_PATH)?;
    }
    Ok(())
}
pub fn save_to_file<T: serde::Serialize>(data: &T, filename: &str) -> Result<(), Box<dyn Error>> {
    create_folder()?;
    println!(
        "file_save > Data saved to file '{}/{}'",
        FOLDER_PATH, filename
    );

    let filename = format!("{}/{}", FOLDER_PATH, filename);
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, data)?;
    Ok(())
}
pub fn load_from_file<T: for<'de> serde::Deserialize<'de>>(
    filename: &str,
) -> Result<T, Box<dyn Error>> {
    create_folder()?;
    let filename = format!("{}/{}", FOLDER_PATH, filename);
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data = serde_json::from_reader(reader)?;
    Ok(data)
}
pub fn save_binary_file<T: serde::Serialize>(
    data: &T,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    create_folder()?;
    let filename = format!("{}/{}", FOLDER_PATH, filename);
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}
pub fn load_binary_file<T: for<'de> serde::Deserialize<'de>>(
    filename: &str,
) -> Result<T, Box<dyn Error>> {
    create_folder()?;
    let filename = format!("{}/{}", FOLDER_PATH, filename);
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}
pub fn exist_file(filename: &str) -> Result<bool, Box<dyn Error>> {
    create_folder()?;
    let filename = format!("{}/{}", FOLDER_PATH, filename);
    let path = Path::new(&filename);
    Ok(path.exists())
}
