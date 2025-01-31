use serde;
pub use serde::*;
use serde_json;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn save_to_file<T: serde::Serialize>(data: &T, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, data)?;
    Ok(())
}
pub fn load_from_file<T: for<'de> serde::Deserialize<'de>>(
    filename: &str,
) -> Result<T, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data = serde_json::from_reader(reader)?;
    Ok(data)
}
pub fn save_binary_file<T: serde::Serialize>(
    data: &T,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}
pub fn load_binary_file<T: for<'de> serde::Deserialize<'de>>(
    filename: &str,
) -> Result<T, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}
