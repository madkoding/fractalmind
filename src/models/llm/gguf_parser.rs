#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use tracing::{debug, info, warn};

use super::fractal_model::ModelArchitecture;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" en little-endian
const GGUF_VERSION: u32 = 3;

/// Tipos de datos en GGUF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GGUFValueType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => bail!("Invalid GGUF value type: {}", value),
        }
    }
}

/// Valor de metadatos en GGUF
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            Self::UInt64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match self {
            Self::String(s) => Some(s.clone()),
            _ => None,
        }
    }
}

/// Información del tensor en GGUF
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

/// Parser de archivos GGUF
pub struct GGUFParser {
    metadata: HashMap<String, GGUFValue>,
    tensors: Vec<GGUFTensorInfo>,
    version: u32,
}

impl GGUFParser {
    /// Parse un archivo GGUF y extrae metadatos
    pub fn parse_file(file_path: &str) -> Result<Self> {
        info!("Parsing GGUF file: {}", file_path);

        let file = File::open(file_path)
            .context("Failed to open GGUF file")?;
        
        let mmap = unsafe {
            Mmap::map(&file).context("Failed to memory-map file")?
        };

        Self::parse_bytes(&mmap)
    }

    /// Parse bytes de un archivo GGUF
    fn parse_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Leer cabecera
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic number: expected {:#x}, got {:#x}", GGUF_MAGIC, magic);
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION {
            warn!("GGUF version mismatch: expected {}, got {}", GGUF_VERSION, version);
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;

        debug!("GGUF version: {}, tensors: {}, metadata: {}", version, tensor_count, metadata_kv_count);

        // Leer metadatos
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = Self::read_metadata_kv(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Leer información de tensores
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let tensor_info = Self::read_tensor_info(&mut cursor)?;
            tensors.push(tensor_info);
        }

        Ok(Self {
            metadata,
            tensors,
            version,
        })
    }

    /// Lee un par clave-valor de metadatos
    fn read_metadata_kv(cursor: &mut Cursor<&[u8]>) -> Result<(String, GGUFValue)> {
        let key = Self::read_string(cursor)?;
        let value_type = cursor.read_u32::<LittleEndian>()?;
        let value_type = GGUFValueType::try_from(value_type)?;
        let value = Self::read_value(cursor, value_type)?;
        Ok((key, value))
    }

    /// Lee un string de GGUF
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let len = cursor.read_u64::<LittleEndian>()? as usize;
        let mut buffer = vec![0u8; len];
        cursor.read_exact(&mut buffer)?;
        String::from_utf8(buffer).context("Invalid UTF-8 in GGUF string")
    }

    /// Lee un valor según su tipo
    fn read_value(cursor: &mut Cursor<&[u8]>, value_type: GGUFValueType) -> Result<GGUFValue> {
        match value_type {
            GGUFValueType::UInt8 => Ok(GGUFValue::UInt8(cursor.read_u8()?)),
            GGUFValueType::Int8 => Ok(GGUFValue::Int8(cursor.read_i8()?)),
            GGUFValueType::UInt16 => Ok(GGUFValue::UInt16(cursor.read_u16::<LittleEndian>()?)),
            GGUFValueType::Int16 => Ok(GGUFValue::Int16(cursor.read_i16::<LittleEndian>()?)),
            GGUFValueType::UInt32 => Ok(GGUFValue::UInt32(cursor.read_u32::<LittleEndian>()?)),
            GGUFValueType::Int32 => Ok(GGUFValue::Int32(cursor.read_i32::<LittleEndian>()?)),
            GGUFValueType::Float32 => Ok(GGUFValue::Float32(cursor.read_f32::<LittleEndian>()?)),
            GGUFValueType::Bool => Ok(GGUFValue::Bool(cursor.read_u8()? != 0)),
            GGUFValueType::String => Ok(GGUFValue::String(Self::read_string(cursor)?)),
            GGUFValueType::UInt64 => Ok(GGUFValue::UInt64(cursor.read_u64::<LittleEndian>()?)),
            GGUFValueType::Int64 => Ok(GGUFValue::Int64(cursor.read_i64::<LittleEndian>()?)),
            GGUFValueType::Float64 => Ok(GGUFValue::Float64(cursor.read_f64::<LittleEndian>()?)),
            GGUFValueType::Array => {
                let array_type = cursor.read_u32::<LittleEndian>()?;
                let array_type = GGUFValueType::try_from(array_type)?;
                let array_len = cursor.read_u64::<LittleEndian>()? as usize;
                
                let mut values = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    values.push(Self::read_value(cursor, array_type)?);
                }
                Ok(GGUFValue::Array(values))
            }
        }
    }

    /// Lee información de un tensor
    fn read_tensor_info(cursor: &mut Cursor<&[u8]>) -> Result<GGUFTensorInfo> {
        let name = Self::read_string(cursor)?;
        let n_dimensions = cursor.read_u32::<LittleEndian>()? as usize;
        
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for _ in 0..n_dimensions {
            dimensions.push(cursor.read_u64::<LittleEndian>()?);
        }
        
        let tensor_type = cursor.read_u32::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;

        Ok(GGUFTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        })
    }

    /// Extrae la arquitectura del modelo desde los metadatos
    pub fn extract_architecture(&self) -> Result<ModelArchitecture> {
        let model_type = self.get_metadata_string("general.architecture")?;
        
        let prefix = format!("{}", model_type);
        
        let n_layers = self.get_metadata_u32(&format!("{}.block_count", prefix))?;
        let embedding_dim = self.get_metadata_u32(&format!("{}.embedding_length", prefix))?;
        let n_heads = self.get_metadata_u32(&format!("{}.attention.head_count", prefix))?;
        let ffn_dim = self.get_metadata_u32(&format!("{}.feed_forward_length", prefix))?;
        
        // Intentar obtener vocab_size de diferentes campos posibles
        let vocab_size = self.get_metadata_u32("tokenizer.ggml.tokens.length")
            .or_else(|_| self.get_metadata_u32(&format!("{}.vocab_size", prefix)))
            .unwrap_or(32000); // Valor por defecto

        Ok(ModelArchitecture {
            model_type,
            n_layers,
            embedding_dim,
            vocab_size,
            n_heads,
            ffn_dim,
        })
    }

    /// Obtiene un metadato como string
    fn get_metadata_string(&self, key: &str) -> Result<String> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_string())
            .context(format!("Missing or invalid metadata: {}", key))
    }

    /// Obtiene un metadato como u32
    fn get_metadata_u32(&self, key: &str) -> Result<u32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .context(format!("Missing or invalid metadata: {}", key))
    }

    /// Obtiene información sobre los tensores
    pub fn get_tensors(&self) -> &[GGUFTensorInfo] {
        &self.tensors
    }

    /// Obtiene todos los metadatos
    pub fn get_metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_value_type_conversion() {
        assert_eq!(GGUFValueType::try_from(0).unwrap(), GGUFValueType::UInt8);
        assert_eq!(GGUFValueType::try_from(8).unwrap(), GGUFValueType::String);
        assert!(GGUFValueType::try_from(99).is_err());
    }

    #[test]
    fn test_gguf_value_as_u32() {
        let val = GGUFValue::UInt32(42);
        assert_eq!(val.as_u32(), Some(42));

        let val = GGUFValue::String("test".to_string());
        assert_eq!(val.as_u32(), None);
    }
}
