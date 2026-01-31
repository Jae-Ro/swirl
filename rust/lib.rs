use pyo3::prelude::*;
use pyo3::types::PyDict; // PyTuple removed here
use pyo3::IntoPyObject; 
use pest::Parser;
use pest_derive::Parser;
use rayon::prelude::*;
use regex::Regex;
use std::sync::OnceLock;

#[derive(Parser)]
#[grammar = "rust/grammar.pest"]
pub struct KVParser;

static HEADER_FIX: OnceLock<Regex> = OnceLock::new();
static SPLIT_PATTERN: OnceLock<Regex> = OnceLock::new();

#[pyfunction]
pub fn smart_parse_batch(py: Python<'_>, logs: Vec<String>) -> PyResult<Vec<Py<PyAny>>> {
    let header_fix = HEADER_FIX.get_or_init(|| {
        Regex::new(r"(?P<name>[a-zA-Z]+)\s+(?P<id>\d+):\s*").unwrap()
    });
    
    let split_pattern = SPLIT_PATTERN.get_or_init(|| {
        Regex::new(r"(?:,\s*|\s+)[a-zA-Z_]\w*\s*[:=]").unwrap()
    });

    let processed_data: Vec<(String, Vec<(String, String)>, Vec<String>)> = py.detach(|| {
        logs.par_iter()
            .map(|raw_str| {
                let mut extracted = Vec::new();
                let mut unparsed_segments = Vec::new();

                let content = header_fix.replace_all(raw_str, "$name=$id, ").to_string();

                let mut segments = Vec::new();
                let mut last = 0;
                for mat in split_pattern.find_iter(&content) {
                    segments.push(&content[last..mat.start()]);
                    let match_str = mat.as_str();
                    let key_start_offset = match_str.find(|c: char| c.is_alphanumeric() || c == '_').unwrap_or(0);
                    last = mat.start() + key_start_offset;
                }
                segments.push(&content[last..]);

                for seg in segments {
                    let seg_trimmed = seg.trim();
                    if seg_trimmed.is_empty() { continue; }

                    if let Ok(mut pairs) = KVParser::parse(Rule::pair_segment, seg_trimmed) {
                        let pair = pairs.next().unwrap();
                        let mut inner = pair.into_inner();
                        
                        let k = inner.next().unwrap().as_str().to_string();
                        let _delim = inner.next().unwrap();
                        
                        let v = inner.next()
                            .map(|val| {
                                let s = val.as_str().trim();
                                s.strip_suffix(',').unwrap_or(s).trim().to_string()
                            })
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| "None".to_string());
                        
                        extracted.push((k, v));
                    } else {
                        unparsed_segments.push(seg_trimmed.to_string());
                    }
                }

                (raw_str.clone(), extracted, unparsed_segments)
            })
            .collect()
    });

    let mut results = Vec::with_capacity(processed_data.len());
    for (original_str, pairs, unparsed) in processed_data {
        let dict = PyDict::new(py);
        for (k, v) in pairs {
            let _ = dict.set_item(k, v);
        }
        if !unparsed.is_empty() {
            let _ = dict.set_item("_unparsed", unparsed.join(" "));
        }
        
        // Convert the Rust tuple (String, Bound<PyDict>) into a Python tuple
        let log_tuple = (original_str, dict).into_pyobject(py)?;
        results.push(log_tuple.into_any().unbind());
    }

    Ok(results)
}

#[pymodule]
fn rust_ingestion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smart_parse_batch, m)?)?;
    Ok(())
}