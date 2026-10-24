//! Service scheduling and cache capacities are independent of per-request context.
use anyhow::{Context, Result, bail, ensure};
use std::path::PathBuf;

pub struct Options {
    pub projector: Option<String>,
    pub batch: bool,
    pub maximum_sequences: usize,
    pub kv_tokens: usize,
    pub host_bytes: usize,
    pub disk_bytes: u64,
    pub cache_directory: PathBuf,
    pub(crate) paging_requested: bool,
}

impl Options {
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let cache = if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(root)
        } else {
            PathBuf::from(std::env::var_os("HOME").context("missing home directory for KV cache")?)
                .join(".cache")
        };
        let mut options = Self {
            projector: None,
            batch: false,
            maximum_sequences: 8,
            kv_tokens: 65536,
            host_bytes: 1024 * 1024 * 1024,
            disk_bytes: 2 * 1024 * 1024 * 1024,
            cache_directory: cache.join("lia.rs/kv"),
            paging_requested: false,
        };
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            if argument != "--mmproj" {
                options.paging_requested = true;
            }
            match argument.as_str() {
                "--batch" => options.batch = true,
                "--no-batch" => options.batch = false,
                "--mmproj" => {
                    options.projector = Some(arguments.next().context("missing projector path")?)
                }
                "--kv-cache-dir" => {
                    options.cache_directory = arguments
                        .next()
                        .context("missing KV cache directory")?
                        .into()
                }
                "--max-sequences" => {
                    options.maximum_sequences = integer(&mut arguments, &argument)?
                }
                "--kv-pool-tokens" => options.kv_tokens = integer(&mut arguments, &argument)?,
                "--kv-host-mib" => {
                    options.host_bytes = integer(&mut arguments, &argument)?
                        .checked_mul(1024 * 1024)
                        .context("host KV budget overflow")?
                }
                "--kv-ssd-mib" => {
                    options.disk_bytes = u64::try_from(integer(&mut arguments, &argument)?)?
                        .checked_mul(1024 * 1024)
                        .context("disk KV budget overflow")?
                }
                _ => bail!("unknown service option {argument}"),
            }
        }
        ensure!(
            (1..=64).contains(&options.maximum_sequences),
            "max-sequences must be 1..=64"
        );
        ensure!(
            (32..=131072).contains(&options.kv_tokens) && options.kv_tokens.is_multiple_of(32),
            "KV pool must be 32..=131072 tokens in whole 32-token pages"
        );
        Ok(options)
    }
}

fn integer(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<usize> {
    arguments
        .next()
        .with_context(|| format!("missing value for {name}"))?
        .parse()
        .with_context(|| format!("invalid value for {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_toggle_keeps_cache_budgets() -> Result<()> {
        let enabled = Options::parse(
            ["--batch", "--kv-host-mib", "512", "--kv-ssd-mib", "1024"].map(str::to_owned),
        )?;
        let disabled = Options::parse(
            ["--no-batch", "--kv-host-mib", "512", "--kv-ssd-mib", "1024"].map(str::to_owned),
        )?;
        assert!(enabled.batch && !disabled.batch);
        assert_eq!(enabled.host_bytes, disabled.host_bytes);
        assert_eq!(enabled.disk_bytes, disabled.disk_bytes);
        assert!(Options::parse(["--kv-pool-tokens", "33"].map(str::to_owned)).is_err());
        Ok(())
    }
}
