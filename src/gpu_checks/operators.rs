use super::*;

pub fn run() -> Result<()> {
    super::reductions::run()?;
    super::convolution::run()?;
    super::delta::run()?;
    super::basic_attention::run()?;
    Ok(())
}
