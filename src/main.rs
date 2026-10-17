#![feature(f16)]
use anyhow::anyhow;
mod chat;
mod codebooks;
mod gguf;
#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
mod gpu_checks;
#[cfg(feature = "cuda")]
mod inference;
mod quant;
#[cfg(feature = "cuda")]
mod server;
mod tokenizer;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("check-speculation") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let initial: u32 = args
            .next()
            .ok_or_else(|| anyhow!("missing initial token ID"))?
            .parse()?;
        let expected: Vec<u32> = args
            .next()
            .ok_or_else(|| anyhow!("missing reference token sequence"))?
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let mut engine = inference::Engine::load(&path, 64)?;
        engine.forward(initial)?;
        let mut first = engine.current_token()?;
        let mut actual = Vec::new();
        let mut proposed = 0;
        let mut accepted = 0;
        let mut rounds = 0;
        let start = std::time::Instant::now();
        while actual.len() < expected.len() {
            let (tokens, next, n, a) =
                engine.speculate(first, expected.len() - actual.len(), 248046)?;
            actual.extend(tokens);
            first = next;
            proposed += n;
            accepted += a;
            rounds += 1;
        }
        println!(
            "tokens={actual:?} proposed={proposed} accepted={accepted} rounds={rounds} seconds={}",
            start.elapsed().as_secs_f64()
        );
        if actual != expected {
            return Err(anyhow!("speculative tokens differ from reference"));
        }
        println!(
            "Native MTP proposals, batched verification and recurrent rollback match the supplied greedy reference"
        );
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("check-prefill-gpu") {
        return gpu_checks::prefill_matrix();
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("check-int-gpu") {
        return gpu_checks::integer_matmul();
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("forward-batch") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let tokens: Vec<u32> = args
            .next()
            .ok_or_else(|| anyhow!("missing comma-separated token IDs"))?
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let mut engine = inference::Engine::load(&path, tokens.len().max(64))?;
        let start = std::time::Instant::now();
        let logits = engine.forward_many(&tokens, true)?;
        eprintln!(
            "batch={} seconds={}",
            tokens.len(),
            start.elapsed().as_secs_f64()
        );
        for (position, row) in logits.as_chunks::<248320>().0.iter().enumerate() {
            let (token, logit) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or_else(|| anyhow!("empty logits"))?;
            println!("position={position} token={token} logit={logit}");
        }
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("kernel-info") {
        return gpu::kernel_info();
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("kernel-names") {
        println!(
            "IQ4_XS {}",
            gpu::kernels::matrix::device::matvec_packed_ptx_name::<23, 136, 256>()
        );
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("serve") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let port = args.next().map(|s| s.parse()).transpose()?.unwrap_or(18474);
        let capacity = args.next().map(|s| s.parse()).transpose()?.unwrap_or(16384);
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        return server::run(&path, port, capacity);
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("forward") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let tokens: Vec<u32> = args
            .next()
            .ok_or_else(|| anyhow!("missing comma-separated token IDs"))?
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let start = std::time::Instant::now();
        let mut engine = inference::Engine::load(&path, 64)?;
        eprintln!("load_seconds={}", start.elapsed().as_secs_f64());
        for (position, token) in tokens.iter().enumerate() {
            let start = std::time::Instant::now();
            let logits = engine.forward(*token)?;
            eprintln!(
                "position={position} forward_seconds={}",
                start.elapsed().as_secs_f64()
            );
            let mut top: Vec<_> = logits.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let next = top[0].0 as u32;
            let draft = engine.draft(next)?;
            let draft_next = draft
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or_else(|| anyhow!("empty draft logits"))?
                .0;
            eprintln!("position={position} mtp_input={next} mtp_draft={draft_next}");
            println!("position={position}");
            for (token, logit) in top.iter().take(10) {
                println!("{token} {logit}");
            }
        }
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("check-model-gpu") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        return gpu::check_model(&path);
    }
    #[cfg(feature = "cuda")]
    if command.as_deref() == Some("check-gpu") {
        gpu::check()?;
        gpu_checks::run()?;
        gpu_checks::long_attention()?;
        return gpu_checks::cached_rotary();
    }
    if command.as_deref() == Some("render") {
        use std::io::Read;
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let model = gguf::Model::open(path)?;
        let mut input = String::new();
        std::io::stdin()
            .take(16 * 1024 * 1024)
            .read_to_string(&mut input)?;
        let request: serde_json::Value = serde_json::from_str(&input)?;
        print!("{}", chat::render(&model, &request)?);
        return Ok(());
    }
    if command.as_deref() == Some("decode-block") {
        use std::io::{Read, Write};
        let kind: u32 = args
            .next()
            .ok_or_else(|| anyhow!("missing storage type"))?
            .parse()?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let mut bytes = Vec::new();
        std::io::stdin().take(1024).read_to_end(&mut bytes)?;
        let decoded = quant::decode_block(kind, &bytes)?;
        let mut stdout = std::io::stdout().lock();
        for value in decoded {
            stdout.write_all(&value.to_le_bytes())?;
        }
        return Ok(());
    }
    if command.as_deref() == Some("tokenize") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let text = args.next().ok_or_else(|| anyhow!("missing text"))?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let model = gguf::Model::open(path)?;
        let tokenizer = tokenizer::Tokenizer::from_model(&model)?;
        let ids = tokenizer.encode(&text, true)?;
        if tokenizer.decode(&ids)? != text.as_bytes() {
            return Err(anyhow!("token round trip failed"));
        }
        println!("{ids:?}");
        return Ok(());
    }
    if command.as_deref() != Some("inspect") {
        return Err(anyhow!("usage: lia inspect MODEL.gguf"));
    }
    let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
    if args.next().is_some() {
        return Err(anyhow!("unexpected argument"));
    }
    let model = gguf::Model::open(path)?;
    println!(
        "GGUF v{}; {} tensors; data offset {}",
        model.version,
        model.tensors.len(),
        model.data_offset
    );
    for (key, value) in &model.metadata {
        println!("{key} = {value:?}");
    }
    for tensor in &model.tensors {
        println!(
            "{} {:?} type={} offset={} bytes={}",
            tensor.name, tensor.shape, tensor.kind, tensor.offset, tensor.byte_len
        );
    }
    Ok(())
}
