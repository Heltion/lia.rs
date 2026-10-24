use anyhow::anyhow;
use lia::{chat, gguf, quant, tokenizer};
#[cfg(feature = "cuda")]
use lia::{gpu, inference, server};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    if matches!(command.as_deref(), None | Some("--help" | "-h" | "help")) {
        println!(
            "lia.rs — local inference in Rust\n\nCommands:\n  inspect MODEL\n  tokenize MODEL TEXT\n  render MODEL\n  decode-block TYPE"
        );
        #[cfg(feature = "cuda")]
        println!(
            "  serve MODEL [PORT] [CONTEXT] [--mmproj PROJECTOR]\n    MiniCPM5: --batch | --no-batch, --max-sequences N, --kv-pool-tokens N\n    --kv-host-mib N --kv-ssd-mib N --kv-cache-dir PATH\n  forward MODEL TOKEN_IDS\n  forward-batch MODEL TOKEN_IDS\n  kernel-info"
        );
        #[cfg(feature = "image-tools")]
        println!("  image-tools\n  read-image INPUT [OUTPUT.png] [--time SECONDS]");
        return Ok(());
    }

    #[cfg(feature = "image-tools")]
    if command.as_deref() == Some("image-tools") {
        anyhow::ensure!(args.next().is_none(), "usage: lia image-tools");
        return lia::image_tools::serve();
    }
    #[cfg(feature = "image-tools")]
    if command.as_deref() == Some("read-image") {
        let input = args
            .next()
            .ok_or_else(|| anyhow!("usage: lia read-image INPUT [OUTPUT.png] [--time SECONDS]"))?;
        let mut output = None;
        let mut time = None;
        while let Some(arg) = args.next() {
            if arg == "--time" {
                anyhow::ensure!(time.is_none(), "duplicate --time");
                time = Some(
                    args.next()
                        .ok_or_else(|| anyhow!("missing --time seconds"))?
                        .parse()?,
                );
            } else {
                anyhow::ensure!(output.is_none(), "unexpected argument");
                output = Some(arg);
            }
        }
        return lia::image_tools::read_cli(&input, output.as_deref(), time);
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
        for (position, row) in logits.chunks_exact(engine.vocabulary_size()).enumerate() {
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
    if command.as_deref() == Some("serve") {
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        let port = args.next().map(|s| s.parse()).transpose()?.unwrap_or(18474);
        let capacity = args.next().map(|s| s.parse()).transpose()?.unwrap_or(16384);
        let options = server::Options::parse(args)?;
        return server::run_with_options(&path, port, capacity, options);
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
            if engine.mtp_enabled() {
                let draft = engine.draft(next)?;
                let draft_next = draft
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .ok_or_else(|| anyhow!("empty draft logits"))?
                    .0;
                eprintln!("position={position} mtp_input={next} mtp_draft={draft_next}");
            }
            println!("position={position}");
            for (token, logit) in top.iter().take(10) {
                println!("{token} {logit}");
            }
        }
        return Ok(());
    }
    if command.as_deref() == Some("render") {
        use std::io::Read;
        let path = args.next().ok_or_else(|| anyhow!("missing model path"))?;
        if args.next().is_some() {
            return Err(anyhow!("unexpected argument"));
        }
        let mut input = String::new();
        std::io::stdin()
            .take(16 * 1024 * 1024)
            .read_to_string(&mut input)?;
        let request: serde_json::Value = serde_json::from_str(&input)?;
        let rendered = if std::path::Path::new(&path).is_dir() {
            let template =
                std::fs::read_to_string(std::path::Path::new(&path).join("chat_template.jinja"))?;
            chat::render_template(&template, &request)?
        } else {
            chat::render(&gguf::Model::open(path)?, &request)?
        };
        print!("{rendered}");
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
        let tokenizer = if std::path::Path::new(&path).is_dir() {
            tokenizer::Tokenizer::from_huggingface(&path)?
        } else {
            tokenizer::Tokenizer::from_model(&gguf::Model::open(path)?)?
        };
        let ids = tokenizer.encode(&text, true)?;
        println!("{ids:?}");
        return Ok(());
    }
    if command.as_deref() != Some("inspect") {
        return Err(anyhow!("unknown command; run lia --help"));
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
