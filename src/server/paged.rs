//! MiniCPM5 service with one shared weight set and independently paged requests.
use super::*;
use crate::{
    inference::page_cache::{PageCache, Sequence},
    prefix_cache::{Key, Store},
    sampling::Sampling,
};
use std::collections::VecDeque;
mod preempt;
mod request;
mod schedule;

struct Request {
    job: Job,
    tokens: Vec<u32>,
    prompt_tokens: usize,
    cached: usize,
    computed: usize,
    maximum: usize,
    generated: Vec<u32>,
    sampling: Option<Sampling>,
    presence: f32,
    sequence: Sequence,
    streaming: stream::Text,
    first: Option<Instant>,
    stopped: bool,
    done: bool,
}

struct Runtime {
    engine: Engine,
    tokenizer: Tokenizer,
    template: String,
    bos: String,
    eos: Vec<u32>,
    cache: PageCache,
    options: options::Options,
    context: usize,
    gpu_sampling: bool,
    queue: VecDeque<Job>,
    active: Vec<Option<Request>>,
    paused: VecDeque<Request>,
    preemptions: u64,
}

pub(super) fn worker(
    path: String,
    context: usize,
    options: options::Options,
    ready: std::sync::mpsc::SyncSender<Result<()>>,
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<Job>,
) {
    let loaded = Runtime::load(&path, context, options);
    let mut runtime = match loaded {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    loop {
        if runtime.queue.is_empty()
            && runtime.paused.is_empty()
            && runtime.active.iter().all(Option::is_none)
        {
            let Some(job) = jobs.blocking_recv() else {
                break;
            };
            runtime.receive(job);
        }
        // Admission can tokenize a prompt while more HTTP requests arrive.
        // Collect them again before launching GPU work, without adding a wait.
        for _ in 0..2 {
            for _ in 0..64 {
                match jobs.try_recv() {
                    Ok(job) => runtime.receive(job),
                    Err(_) => break,
                }
            }
            runtime.admit();
        }
        if let Err(error) = runtime.step() {
            eprintln!("paged inference worker failed: {error:#}");
            let message = format!("{error:#}");
            for request in runtime.active.iter_mut().filter_map(Option::take) {
                finish_job(request.job, Err(anyhow!(message.clone())));
            }
            for request in runtime.paused.drain(..) {
                finish_job(request.job, Err(anyhow!(message.clone())));
            }
            for job in runtime.queue.drain(..) {
                finish_job(job, Err(anyhow!(message.clone())));
            }
            break;
        }
    }
}

fn finish_job(job: Job, response: Result<Json>) {
    if let Some(output) = &job.stream {
        output.finish(&response);
    }
    let _ = job.reply.send(response);
}

impl Runtime {
    fn load(path: &str, context: usize, options: options::Options) -> Result<Self> {
        anyhow::ensure!(options.projector.is_none(), "MiniCPM5 is text-only");
        anyhow::ensure!(
            (1..=131072).contains(&context),
            "invalid MiniCPM5 request context"
        );
        anyhow::ensure!(
            context <= options.kv_tokens,
            "request context exceeds physical KV pool"
        );
        let directory = std::path::Path::new(path);
        let mut profile = vec![
            6, // Numerical implementation revision; incompatible KV must not be restored.
            u8::from(std::env::var("LIA_LLAMA_FLASH").as_deref() != Ok("0")),
            u8::from(std::env::var("LIA_BF16_VECTOR").as_deref() != Ok("0")),
            u8::from(std::env::var("LIA_LLAMA_FLASH_DECODE").as_deref() != Ok("0")),
            u8::from(std::env::var("LIA_LLAMA_FUSION").as_deref() != Ok("0")),
            u8::from(std::env::var("LIA_LLAMA_FLASH_WIDE").as_deref() == Ok("1")),
            u8::from(std::env::var("LIA_LLAMA_GATED").as_deref() != Ok("0")),
            u8::from(std::env::var("LIA_LLAMA_FULL_PREFILL_SPAN").as_deref() == Ok("1")),
        ];
        profile.extend_from_slice(
            &(Engine::requested_decode_span().unwrap_or(0) as u16).to_le_bytes(),
        );
        let shadow_mib = std::env::var("LIA_LLAMA_WEIGHT_CACHE_MIB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(4096);
        profile.extend_from_slice(&shadow_mib.to_le_bytes());
        let namespace = Key::checkpoint(directory)
            .context("identifying KV cache checkpoint")?
            .variant(&profile);
        let store = Store::open(
            &options.cache_directory,
            namespace,
            options.host_bytes,
            options.disk_bytes,
            crate::inference::MINICPM5_PAGE_BYTES,
        )?;
        let cache = PageCache::new(namespace, options.kv_tokens / 32, store)?;
        let tokenizer = Tokenizer::from_huggingface(path)?;
        let template = std::fs::read_to_string(directory.join("chat_template.jinja"))?;
        let config: Json = serde_json::from_reader(std::fs::File::open(
            directory.join("tokenizer_config.json"),
        )?)?;
        let bos = config["bos_token"]
            .as_str()
            .context("missing MiniCPM5 BOS token")?
            .to_owned();
        let generation: Json = serde_json::from_reader(std::fs::File::open(
            directory.join("generation_config.json"),
        )?)?;
        let eos: Vec<u32> = serde_json::from_value(generation["eos_token_id"].clone())?;
        let count = if options.batch {
            options.maximum_sequences
        } else {
            1
        };
        let mut engine = Engine::load(path, options.kv_tokens)?;
        engine.configure_paging(count, context)?;
        Ok(Self {
            engine,
            tokenizer,
            template,
            bos,
            eos,
            cache,
            options,
            context,
            gpu_sampling: std::env::var("LIA_GPU_NUCLEUS").as_deref() != Ok("0"),
            queue: VecDeque::new(),
            paused: VecDeque::new(),
            preemptions: 0,
            active: (0..count).map(|_| None).collect(),
        })
    }

    fn receive(&mut self, job: Job) {
        if let Some(channels) = &job.realtime {
            let _ = channels
                .output
                .blocking_send(json!({"type":"error","error":{
                "type":"invalid_request_error","message":"native duplex requires MiniCPM-o 4.5"}}));
            finish_job(job, Err(anyhow!("native duplex requires MiniCPM-o 4.5")));
            return;
        }
        if job.path == "/v1/chat/completions" {
            self.queue.push_back(job);
            return;
        }
        let response = match job.path.as_str() {
            "/health" => Ok(
                json!({"status":"ok","backend":"lia.rs","architecture":"minicpm5","batch":self.options.batch,"maximum_sequences":self.active.len(),"active_requests":self.active.iter().flatten().count(),"paused_requests":self.paused.len(),"kv_preemptions":self.preemptions,"context":self.context,"kv_pool_tokens":self.options.kv_tokens,"kv_cache":"f16","prefix_cache":true,"kv_transfer_backend":self.engine.kv_transfer_backend(),"host_kv_bytes":self.options.host_bytes,"ssd_kv_bytes":self.options.disk_bytes,"gpu_cache_hits":self.cache.gpu_hits,"ram_cache_hits":self.cache.store.statistics.ram_hits,"ssd_cache_hits":self.cache.store.statistics.disk_hits,"cache_misses":self.cache.store.statistics.misses,"gpu_evictions":self.cache.evictions,"decode_cuda_graph":self.engine.cuda_graph_enabled() && self.engine.buffers_preallocated(),"cuda_graph_stats":self.engine.cuda_graph_stats(),"bf16_vector":self.engine.bf16_vector_enabled(),"attention_backend":self.engine.llama_attention_backend(),"tensor_decode":self.engine.tensor_decode_enabled(),"decode_span_override":Engine::requested_decode_span(),"full_prefill_span":self.engine.full_prefill_span_enabled(),"prefill_fusion":self.engine.fused_prefill_enabled(),"gated_projection_fusion":self.engine.gated_projection_fusion(),"prefill_weight_cache_bytes":self.engine.prefill_weight_cache_bytes(),"gpu_nucleus_candidates":self.gpu_sampling,"nucleus_fallbacks":self.engine.nucleus_fallbacks(),"prefill_batch":self.engine.prefill_batch(),"gpu_buffers_preallocated":self.engine.buffers_preallocated(),"mtp":false,"thinking":true}),
            ),
            "/apply-template" => {
                crate::chat::render_template_with_bos(&self.template, &job.input, &self.bos)
                    .map(|prompt| json!({"prompt":prompt}))
            }
            "/tokenize" => self
                .tokenizer
                .encode(
                    job.input["content"].as_str().unwrap_or(""),
                    job.input["parse_special"].as_bool().unwrap_or(true),
                )
                .map(|tokens| json!({"tokens":tokens})),
            "/cache/evict" => (|| -> Result<Json> {
                anyhow::ensure!(
                    self.active.iter().all(Option::is_none),
                    "cache control requires idle requests"
                );
                self.cache.evict_idle(&self.engine)?;
                if job.input["ram"] == true {
                    self.cache.store.clear_ram()?;
                }
                Ok(json!({"status":"ok"}))
            })(),
            _ => Err(anyhow!("unknown endpoint")),
        };
        finish_job(job, response);
    }
}
