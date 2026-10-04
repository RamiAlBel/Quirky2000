//! Quirky Round H trainer: HalfKA kb16 (mirrored) + threat inputs, FT pairwise, L1 -> L2 -> 1, 8 output buckets.
//! Configured by env vars (QK_*), see `cfg` calls below. All weights are saved as raw f32;
//! quantisation to the engine format (LNN6) happens in exp2/bullet/lnn6.py.
//!
//! QK_MODE=train (default) | eval (print 400*output for FENs in QK_FENS, weights from QK_RESUME)
//!        | dump (print feature indices for FENs in QK_FENS, no GPU)
mod inputs;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry},
        save::save_to_checkpoint,
    },
};
use bullet_trainer::{
    model::{InitSettings, ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, HostPool, TrainingSchedule, TrainingSteps, train},
};

const OUTPUT_BUCKETS: usize = 8;
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

// king square (rank-major, files a-d after mirroring) -> bucket; must match engine nnue6.cpp
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

fn cfg<T: std::str::FromStr>(key: &str, default: T) -> T {
    match std::env::var(key) {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("bad value for {key}: {v}")),
        Err(_) => default,
    }
}

fn cfg_s(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn read_fens(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {path}"))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn main() {
    let mode = cfg_s("QK_MODE", "train");
    let name = cfg_s("QK_NAME", "h_test");
    let threats = cfg("QK_THREATS", 1u32) == 1;
    let dual = cfg("QK_DUAL", 1u32) == 1;
    let scale: f32 = cfg("QK_SCALE", 0.39);
    let l1: usize = cfg("QK_L1", 1024);
    let l2: usize = cfg("QK_L2", 32);
    let l3: usize = cfg("QK_L3", 32);
    let l0reg: f32 = cfg("QK_L0REG", 0.005);

    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let thr = inputs::Threats::new();
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;

    if mode == "dump" {
        for fen in read_fens(&cfg_s("QK_FENS", "fens.txt")) {
            let pos: ChessBoard = format!("{fen} | 0 | 0.0").parse().unwrap();
            let (mut s, mut n) = inputs::features(&pos, &psqt, threats);
            s.sort_unstable();
            n.sort_unstable();
            println!("FEN {fen}");
            println!("STM {}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
            println!("NTM {}", n.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        }
        return;
    }

    println!("[quirky] name={name} threats={threats} dual={dual} scale={scale} L1={l1} L2={l2} L3={l3}");
    println!("[quirky] psqt inputs={} threat inputs={}", psqt.num_inputs(), thr.num_inputs());

    let inputs = ModelInputs::default()
        .add_sparse("stm/thr", (thr.num_inputs(), 1), thr.max_active())
        .add_sparse("ntm/thr", (thr.num_inputs(), 1), thr.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_thr, ntm_thr), stm_psqt), ntm_psqt), output_buckets), target)| {
            // threat weights + FT bias live in this affine
            let l0_thr = builder.new_affine("l0/thr/", thr.num_inputs(), l1);

            let l0f = builder.new_weights("l0/fac", (l1, 768), InitSettings::Zeroed);
            let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
            let mut l0_psqt = builder.new_weights("l0/psqt", (l1, psqt.num_inputs()), psqt_init);
            l0_psqt = l0_psqt + l0f.repeat(INPUT_BUCKETS);

            let l1a = builder.new_affine("l1/", l1, OUTPUT_BUCKETS * l2);
            let l2a = builder.new_affine("l2/", if dual { 2 * l2 } else { l2 }, OUTPUT_BUCKETS * l3);
            let l3a = builder.new_affine("l3/", l3, OUTPUT_BUCKETS);

            let ft = |t, p, start, end| {
                (l0_thr.slice(start, end).forward(t) + l0_psqt.slice_rows(start, end).matmul(p)).crelu()
            };
            let stm_hidden = ft(stm_thr, stm_psqt, 0, l1 / 2) * ft(stm_thr, stm_psqt, l1 / 2, l1);
            let ntm_hidden = ft(ntm_thr, ntm_psqt, 0, l1 / 2) * ft(ntm_thr, ntm_psqt, l1 / 2, l1);
            let l0_out = stm_hidden.concat(ntm_hidden);
            let l0_out_norm = l0_out.reduce_sum_rows() / (l1 as f32);

            let l1_out = l1a.forward(l0_out).select(output_buckets);
            let hl2 = if dual { l1_out.concat(l1_out.abs_pow(2.0)).crelu() } else { l1_out.crelu() };
            let hl3 = l2a.forward(hl2).select(output_buckets).crelu();
            let out = l3a.forward(hl3).select(output_buckets);

            let loss = out.sigmoid().squared_error(target) + l0reg * l0_out_norm;
            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), out)])
        },
    );

    let seed: u64 = cfg("QK_SEED", 12412421);
    let weights = ModelWeights::new(&defn, seed);
    let device = DefaultDevice::new(0).unwrap();
    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), AdamWParams::default()).unwrap();

    // int16 FT at Q0=255: psqt+fac merged must stay within +-1.98 -> clip each at 0.99
    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/fac", l0_clip);
    optimiser.set_params_for_weight("l0/psqt", l0_clip);
    // threat rows are int8 at Q0=255
    let thr_range = 127.0 / 255.0;
    let thr_clip = AdamWParams { max_weight: thr_range, min_weight: -thr_range, ..Default::default() };
    optimiser.set_params_for_weight("l0/thr/w", thr_clip);
    // l1/w default clip +-1.98 == int8 at Q1=64

    let resume = cfg_s("QK_RESUME", "");
    if !resume.is_empty() {
        optimiser.load_from_checkpoint(&format!("{resume}/optimiser_state")).unwrap();
        println!("[quirky] resumed from {resume}");
    }

    if mode == "eval" {
        evaluator.load_device_weights(optimiser.weights()).unwrap();
        let mapper = inputs::make_inputs_mapper((&inputs, psqt, output_buckets), threats, scale, wdl::ConstantWDL {
            value: 0.0,
        });
        let pool = HostPool::new(device.clone());
        for fen in read_fens(&cfg_s("QK_FENS", "fens.txt")) {
            let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
            let x = mapper.map(&pool, &[pos], Default::default(), 1).unwrap().to_device(&device).unwrap();
            let out = evaluator.evaluate(&x).unwrap().get("output").unwrap();
            let [v] = out.to_host().unwrap().f32()[..] else { panic!() };
            println!("EVAL {fen} | {}", 400.0 * v);
        }
        return;
    }

    let saved_format: Vec<SavedFormat> =
        ["l0/psqt", "l0/fac", "l0/thr/w", "l0/thr/b", "l1/w", "l1/b", "l2/w", "l2/b", "l3/w", "l3/b"]
            .iter()
            .map(|id| SavedFormat::id(id))
            .collect();

    let data = cfg_s("QK_DATA", "");
    let paths: Vec<&str> = data.split(',').filter(|s| !s.is_empty()).collect();
    assert!(!paths.is_empty(), "QK_DATA must list binpack files");
    let min_ply: u16 = cfg("QK_MINPLY", 8);
    let filter = move |e: &TrainingDataEntry| {
        e.ply >= min_ply
            && e.score != 32002
            && e.score.unsigned_abs() <= 10000
            && !e.pos.is_checked(e.pos.side_to_move())
            && e.mv.mtype() == MoveType::Normal
            && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None
    };
    let reader =
        SfBinpackLoader::new_concat_multiple(&paths, cfg("QK_BUF_MB", 4096), cfg("QK_READ_THREADS", 4), filter);

    let sb: usize = cfg("QK_SB", 40);
    let start: usize = cfg("QK_START", 1);
    let lr0: f32 = cfg("QK_LR0", 1e-3);
    let lr1: f32 = cfg("QK_LR1", 1e-3 * 0.3f32.powi(5));
    let warm: usize = cfg("QK_WARMUP", 500);
    let wdl0: f32 = cfg("QK_WDL0", 0.2);
    let wdl1: f32 = cfg("QK_WDL1", 0.4);
    let save_rate: usize = cfg("QK_SAVE", 10);
    let out_dir = cfg_s("QK_OUT", "checkpoints");
    std::fs::create_dir_all(&out_dir).unwrap();

    let lr_sched = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: lr0, final_lr: lr1, final_superbatch: sb },
        warmup_batches: warm,
    };
    println!("[quirky] data={paths:?} sb={start}..{sb} lr {lr0}->{lr1} warmup {warm} wdl {wdl0}->{wdl1}");

    train(
        &mut optimiser,
        TrainingSchedule {
            steps: TrainingSteps {
                batch_size: cfg("QK_BS", 16_384),
                batches_per_superbatch: cfg("QK_BPS", 6104),
                start_superbatch: start,
                end_superbatch: sb,
            },
            lr_schedule: lr_sched.boxed(),
            log_rate: 128,
        },
        ReadMapLoader::new(
            reader,
            inputs::make_inputs_mapper((&inputs, psqt, output_buckets), threats, scale, wdl::LinearWDL {
                start: wdl0,
                end: wdl1,
            }),
            cfg("QK_MAP_THREADS", 8u8),
        ),
        |_, _, _| {},
        |optimiser, step| {
            let s = step.superbatch();
            if s.is_multiple_of(save_rate) || s == step.final_superbatch() {
                let path = format!("{out_dir}/{name}-{s}");
                save_to_checkpoint(optimiser, &saved_format, &path);
                println!("Saved [{path}]");
            }
        },
    )
    .unwrap();
}
