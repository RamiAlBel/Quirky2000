# Experiment ledger (baseline: t25p net, engine exp2/src commit "baseline")
Every idea is a switch/variant; accepted only if its test beats the current baseline.
SPRT: fastchess, 6+0.06, H0 0 Elo / H1 +8 Elo, alpha=beta=0.05, 8moves_v3 book.

| id | idea | type | test | result | decision |
|---|---|---|---|---|---|
| S1 | refresh cache (Finny) | engine speed, exact | bench nodes identical; nps | +4% nps (884k->922k) | **ACCEPT** (with S2) |
| S2 | lazy accumulator | engine speed, exact | bench nodes identical; nps | +5%; both +7%; SPRT S1+S2 +14.9 ±9.1 (2334 g), H1 | **ACCEPT** |
| S3 | continuation history | search | SPRT | +25.7 ±12.5 (1312 g), H1 | **ACCEPT** |
| S4 | correction history (CorrDiv 256) | search | SPRT | -13.48 ±13.44 (2568 g), H0 | reject (retry with other params later) |
| S5 | singular extensions (depth>=8, margin 2*d) | search | SPRT | -16.3 ±11.2 (1518 g), H0 | reject (retry w/ other params later) |
| S6 | capture history | search | SPRT | queued | |
| S7 | ProbCut | search | SPRT | queued | |
| S8 | razoring | search | SPRT | queued | |
| S9 | time management (soft/hard + stability) | search | SPRT | +55.41 ±22.67 (902 g), H1 | **ACCEPT** |
| B1 | Syzygy 3-4-5 (Fathom) | bot/engine | SPRT | queued | |
| B2 | own book (July 2200+ games, Polyglot) | bot/engine | SPRT from startpos | book building | |
| B3 | pondering (ponderhit keeps the ponder start as clock start) | bot/engine | ponder_match.py, 6+0.06, wall clocks | +107.5 [+91, +124] (750 g, W/D/L 286/403/61) | **ACCEPT** (needs a spare core on the bot machine) |
| N* | net variants (N0 control, hkb32/16/8, psqt, pow25, ema, pair, w256/384/768, ft8, thr32) | net | val + SPRT vs t25p | GPU queue | |
| B4 | two-net: 128-wide small net for |material|>thr | engine+net | SPRT | small128 net queued | |

Net recipe note: N0 (t25p architecture trained from scratch in one stage) reached only val 0.05168 vs t25p 0.05006:
the 8-neuron head throttles FT learning. All net variants therefore use the t25p two-stage recipe (pipe.sh):
stage 1 full training with acc512's head (16->32 crelu, scale 400), stage 2 t25p head fitted on the frozen FT + 1 polish
epoch (stage2.py). R0 = control (should reproduce t25p); every R* differs from R0 in one factor. One-stage runs
(N0, hkb32/16/8, psqt, pow25) are kept as a secondary comparison under the one-stage recipe.
| N-psqt | PSQT shortcut, one-stage t25 head | net | val | 0.08695 (head died: material-only quality) | judged via Rpsqt |
| N-kb | hkb32 / hkb16 / hkb8, one-stage | net | val | 0.04728 / 0.04732 / 0.04733 vs N0 0.05168 (t25p 0.05006) | hkb32 SPRT vs t25p running |
| N-pow | loss power 2.5, one-stage | net | val | 0.05287 vs 0.05168 (metric biased to MSE) | judged via Rpow25 SPRT |

## Net validation (MSE of tanh(cp/400) on the last 1M old positions, engine-like quantized L1; lower = better)
Run-to-run noise ~1% (Rhkb32 stage1 0.04809 vs one-stage hkb32 0.04728). Elo decided by the net tournament (netT1).
| net | factor vs R0 | val |
|---|---|---|
| t25p | reference (acc512 + head HPO + polish) | 0.05006 |
| R0_s2 | control (two-stage t25p recipe) | 0.04889 |
| hkb8_s2 / hkb16_s2 / hkb32 (1-stage) | king-bucket features 8/16/32, one-stage + stage 2 | **0.04589** / 0.04699 / 0.04728 |
| Rhkb16_s2 / Rhkb32_s2 | king buckets, two-stage | 0.04657 / 0.04733 |
| Rthr32_s2 | threat features (32 buckets) | 0.04674 |
| Rpair_s2 | pairwise FT activation | 0.04867 |
| Rpsqt_s2 | PSQT shortcut | 0.04936 |
| Rft8_s2 | int8 FT weights | 0.04942 |
| Rema_s2 | EMA 0.999 | 0.04960 (no help) |
| Rpow25_s2 | loss power 2.5 | 0.05060 (metric biased) |
| Rw256_s2 / Rw768_s2 | width 256 / 768 | 0.05055 / 0.04872 |
| Rsmall128_s2 | width 128 (two-net small net) | 0.05152 |
| Rw384 | width 384 | dead init (0.367) -> rerun Rw384b |
| N-prune | train 1024 wide (T1024), prune FT lanes by act-std x L1-weight-norm to 512/384, stage 2 w/ 300M polish | net | val + tournament vs same-width from-scratch (hkb8_sf) | queued after T1024 | |
| D-phase | phase-balanced sampling (piece-count bands, tau 0.5 / 1.0), hkb8 July+old | data | val by band + tournament | queued (strata job) | |
| D-hard | hard-example mining: sample error bins of hkb8_sfold_s2 with p ~ mean_err^0.5 | data | val + tournament | queued (strata job) | |

Diagnostic (phase_diag.py, val MSE by pieces on board): data is 49% 25-32 pcs, 30% 17-24, 16% 9-16, 5% 2-8;
error 2-2.5x higher at 9-24 pcs than at 25-32 (hkb8_sfold_s2: 0.026 / 0.057 / 0.057 / 0.042 for 25-32/17-24/9-16/2-8).
| robustness | train_v4 restarts with a fresh init if val > 0.2 after epoch 1 (dead nets: Rw384, hkb8_sf_e20) | infra | - | - | added |
| P384_s2 (pruned T1024 -> 384) | pruning | 0.04758 (worse than 512 from scratch; T1024 teacher weak, redo with T1024b) |
| hkb8_sfold_hard05_s2 | hard-example sampling | 0.04806 vs 0.04097 unstratified -> REJECT |
| hkb8_sfold_phase05 / phase10 (stage 1) | phase-balanced sampling | 0.04195 / 0.04222 vs 0.04200 -> neutral on overall val; per-band + tournament pending |
| C_hkb8pair_sfold, C_thr8pair_sfold, C_thr8_sfold | combinations (submitted 578831-3) | pending |
Phase diag (val MSE by pieces 2-8 / 9-16 / 17-24 / 25-32):
  hkb8_sfold_s2          0.0424 / 0.0566 / 0.0568 / 0.0258  all 0.04097
  phase05_s2 (tau .5)    0.0406 / 0.0555 / 0.0570 / 0.0262  all 0.04099
  phase10_s2 (tau 1)     0.0398 / 0.0554 / 0.0579 / 0.0271  all 0.04167
  hard05_s2              0.0621 / 0.0655 / 0.0651 / 0.0302  all 0.04806  -> REJECT
  => phase balancing trades opening accuracy for endgame (-4..-6% err at 2-16 pcs); Elo decides (netT2).
| hkb8_sf_e20b (20 epochs vs 10) | training length | stage1 0.04078 vs 0.04259 -> longer helps; combos resubmitted with --epochs 20 (578847-50, incl. C_hkb8pair_sfold_ph05) |
| hkb8_sf_e20_s2 | 20 epochs July only | **0.04018** (best so far) |
| hkb8_sft_s2 | distillation (noeval labelled by T1024) | 0.04162 vs 0.04097 no-distill -> no gain with weak teacher |
| T1024b (lr 1e-3) | teacher retry | 0.04431 (T1024 0.04418) vs 512-wide same recipe 0.04259 -> 1024 not stronger at this budget |
| N-prune / distill | verdict | REJECT: pruned P512/P384 (0.04633/0.04758) and distilled hkb8_sft_s2 (0.04162) all worse than plain 512 |
| **C_hkb8pair_sfold_s2** | hkb8 + pairwise + July+old, 20 epochs | stage1 0.03890, **s2 0.03922** (best); xcheck OK; SPRT vs t25p = net_C1 (578904) |
Speed (bench 13, UseFinny+UseLazyAcc, one core): t25p 825k nps | hkb8 828k | hkb8+pair 881k (+7%) | thr8 620k (-25%).
| C_thr8pair_sfold | thr8 + pair + July+old, 20 ep | stage1 **0.03765** (but threats -25% nps) -> head-to-head SPRT vs C_hkb8pair |
| C_thr8_sfold | thr8 + July+old, 20 ep | stage1 0.03975 |
| C_hkb8pair_sfold_ph05 | + phase balance tau .5 | stage1 0.03922 vs 0.03890 -> Elo test |
| C_hkb8pair_sfold_ph05_s2 | phase-balanced combo | 0.03920 (= unbalanced 0.03922) -> SPRT phase_vs_flat |
| C_thr8pair_sfold_s2 | threat+pair combo | **0.03769**, xcheck OK -> SPRT thr_vs_hkb |
| C_thr8_sfold_s2 | threat (no pair) combo | 0.03858 (dominated by thr8pair: slower, less accurate) |
| Syzygy 3-4-5 (SyzygyPath) | SPRT H1 accepted: Elo: 8.65 +/- 6.38, nElo: 13.37 +/- 9.85 | **ACCEPT** |
| net_C1: C_hkb8pair_sfold_s2 vs t25p | SPRT H1: Elo: 185.66 +/- 35.96, nElo: 260.60 +/- 41.75 | **ACCEPT** |
netT1 (cancelled after 2.5h, superseded by combos): standings in ledger/netT1_final.txt
netT1 verdicts: king buckets (hkb8 best +47 vs field, t25p -50); psqt, ft8, pow25, w768 hurt -> REJECT; w256 +17 (speed) -> retry width 256/384 on the combo.
| thr_vs_hkb | threats cost 25% nps | -29 ±16 @794 g, heading to H0 -> threats REJECT (thr e40 cancelled) |
| C_hkb8pair_sfold_w256_s2 | width 256 combo | 0.04204 (512: 0.03922), xcheck OK -> SPRT w256_vs_512 |
| C_hkb8pair_sfold_w384_s2 | width 384 combo | 0.04036, xcheck OK -> SPRT w384_vs_512 |
| C_hkb8pair_sfold_e40 | 40 epochs | stage1 0.03828 (e20 0.03890) |
| C_e40_h16_s2 | e40 + stage-2 head h1=16 | 0.03842 (h1=8: 0.03895) -> SPRT vs C1 |
| C_e40_h16x32_s2 | e40 + head 16->32 | 0.03824 |
| Razoring | stopped after 8802 g: +2.3 ±4.7, no verdict | NEUTRAL -> not adopted (tiny effect; revisit in SPSA) |
Head speed (nps): h8 852k | h16 813k (-4.5%) | h16x32 749k (-10%); h16x32 skipped (only -0.5% err vs h16).
NOTE: speed.sh on bin/dev with a 384 net silently fell back to t25p (bench 475856) - width must match binary.
| w256_vs_512 | SPRT H0: Elo: -10.44 +/- 9.31, nElo: -16.19 +/- 14.42 Games: 2230 | width 256 REJECT |
| phase_vs_flat | SPRT H0: Elo: -2.55 +/- 5.77, nElo: -4.02 +/- 9.11 Games: 5588 | phase balancing REJECT |
| capthist | SPRT H1: Elo: 7.70 +/- 5.88, nElo: 11.85 +/- 9.03 Games: 5684 | **ACCEPT** |
| C_hkb8pair_sfold_e80 | 80 epochs | stage1 0.03820 (e40 0.03828), s2 0.03851 (e40 0.03895) -> saturating |
| w384_vs_512 | SPRT H0: Elo: -3.97 +/- 6.64, nElo: -6.16 +/- 10.29 Games: 4376 | width 384 REJECT -> keep 512 |
| comb0_stc: C1 + finny/lazy/conthist/capthist/TM/syzygy vs t25p all-off, 6+0.06 | SPRT H1: Elo: 353.53 +/- 50.90, nElo: 612.32 +/- 46.77 | **ACCEPT** |
| comb0_ltc: same, 40+0.4 | SPRT H1: Elo: 254.12 +/- 36.52, nElo: 436.12 +/- 44.52 | **ACCEPT** |
| e40h16_vs_C1 | stopped @5598 g: -0.3 ±6.0 | 16-neuron head NEUTRAL -> keep h8 (s2e80h16 cancelled) |
| ProbCut | stopped @6798 g: +1.9 ±5.3 | NEUTRAL -> not adopted |
| Own book (book_jul2200, from startpos, on top of combined config) | SPRT H1: Elo: 21.49 +/- 10.87, nElo: 36.20 +/- 18.25 | **ACCEPT** (caveat: non-book side has little opening variety) |
| corr1024 (before fix) | -24 ±14 @798 g; BUG found: corrhist learned from TB win scores (~31000) -> fixed (2dd9a1b), rerun corrfix |
| e80_vs_C1 | 20000 g cap, no verdict: +3.1 ±3.1 | marginal, keep C1 (bundle) |
| sing3 (singular, SingMargin=3, on combined config) | SPRT H1: +21.7 ±10.5 (1350 g) | **ACCEPT** |
| corrfix (corrhist after TB fix, on combined) | SPRT H0: -47.7 ±17.3 (550 g) | REJECT |
| spsa1 (16 search params, 800 iters x 22 g, 6+0.06) | theta: RfpDepth=9 RfpMargin=81 RfpImp=24 NmpBase=3 NmpDiv=3 NmpEvalDiv=196 LmpBase=4 FutBase=99 FutMul=114 SeeQuiet=26 SeeNoisy=99 HistDiv=5715 LmrBase=85 LmrDiv=206 AspDelta=17 QsFut=200 | verifying: spsa_check |

## Round E (2026-09-28): on top of bundle D (baseline = D config, ledger/D_opts.txt, bin/E_base = src c5a3b8d)
D vs C1 confirmed: +26.4 ±11.8 (6+0.06), +22.7 ±9.4 (40+0.4), both H1.
SPRT 6+0.06, elo0 0 / elo1 6. Bench with all E switches off = 489384 (identical to D).
| id | idea | bench nodes (D 489384) | result | decision |
|---|---|---|---|---|
| E-corr2 | corrhist v2: weighted average (old one was an unbounded sum -> saturated), pawn + non-pawn keys | 529219 | running | |
| E-ttpv | ttPv bit in TT (depth bit 7), non-PV ttPv nodes r-- | 553363 | running | |
| E-histprune | quiet with history < -2000*depth at lmrDepth < 4 pruned | 497097 | running | |
| E-cont4 | continuation history 4 plies back | 713395 | running | |
| E-dblext | double singular extension (non-PV, sv < sBeta-20, <= 6 per line) | 455021 | running | |
| E-nodetm | soft limit x (1.5 - bestmove node share) x 1.35 | (TM only) | running | |
| E-rfpblend | RFP returns (eval+beta)/2 | 520027 | running | |
| E-lmrttcap | quiets r++ when TT move is a capture | 480633 | running | |
| E-wdl20/40 | C1 recipe + WDL blend 0.2 / 0.4 (July results) | net | training | |
| E-edb | Lichess eval DB (410M, deep SF, multi-PV) -> data/evaldb/edb.* (+ best move for policy) | data | extracting | |
| E-pawnhist | pawn history (1024 pawn keys x piece x to) in quiet score | 547638 | running (bin/E2_base, src 2nd E commit) | |
| E-priorbonus | fail-low node -> history+conthist bonus to the opponent's quiet move that led here | 611747 | running | |
| E-aspfh | aspiration fail high -> re-search 1 ply shallower (cumulative) | 468385 | running | |
| E-capfut | capture futility: eval+200+150*lmrDepth+victim <= alpha | 554432 | running | |
| E-deeper | LMR re-search deeper (v > best+40+2d) / shallower (v < best+d) | 480756 | running | |
| E-selfplay | engine `datagen` command (src): self-play from startpos+8 random plies, 5000 nodes/move, D options, REC4 with results; data/selfplay/*.bin (array job datagen.sh) | data | generating (~74 games/min/core, ~114 pos/game) | |
| E-sigma | uncertainty head (training/sigma.py, SIG1): smoke sigma (July labels, 10M) beats table baseline by 10% MSE, cell Spearman 0.42; engine xcheck OK; costs ~12% nps | | SigRef 100: -43.9 [-74,-14] @215 g (miscalibrated: median sigma at pruning nodes = 55 cp) -> stopped | retry SigRef 55 (sigma55, sigma55m30) |
| E-sigtap | sigma from the eval head's hidden layer (free) | | only +2.8% vs table -> not pursued | |
| E-flag | flag mode: opp clock < FlagOppMs and ours >= 2x theirs -> limits x60%, draw contempt 30 cp | TC 10+0 (no increment, so clocks run low), FlagOppMs 3000 | running | |
| E-buckets | output-bucket schemes on frozen C1 FT (training/buckets.py, CPU): pc8 (current) / pc16 / pcq8 / pcq16 / npm8, 2x10M head refit | val | running | |
| E-edb data | eval DB extracted: 409.7M lines -> 279.9M quiet positions (data/evaldb/edb.*) | data | done | |
| E-edbnet | C1 recipe + eval DB in the mix (sf 3.5, old 1, edb 1.5), 20 epochs | net | GPU queue | |
| E-sigedb | sigma retrained on eval-DB labels (30M x 2) | val | running | |
| **E-corr2** | SPRT H1: Elo 24.05 +/- 9.89 (1548 g) | **ACCEPT** |
| **E-dblext** | SPRT H1: Elo 20.66 +/- 8.60 (1566 g) | **ACCEPT** |
| **E-nodetm** | SPRT H1: Elo 14.94 +/- 7.14 (2188 g) | **ACCEPT** |
| E-cont4 / histprune / pawnhist / priorbonus | stopped ~1700-2100 g at +1.0 / -1.3 / +0.6 / -0.4 | NEUTRAL -> not adopted |
| E-sigma (search use) | sigma55 -30 [-44,-17] @965, sigma55m30 H0 -43.5 @852, siglmr -21 [-33,-9] @1223 | REJECT at timed TC (12% nps cost + no visible quality gain); fixed-node test E_signodes (sig_edb) to separate speed from quality |
| E-sigedb | sigma on eval-DB labels: MSE -11.8% vs table, but Spearman 0.66 < table 0.73, cell Spearman 0.34 | weak |
| E-buckets | val (2x10M refit; original C1 head 0.03922): pc8 0.03925, pcq16 0.03925, pc16 0.03933, npm8 0.03950, pcq8 0.03950 | REJECT: bucket rule does not matter (FT already encodes phase) |
| E1 = D + corr2 + dblext + nodetm | ledger/E1_opts.txt; SPRT vs D at 6+0.06 and 40+0.4 | running |
| **E1 vs D 6+0.06** | SPRT H1: Elo 49.21 +/- 13.89 (732 g) | **ACCEPT** |
| Dropped 2026-09-28 04:30 (user: drop GPU nets + unpromising) | GPU nets (wdl20/40, edbnet) cancelled before start; self-play datagen stopped (~9M positions kept in data/selfplay) | |
| stopped as not promising | signodes -9 [-17,-1] @3191 (sigma loses even at fixed nodes -> sigma idea REJECTED); flag -2 @1917; ttpv 0.0 @3136; capfut +3.8 @2821; rfpblend +3.6 @3383; lmrttcap +2.7 @3119 | REJECT |
| kept running | aspfh +9.5 [+2,+17], deeper +9.0 [+1,+17] (vs D); E1 vs D 40+0.4 | |
| **bundle E** (2026-09-28 04:43) | /scratch/ralbe/chess_nnue/bundle_E(.zip): src c4cff49 (zig x86_64-windows-gnu), C1 net, D book; E = D + UseCorr2 UseDblExt UseNodeTM UseAspFH UseDeeper (ledger/E_opts.txt); bench 13 = 574687 nodes | built early on request (aspfh/deeper not concluded) | checks running: E_vs_E1 (6+0.06), E_vs_D_t8 (8 threads, 10+0.1) |
| E-aspfh / E-deeper | SPRT H1 vs D: +11.3 [+4,+18] (3549 g) / +12.5 [+5,+20] (3533 g) | **ACCEPT** (bundle E notes updated) |
| final (cancelled 05:06 on request, no SPRT verdict) | E vs E1 6+0.06: +18.9 [+5.9,+32.0] @790 g; E vs D 8 threads 10+0.1: +66.2 [+34.5,+99.0] @85 g; E1 vs D 40+0.4: +45.5 [+29.2,+61.9] @369 g | all clearly positive -> bundle E confirmed |
