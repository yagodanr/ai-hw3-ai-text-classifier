## Scorecard (fixed validation split, 5-fold out-of-fold)

| candidate | macro_f1 (5-fold) | f1_human | f1_ai | f1_fusion | human<->fusion confusion | macro_f1 de-DE | macro_f1 en-EN | worst domain |
|---|---|---|---|---|---|---|---|---|
| 04_mdeberta | 0.9055 ± 0.0073 | 0.8489 | 0.9928 | 0.8749 | 0.1304 | 0.9040 | 0.9058 | arxiv_papers 0.680 |
| 05_tfidf | 0.8602 ± 0.0077 | 0.7875 | 0.9877 | 0.8052 | 0.1919 | 0.8813 | 0.8544 | gutenberg 0.584 |
| ensemble: weighted average | 0.9164 ± 0.0056 | 0.8670 | 0.9938 | 0.8885 | 0.1157 | 0.9211 | 0.9151 | gutenberg 0.720 |
| ensemble: weighted + class offsets | 0.9177 ± 0.0073 | 0.8697 | 0.9944 | 0.8891 | 0.1150 | 0.9265 | 0.9153 | arxiv_papers 0.734 |

## Error correlation between members

| index | 04_mdeberta | 05_tfidf |
|---|---|---|
| 04_mdeberta | 1.0000 | 0.2630 |
| 05_tfidf | 0.2630 | 1.0000 |

## Leave-one-out gain

| index | leave-one-out gain |
|---|---|
| 04_mdeberta | 0.0563 |
| 05_tfidf | 0.0109 |

## Long-text rules (mDeBERTa, validation)

| rule | macro_f1 | f1_human | f1_ai | f1_fusion | macro_f1_de-DE | macro_f1_en-EN |
|---|---|---|---|---|---|---|
| first+last (04) | 0.9055 | 0.8489 | 0.9928 | 0.8749 | 0.9040 | 0.9058 |
| chunks mean | 0.8700 | 0.7783 | 0.9873 | 0.8444 | 0.8125 | 0.8843 |
| chunks max-fusion | 0.8304 | 0.6909 | 0.9855 | 0.8148 | 0.6845 | 0.8617 |
| first 512 | 0.8282 | 0.7015 | 0.9881 | 0.7951 | 0.6684 | 0.8593 |

## All experiments (top 3 per experiment, validation)

| experiment | model | macro_f1 | f1_human | f1_ai | f1_fusion |
|---|---|---|---|---|---|
| 01_baseline | LogReg C=4 no class_weight / word+char | 0.8601 | 0.7875 | 0.9877 | 0.8052 |
| 01_baseline | LogReg C=4 / word+char | 0.8584 | 0.7880 | 0.9878 | 0.7993 |
| 01_baseline | LogReg C=16 / word+char | 0.8556 | 0.7816 | 0.9867 | 0.7986 |
| 02_hf_research | a) word+char + features + domain / LogReg C=4 | 0.8820 | 0.8240 | 0.9887 | 0.8334 |
| 02_hf_research | base LogReg C=4 / word+char (01 reproduced) | 0.8584 | 0.7880 | 0.9878 | 0.7993 |
| 04_encoder | mdeberta / n=38737 ep=2 len=512 head+tail | 0.9055 | 0.8489 | 0.9928 | 0.8749 |
| 04_encoder | mdeberta / n=500 ep=1 len=512 head+tail | 0.5434 | 0.3105 | 0.7491 | 0.5707 |
| 05_final | FINAL ensemble: weighted + class offsets | 0.9177 | 0.8697 | 0.9944 | 0.8891 |
| 05_long_text | mdeberta / chunks mean | 0.8700 | 0.7783 | 0.9873 | 0.8444 |
| 05_long_text | mdeberta / chunks max-fusion | 0.8304 | 0.6909 | 0.9855 | 0.8148 |
| 05_long_text | mdeberta / first 512 | 0.8282 | 0.7015 | 0.9881 | 0.7951 |

## Submissions

- final: `submissions/05_ensemble.csv` = ensemble: weighted + class offsets

- hedge: `submissions/05_hedge.csv` = 04_mdeberta
