# Phase 1 — data-quality report

## Section-7 implausible-value audit (computed | doc)

| check | computed | doc | ok |
| --- | --- | --- | --- |
| empty_param_rows | 2403 | 2403 | yes |
| empty_param_records | 1594 | 1594 | yes |
| MAP_eq_0 | 86 | 86 | yes |
| PaO2_eq_0 | 3 | 3 | yes |
| SaO2_eq_0 | 2 | 2 | yes |
| Weight_neg1 | 1001 | 1001 | yes |
| Weight_le20 | 353 | 353 | yes |
| Weight_gt300 | 1 | 1 | yes |
| Height_lt100 | 17 | 17 | yes |
| Height_gt250 | 11 | 11 | yes |
| MechVent_rows | 91644 | 91644 | yes |

MechVent distinct values: [1.0] (existence flag only — cardiovascular stays capped 0/1).

## Record duration

- last observation (h): min 0.0, p5 46.3, median 47.5, max 48.0
- records ending before 36 h: 83
- records ending before 42 h: 171
- records ending before 46 h: 406
- records ending before 47 h: 1020

## Target-window coverage (% of 12,000 records with >=1 valid value)

```
 origin_h   window  GCS   HR  Creatinine  Platelets  Bilirubin  PaO2  FiO2  MAP|NIMAP
       24  [6,30)h 98.3 98.4        96.6       95.6       27.5  66.2  61.5       98.4
       30 [12,36)h 98.3 98.4        95.6       94.8       24.7  62.4  58.6       98.3
       36 [18,42)h 98.2 98.4        95.0       94.2       22.6  59.0  55.5       98.3
       42 [24,48)h 98.1 98.2        96.3       95.5       21.7  56.5  52.1       98.2
```

## Target table

```
          sofa_total  sofa_now  sofa_delta  deteriorate_24h  cardio_instability  target_window_empty
origin_h                                                                                            
24             5.275     5.747      -0.483            0.046               0.684                0.002
30             4.982     5.264      -0.294            0.038               0.653                0.002
36             4.810     4.970      -0.172            0.042               0.627                0.002
42             4.631     4.798      -0.179            0.039               0.603                0.003
```

- rows: 48000  (12000 records x 4 horizons)
- deterioration (SOFA rises >= 2 in 24 h) base rate: 0.041

## Feature table

- shape: (48000, 409)  (407 feature columns)
- columns fully populated: 60; median column NaN fraction: 0.18
- most-missing feature: v_Cholesterol_slope12h (1.00)

## Split

```
            n   pct  sofa_mean  sofa_p25  sofa_p75  death_rate  sofa_missing   icu1   icu2   icu3   icu4
train  8400.0  70.0       6.66       3.0       9.0       0.142         282.0  0.146  0.211  0.358  0.285
val    1800.0  15.0       6.66       3.0       9.0       0.143          61.0  0.155  0.210  0.353  0.282
test   1800.0  15.0       6.67       3.0       9.0       0.142          60.0  0.148  0.210  0.362  0.280
```
