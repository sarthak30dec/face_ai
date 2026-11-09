# Dataset validation report

## wider_face: valid_with_warnings

| split | annotated images | on disk | missing | corrupt | duplicates | faces | degenerate | invalid flag | out-of-bounds | <4px |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 12880 | 12880 | 0 | 0 | 2 | 159420 | 27 | 2399 | 1 | 1456 |
| val | 3226 | 3226 | 0 | 0 | 0 | 39708 | 11 | 585 | 1 | 440 |

Evaluation ground truth: wider_easy_val.mat, wider_face_val.mat, wider_hard_val.mat, wider_medium_val.mat; val faces .mat=39708 txt=39708

Byte-identical images shared by train and val: 1
- PROBLEM: 1 byte-identical images appear in both train and val
- note: __MACOSX/ is macOS archive metadata from eval_tools.zip; ignored

## lfw: valid

- images: 13233, identities: 5749 (>=2 imgs: 1680, >=3: 901, >=5: 423)
- image sizes: ['250x250']
- corrupt: 0, misnamed: 0, duplicate groups: 2
- pairs.txt: 10 folds, 6000 pairs (3000 same / 3000 different), missing referenced images: 0
- note: 2 groups of byte-identical images (kept; reported)
