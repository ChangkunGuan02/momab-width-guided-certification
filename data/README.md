This directory contains the two prepared benchmark instances used by the
computational study.

## Included Prepared Instances

- `spotify_genre_instance_d6.npz`
- `kuairec_cohort_instance_d26.npz`

The files are the canonical instances used by the published result outputs.
Checksums and array schemas are recorded in
`prepared_instances_manifest.json`. Data attribution and redistribution notes
are recorded in `LICENSES.md`.

## Raw Data Provenance

The raw Spotify and KuaiRec files are not redistributed in this artifact.
The preparation scripts in `../src/` are included so that the prepared
instances can be rebuilt from separately downloaded raw data.

Spotify source:

- Dataset: Spotify Tracks dataset mirror on Hugging Face
- URL: `https://huggingface.co/datasets/sfiore/spotify-tracks-dataset`
- License listed by the dataset card used by our scripts: `bsd`

KuaiRec source:

- Dataset: KuaiRec 2.0
- URL: `https://github.com/chongminggao/KuaiRec`
- License in the release used here: Creative Commons
  Attribution-ShareAlike 4.0 International

## Preparation Commands

The Spotify instance treats genres as arms and track feature vectors as reward
samples. The prepared file used in the experiments keeps the six objectives
reported in the paper.
The commands below are written to be run from the artifact root,
`code_publish/`.

```bash
python3 src/prepare_spotify_instance.py \
  --input <spotify_tracks.csv> \
  --outdir data \
  --top-k 114 \
  --min-tracks 200 \
  --rank-by track_count \
  --objectives popularity,danceability,energy,loudness,acousticness,valence
mv data/spotify_genre_instance.npz data/spotify_genre_instance_d6.npz
```

The KuaiRec instance treats videos as arms and user cohorts as objectives.
The prepared file keeps the 26 largest retained cohorts formed from user
activity, follow-count range, and registration-age range.

```bash
python3 src/prepare_kuairec_cohort_instance.py \
  --input-dir <kuairec_root> \
  --outdir data \
  --cohort-cols user_active_degree,follow_user_num_range,register_days_range \
  --cohort-mode joint \
  --max-objectives 26 \
  --top-k 300 \
  --rank-by mean_sum
mv data/kuairec_cohort_instance.npz data/kuairec_cohort_instance_d26.npz
```

To substitute a different prepared instance without editing the scripts, pass
the path through the `SPOTIFY_INSTANCE` or `KUAIREC_INSTANCE` environment
variable when invoking the real-data submit script. These replacement `.npz`
files must come from trusted sources, because the empirical reward arrays use
NumPy object arrays.
