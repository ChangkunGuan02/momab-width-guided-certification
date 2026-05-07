# Data Attribution and License Notes

The source code in this artifact is covered by the repository-level MIT
license. The prepared benchmark instances in this directory are derived from
third-party datasets and are distributed subject to the corresponding upstream
data terms.

## Spotify Prepared Instance

File: `spotify_genre_instance_d6.npz`

Source dataset: Spotify Tracks dataset mirror on Hugging Face  
Source URL: `https://huggingface.co/datasets/sfiore/spotify-tracks-dataset`  
License listed by the Hugging Face dataset card used by the preparation script:
`bsd`

Changes made in the prepared instance:

- genres are treated as arms;
- duplicate track IDs are removed within each genre by default;
- six objectives are retained: popularity, danceability, energy, loudness,
  acousticness, and valence;
- reward coordinates are scaled to `[0,1]`;
- 114 genre arms are retained.

The Hugging Face dataset card used by the preparation script lists the license
as `bsd`, but it does not identify the BSD variant. Redistribution of this
prepared instance is based on that upstream license metadata. Users should
consult the upstream dataset card and comply with its terms before
redistributing modified copies.

## KuaiRec Prepared Instance

File: `kuairec_cohort_instance_d26.npz`

Source dataset: KuaiRec 2.0  
Source URL used for license and attribution: `https://github.com/chongminggao/KuaiRec`  
License used for this artifact: Creative Commons Attribution-ShareAlike 4.0
International, as listed by the KuaiRec GitHub repository used for the
manuscript.
License URL: `https://creativecommons.org/licenses/by-sa/4.0/`

Changes made in the prepared instance:

- videos are treated as arms;
- user cohorts are formed from `user_active_degree`,
  `follow_user_num_range`, and `register_days_range`;
- the 26 largest retained cohorts are used as objectives;
- watch-ratio rewards are clipped and normalized to `[0,1]`;
- 300 video arms are retained.

This prepared file is a derived benchmark instance. Users should cite the
KuaiRec dataset and comply with the Creative Commons terms when redistributing
or modifying it.

The artifact pins the prepared `.npz` file through the SHA-256 checksum in
`prepared_instances_manifest.json`. It does not pin a raw upstream archive
checksum.
