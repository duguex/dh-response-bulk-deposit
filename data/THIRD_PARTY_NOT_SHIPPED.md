# Third-party assets referenced but not redistributed

| Asset | Where it comes from |
|---|---|
| HamGNN-family pretrained checkpoint (base model, incl. charge look-up table) | published models of Zhong et al., npj Comput. Mater. 9, 182 (2023), Zenodo: 10.5281/zenodo.8147630 / 10.5281/zenodo.8147631 |
| OpenMX binary + numerical-atomic-orbital basis / pseudopotential files | OpenMX official distribution (http://www.openmx-square.org) |

Label samples here were produced with those tools under their published settings
(`config/dh_bulk_manifest.yaml`, `scripts/diff_hamgnn/build_dh_label_cache.py`);
raw OpenMX `*.scfout` job outputs (~110 GB) are intentionally excluded — the
packed, QC-gated label samples are the machine-readable record the manuscript consumes.
