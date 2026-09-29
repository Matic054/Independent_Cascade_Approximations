# Dataset links

The raw network datasets are **not redistributed in this repository**.
Download the datasets from the original/public sources below and place the
corresponding files in this `data/` directory before running the real-world
benchmark.

| Dataset | Expected local file | Source |
|---|---|---|
| NetHEPT | `NetHEHT.txt` | https://github.com/jxshang/IMPC/blob/master/NetHEPT.txt |
| WikiVote | `wiki-Vote.txt.gz` | https://snap.stanford.edu/data/wiki-Vote.html |
| Enron | `email-Enron.txt.gz` | https://snap.stanford.edu/data/email-Enron.html |
| Epinions | `soc-Epinions1.txt.gz` | https://snap.stanford.edu/data/soc-Epinions1.html |
| Slashdot | `soc-Slashdot0902.txt.gz` | https://snap.stanford.edu/data/soc-Slashdot0902.html |
| CondMat | `ca-CondMat.txt.gz` | https://snap.stanford.edu/data/ca-CondMat.html |
| cit-HepTh | `cit-HepTh.txt.gz` | https://snap.stanford.edu/data/cit-HepTh.html |

## Notes

- The loader currently expects the NetHEPT file under the historical filename
  `NetHEHT.txt`. You can either save/rename the downloaded NetHEPT file to that
  name, or update the corresponding path in `utilities.py`.
- The SNAP dataset pages contain both the download links and the recommended
  dataset citations.
- The benchmark preprocessing may remove self-loops and/or transform undirected
  edges into the directed representation required by `ICGraph`, depending on
  the dataset loader.
