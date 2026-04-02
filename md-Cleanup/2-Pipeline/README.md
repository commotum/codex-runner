# Pipeline Runner

`script-pipeline.py` orchestrates the full md-cleanup flow:

1. scan `3-Books-In`
2. preprocess each book
3. process each extracted section tree
4. move the completed book directory to `4-Books-Out`

Example:

```bash
python md-Cleanup/2-Pipeline/script-pipeline.py
```

Single book:

```bash
python md-Cleanup/2-Pipeline/script-pipeline.py --book 'A New Kind of Science'
```
