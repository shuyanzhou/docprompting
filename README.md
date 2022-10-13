# DocPrompting: Generating Code by Retrieving the Docs
[[Paper]](https://arxiv.org/pdf/2207.05987.pdf)
## Code
Code comming soon.

## Data
The `data` folder contains the two benchmarks we curated or re-splitted.
* tldr
* conala

On each dataset, we provide 
1. Natural language intent (entry `nl`)
2. Oracle code (entry `cmd`) 
  * Bash for tldr
  * Python for conala
3. Oracle docs (entry `oracle_man`) 
  * In the data files, we only provide the manual ids, their contents could be found in the `{tldr, conala}_docs.json` of `docs.zip`.

## Resources 
* [tldr](https://github.com/tldr-pages/tldr) github repo
* [conla](https://conala-corpus.github.io)

## Citation
```
@article{zhou22doccoder,
    title = {DocPrompting: Generating Code by Retrieving the Docs},
    author = {Shuyan Zhou and Uri Alon and Frank F. Xu and Zhiruo Wang and Zhengbao Jiang and Graham Neubig},
    year = {2022},
    url = {https://arxiv.org/pdf/2207.05987.pdf}
}
```
