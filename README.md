# DocCoder: Generating Code by Retrieving and Reading Code Docs

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
