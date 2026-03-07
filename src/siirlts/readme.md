# SIIRLTS Usage

For all invocations, ensure you are in the project root directory.

$\sqrt{\text{LTS}}\text{-L}$, $\sqrt{\text{LTS}}\text{-H}$, and $\sqrt{\text{LTS}}\text{-LH}$ are all implemented as the same underlying algorithm, controlled with parameters `alpha` and `beta` which is the multiplicative factor in front of the clustering and heuristic rerooter respectively:
- $\sqrt{\text{LTS}}\text{-L}$: `alpha=1`, `beta=0`
- $\sqrt{\text{LTS}}\text{-H}$: `alpha=0`, `beta=1`
- $\sqrt{\text{LTS}}\text{-LH}$: `alpha=1`, `beta=1`

For help usage, 
```shell
./build/release-linux/src/siirlts/siirlts_train --help
./build/release-linux/src/siirlts/siirlts_test --help
```

## Training
Example invocation:
```shell
./build/release-linux/src/siirlts/siirlts_train --environment=sokoban --problems_path=problems/sokoban_unfiltered_train.txt --output_dir=experiments/rltslh/sokoban_s0 --model_path=models/twoheaded_convnet.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --seed=0 --num_train=10000 --num_validate=1000 --num_threads=8 --cluster_level=half --device_num=0 --alpha=1 --beta=1
```


## Testing
Example invocation:
```shell
./build/release-linux/src/siirlts/siirlts_test --environment=sokoban --problems_path=problems/sokoban_unfiltered_test.txt --output_dir=experiments/rltslh/sokoban_s0 --model_path=models/twoheaded_convnet.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --num_threads=8 --cluster_level=half --device_num=0 --alpha=1 --beta=1
```
