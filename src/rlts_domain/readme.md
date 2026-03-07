# SIIRLTS Usage

For all invocations, ensure you are in the project root directory.

For help usage, 
```shell
./build/release-linux/src/rlts_domain/rlts_domain_train --help
./build/release-linux/src/rlts_domain/rlts_domain_test --help
```

## Training
Example invocation:
```shell
./build/release-linux/src/rlts_domain/rlts_domain --environment=sokoban --problems_path=problems/sokoban_unfiltered_train.txt --output_dir=experiments/rltslh/sokoban_s0 --model_path=models/policy_convnet.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --seed=0 --num_train=10000 --num_validate=1000 --num_threads=8 --device_num=0
```


## Testing
Example invocation:
```shell
./build/release-linux/src/rlts_domain/rlts_domain --environment=sokoban --problems_path=problems/sokoban_unfiltered_test.txt --output_dir=experiments/rltslh/sokoban_s0 --model_path=models/policy_convnet.json --search_budget=4000 --inference_batch_size=32 --mix_epsilon=0.01 --num_threads=8 --device_num=0
```
