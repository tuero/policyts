# Best-First Search Usage

For all invocations, ensure you are in the project root directory.

For help usage, 
```shell
./build/release-linux/src/bfs/bfs_train --help
./build/release-linux/src/bfs/bfs_test --help
```

## Training
Example invocation:
```shell
./build/release-linux/src/bfs/bfs_train --environment=sokoban --problems_path=problems/sokoban_unfiltered_train.txt --output_dir=experiments/phs/sokoban_s0 --model_path=models/heuristic_convnet.json --search_budget=4000 --inference_batch_size=32 --seed=0 --weight_g=1 --weight_h=1.5 --num_train=10000 --num_validate=1000 --num_threads=8 --device_num=0
```


## Testing
Example invocation:
```shell
./build/release-linux/src/bfs/bfs_train --environment=sokoban --problems_path=problems/sokoban_unfiltered_test.txt --output_dir=experiments/phs/sokoban_s0 --model_path=models/heuristic_convnet.json --search_budget=4000 --inference_batch_size=32 --weight_g=1 --weight_h=1.5 --num_threads=8 --device_num=0
```

