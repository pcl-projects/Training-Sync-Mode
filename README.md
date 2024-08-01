# The Implementation Code for STAR

This repo contains the implementation code for STAR.

/ps/ is for the PS architecure and /all_reduce/ is for the All-Reduce architecture.

The list below shows the main components for each method.
- **Straggler prediction**: code_for_training_data/train.py (class LstmRegr)
- **Heuristic mode determination**: training_sgd*.py
- **PGNS**: gradient_validation/ and pgns.py
- **ML mode determination**: regression.py
- **Resource consideration**: res_dist_managing.py
- **PS assignment**：blossom/
- **Tree**: training_sgd*.py



# A how-to guide

The PGNS data is stored in /ps/gradient_validation and /all-reduce/gradient_validation.

To run STAR, first pretrain LstmRegr for straggler prediction in a separate cluster.
Modify /ps/job_scheduler.py or /all_reudce/job_scheduler.py and set the IPs of servers to use in rpc_master_addrs; set cmd_client_max_num to the number of servers.
```python
cmd_listener_port = 30009
cmd_client_max_num = 5
rpc_master_addrs = ["172.31.92.17", "172.31.89.15", "172.31.85.164", "172.31.88.45", "172.31.80.104"]
```

Then modify the method to use in pyfname like
```python
run_remote_py(sockm, pyfname="training_sgd.py", path_prefix="/home/ubuntu/sgd/", **kwargs)
```
training_sgd.py is for the PS architecture, and training_sgd_allreduce.py is for the all-reduce architecture.

On the first server, run
```bash
python3 job_scheduler.py
```

On each server, run
```bash
python3 cmd_client.py
```
Remeber to modify host in cmd_client.py to the IP of the first server before you run cmd_client.py
```python
host = "IP_OF_THE_FIRST_SERVER"
```

This will output logs for jobs. Use these logs to train LstmRegr using
On each server, run
```bash
python3 code_for_training_data/train.py
```

Then use the same steps above, run the experiment in the evaluation environment.

## Comparison method

### lb_bsp

Run server.py on the first server, then run client.py on each server.
```bash
python3 server.py

python3 client.py
```

### lgc

Run exp_testing_lgc.py on the first server, then run cmd_client.py on each server.
```bash
python3 exp_testing_lgc.py

python3 cmd_client.py
```

### sync-switch
Run rivanna_code/exp_testing_switch.py on the first server, then run rivanna_code/cmd_client.py on each server.
```bash
python3 rivanna_code/exp_testing_switch.py

python3 rivanna_code/cmd_client.py
```


### zeno++
Pls refer to https://github.com/xcgoner/iclr2020_zeno_async
