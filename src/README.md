# Experiment Source Code

## Usage

1. download and process data 
```bash!
bash data/get_data.sh
bash data/prepare_data.sh
```
2. prepare demographics (will replace existing)
```bash!
python build_demographic.py
```
> builds in `demo/[dataset].pt`

3. run experiments

* train normal recsys (not poisoned): `python recsys.py`
> save in `cache/model_weights/[dataset_modeltype].pth`

* run all experiments sequentially
> The experiments for Table III
```bash!
bash scripts/all_2p.sh -d ciao -a -g <device>
bash scripts/all_2p.sh -d ciao -m -g <device>
bash scripts/all_2p.sh -d epinions -a -g <device>
bash scripts/all_2p.sh -d epinions -m -g <device>
bash scripts/all_2p.sh -d library -a -g <device>
bash scripts/all_2p.sh -d library -m -g <device>
```
> The experiments for Fig. 2 (NOTE: should be conducted after finishing all_2p experiments for the first opponent poison.)
```
bash scripts/all_num.sh -d ciao -g <device>
bash scripts/all_num.sh -d ciao -m -g <device>
bash scripts/all_num.sh -d epinions -g <device>
bash scripts/all_num.sh -d epinions -m -g <device>
bash scripts/all_num.sh -d library -g <device>
bash scripts/all_num.sh -d library -m -g <device>
```
> The experiments for Fig. 4
```
bash scripts/all_opb.sh -d ciao -g <device>
bash scripts/all_opb.sh -d ciao -m -g <device>
bash scripts/all_opb.sh -d epinions -g <device>
bash scripts/all_opb.sh -d epinions -m -g <device>
bash scripts/all_opb.sh -d library -g <device>
bash scripts/all_opb.sh -d library -m -g <device>
```

* run individual experiment
```bash!
python main.py 
    --dataset [ciao/epinions/library] 
    --nop [0,1,2..] 
    --task [mca/ca/ia/none/pga/trial/rev/srwa/none]
    --method [msops/popular/random/none]
```
> Note: msops is msopds.


4. aggregate result
```bash!
python print_tables.py
python quick_fetch_figures.py
```
