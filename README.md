# Code for Bosch*, N., Achterhold*, J., Leal-Taixé, L., Stueckler, J.: "Planning from Images with Deep Latent Gaussian Process Dynamics"
This repository contains code corresponding to:

Bosch*, N., Achterhold*, J., Leal-Taixé. L., Stueckler, J.: \
**Planning from Images with Deep Latent Gaussian Process Dynamics**\
2nd Annual Conference on Learning for Dynamics and Control (L4DC), 2020 \
(* first two authors contributed equally)

Project page: https://dlgpd.is.tue.mpg.de/ \
Full paper: https://arxiv.org/abs/2005.03770

If you use the code, data or models provided in this repository for your research, please cite our paper as:
```
@InProceedings{bosch2020dlgpd,
  title = {Planning from Images with Deep Latent Gaussian Process Dynamics},
  author = {Bosch, Nathanael and Achterhold, Jan and Leal-Taixe, Laura and Stueckler, Joerg},
  booktitle = {Proceedings of the 2nd Conference on Learning for Dynamics and Control},
  pages = {640--650},
  year = {2020},
  editor = {Alexandre M. Bayen and Ali Jadbabaie and George Pappas and Pablo A. Parrilo and Benjamin Recht and Claire Tomlin and Melanie Zeilinger},
  volume = {120},
  series = {Proceedings of Machine Learning Research},
  address = {The Cloud},
  month = {10--11 Jun},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v120/bosch20a/bosch20a.pdf},
  url = {http://proceedings.mlr.press/v120/bosch20a.html},
}
```

## Pre-trained models and pre-generated data
To access pre-trained models (`pretrained_models.tgz`) and pre-generated data for the OpenAI Gym Pendulum-v0 environment (`pregenerated_data.tgz`),
you can download the respective archive files from the following [repository](https://keeper.mpdl.mpg.de/d/a40985f525124c6fa2f3/).
```
Filename                 SHA1SUM
-----------------------------------------------------------------
pregenerated_data.tgz    1104c8751b65f418c0b776596d5d2ac876e999c8
pretrained_models.tgz    3aac9bb1ba3ac3e3358013583a53bedbdc0a1138
```

## Preliminary requirements
- Create a virtual environment with Python 3.6 and install all required python
dependencies with `pip3 install -r requirements.txt`
- Install `xvfb`  to emulate a GL framebuffer
window for the OpenAI environment (e.g, via `apt install xvfb` )
- Make sure to have CUDA installed (we used CUDA 10.1 with CuDNN 7.5 for our experiments)

## Extract training and evidence data
To use the same data we used for training and evaluating the models
reported in the paper, please extract `pregenerated_data.tgz` to the
data directory (`mkdir -p data/; tar -xvz -f pregenerated_data.tgz -C data/`).
If you skip this step, the data will be generated automatically (however, with different seeds).

## Run training
To run training with default hyperparameters (as in the paper), call
```
python -m dlgpd.training with seed=1
```
For the three models used in the paper, we used `seed ∈ {1,2,3}`.
We use [sacred](https://sacred.readthedocs.io/en/stable/) to configure and log experiment runs.
Thus, parameters of the training can be altered by appending `with parameter=abc` to the training call.
See the [sacred documentation](https://sacred.readthedocs.io/en/stable/) for more details.

## Monitor training
Training is continuously monitored with `tensorboard`. Once training is started,
sacred will emit the run id of the current run as
`INFO - dlgpd - Started run with ID "<ID>"`. You can observe monitored
values by calling `tensorboard --logdir=experiments/` in the root directory
of this repository, where all results are grouped by their respective run id.

## Use pre-trained models for planning
We supply pre-trained models with this package in `pretrained_models.tgz`.
Run `mkdir -p experiments/; tar -xvz -f pretrained_models.tgz -C experiments/` to extract them.

Call
```
python -m dlgpd.planning
  [-variation-name VARIATION_NAME]
  [-evidence-bucket-size BUCKET_SIZE]
  [-evidence-bucket-seed 0]
  [-recollect-evidence]
  ./experiments/dlgpd/pretrained_1/models/epoch2000_dlgpd_model.pt
```
to utilize a trained model for control. `VARIATION_NAME` must be one of
[`standard`, `invactions`, `heavierpole`, `lighterpole`] and describes
the variation of the pendulum environment. `BUCKET_SIZE` is the number of
rollouts used as evidence. You can set `-recollect-evidence` to recollect
evidence for the modified environment.

## Evaluate pre-trained models
To evaluate the pre-trained models we provide, extract the content from
`pretrained_models.tgz` and `pregenerated_data.tgz` as described above into the correct subfolders.
Then, run `./scripts_planning_evaluation/create_planning_jobfile.py`
to create a list of planning jobs for the evaluation we presented in the paper.
Please use a job submission system of your favor to run these jobs.
Note that running all these planning rollouts is computationally expensive,
as ~1300 rollouts are performed, taking ~20mins each on our machines.
After this,
call
`./scripts_planning_evaluation/rollouts_to_dataframe.py`
to collect the total return from these rollouts and
`./scripts_planning_evaluation/plot_rollout_dataframe.py`
to plot statistics.

## License

See [LICENSE.md](LICENSE.md).

For license information on 3rd-party software we use in this project, see [3RD_PARTY_LICENSES.md](3RD_PARTY_LICENSES.md).
