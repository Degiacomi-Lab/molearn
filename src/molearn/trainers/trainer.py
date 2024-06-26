import os
import glob
import shutil
import numpy as np
import time
import torch
from molearn.data import PDBData
import json


class TrainingFailure(Exception):
    pass


class Trainer:
    """
    Trainer class that defines a number of useful methods for training an autoencoder.

    :ivar autoencoder: any torch.nn.module network that has methods ``autoencoder.encode`` and ``autoencoder.decode`` with the weights associated with these operations accessible via ``autoencoder.encoder`` and ``autoencoder.decoder``. This can be set using set_autoencoder
    :ivar _autoencoder_kwargs: kwargs used to initialise the network. Saved in every checkpoint under the key 'kwargs'
    :ivar torch.optim.optimiser optimiser: pytorch optimiser with access to self.autoencoder.parameters()
    :ivar torch.Device device: The device used for all operations.
    :ivar int epoch: the current epoch
    :ivar float best: The best validation score corresponding to the current best checkpoint
    :ivar float best_name: the filename corresponding to self.best
    :ivar float std: Standard deviation of the training dataset. Can be used to unscale structures produced by the network.
    :ivar float mol: Biobox molecule containing a single example frame of the protein being trained on. This can be used to save examples during training. It is also used to save a temporary pdb that may be used to initialise thirdparty packages.
    :ivar torch.Dataloader train_dataloader: Training data
    :ivar torch.Dataloader valid_dataloader: Validation data
    :ivar _data: (:func:`molearn.data <molearn.data.PDBata>` Data object given to :func:`set_data <molearn.trainers.Trainer.set_data>`

    """

    def __init__(self, device=None, log_filename="log_file.dat", json_log=False):
        """
        :param torch.Device device: if not given will be determinined automatically based on torch.cuda.is_available()
        :param str log_filename: (default: 'default_log_filename.json') file used to log outputs to
        :param bool json_log: True to use json.dump to save the log file
        """
        if not device:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        print(f"device: {self.device}")
        self.best = None
        self.best_name = None
        self.epoch = 0
        self.scheduler = None
        self.verbose = True
        self.log_filename = "default_log_filename.csv"
        self.scheduler_key = None
        self.json_log = json_log

    def get_network_summary(self):
        """
        returns a dictionary containing information about the size of the autoencoder.
        """

        def get_parameters(trainable_only, model):
            return sum(
                p.numel()
                for p in model.parameters()
                if (p.requires_grad and trainable_only)
            )

        return dict(
            encoder_trainable=get_parameters(True, self.autoencoder.encoder),
            encoder_total=get_parameters(False, self.autoencoder.encoder),
            decoder_trainable=get_parameters(True, self.autoencoder.decoder),
            decoder_total=get_parameters(False, self.autoencoder.decoder),
            autoencoder_trainable=get_parameters(True, self.autoencoder),
            autoencoder_total=get_parameters(False, self.autoencoder),
        )

    def set_autoencoder(self, autoencoder, **kwargs):
        """
        :param autoencoder: (:func:`autoencoder <molearn.models>`,) torch network class that implements ``autoencoder.encode``, and ``autoencoder.decode``. Please pass the class not the instance
        :param \*\*kwargs: any other kwargs given to this method will be used to initialise the network ``self.autoencoder = autoencoder(**kwargs)``
        """
        if isinstance(autoencoder, type):
            self.autoencoder = autoencoder(**kwargs).to(self.device)
        else:
            self.autoencoder = autoencoder.to(self.device)
        self._autoencoder_kwargs = kwargs

    def set_dataloader(self, train_dataloader=None, valid_dataloader=None):
        """
        :param torch.DataLoader train_dataloader: Alternatively set using ``trainer.train_dataloader = dataloader``
        :param torch.DataLoader valid_dataloader: Alternatively set using ``trainer.valid_dataloader = dataloader``
        """
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if valid_dataloader is not None:
            self.valid_dataloader = valid_dataloader

    def set_data(self, data, **kwargs):
        """
        Sets up internal variables and gives trainer access to dataloaders.
        ``self.train_dataloader``, ``self.valid_dataloader``, ``self.std``, ``self.mean``, ``self.mol`` will all be obtained from this object.

        :param :func:`PDBData <molearn.data.PDBData>` data: data object to be set.
        :param \*\*kwargs: will be passed on to :func:`data.get_dataloader(**kwargs) <molearn.data.PDBData.get_dataloader>`

        """
        if isinstance(data, PDBData):
            self.set_dataloader(*data.get_dataloader(**kwargs))
        else:
            raise NotImplementedError(
                "Have not implemented this method to use any data other than PDBData yet"
            )
        self.std = data.std
        self.mean = data.mean
        self.mol = data.mol
        self._data = data

    def prepare_optimiser(self, lr=1e-3, weight_decay=0.0001, **optimiser_kwargs):
        """
        The Default optimiser is ``AdamW`` and is saved in ``self.optimiser``.
        With no optional arguments this function is the same as doing:
        ``trainer.optimiser = torch.optim.AdawW(self.autoencoder.parameters(), lr=1e-3, weight_decay = 0.0001)``

        :param float lr: (default: 1e-3) optimiser learning rate.
        :param float weight_decay: (default: 0.0001) optimiser weight_decay
        :param \*\*optimiser_kwargs: other kwargs that are passed onto AdamW
        """
        self.optimiser = torch.optim.AdamW(
            self.autoencoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimiser_kwargs,
        )

    def log(self, log_dict, verbose=None):
        """
        Then contents of log_dict are dumped using ``json.dumps(log_dict)`` and printed and/or appended to ``self.log_filename``
        This function is called from :func:`self.run <molearn.trainers.Trainer.run>`

        :param dict log_dict: dictionary to be printed or saved
        :param bool verbose: (default: False) if True or self.verbose is true the output will be printed
        """

        if verbose or self.verbose:
            max_key_len = max([len(k) for k in log_dict.keys()])
            if "epoch" in log_dict:
                cur_epoch = log_dict["epoch"]
                print(f"{'epoch': <{max_key_len+1}}: {cur_epoch}")
            for k, v in log_dict.items():
                if k != "epoch":
                    print(f"{k: <{max_key_len+1}}: {v:.6f}")
            print()

        if not self.json_log:
            # create header if file doesn't exist => first epoch
            if not os.path.isfile(self.log_filename):
                with open(self.log_filename, "a") as f:
                    f.write(f"{','.join([str(k) for k in log_dict.keys()])}\n")

            with open(self.log_filename, "a") as f:
                # just try to format if it is not a Failure
                if "Failure" not in log_dict.values():
                    f.write(f"{','.join([str(v) for v in log_dict.values()])}\n")
                else:
                    dump = json.dumps(log_dict)
                    f.write(dump + "\n")
        else:
            dump = json.dumps(log_dict)
            with open(self.log_filename, "a") as f:
                f.write(dump + "\n")

    def scheduler_step(self, logs):
        """
        This function does nothing. It is called after :func:`self.valid_epoch <molearn.trainers.Trainer.valid_epoch>` in :func:`Trainer.run() <molearn.trainers.Trainer.run>` and before :func:`checkpointing <molearn.trainers.Trainer.checkpoint>`. It is designed to be overridden if you wish to use a scheduler.

        :param dict logs: Dictionary passed passed containing all logs returned from ``self.train_epoch`` and ``self.valid_epoch``.
        """
        pass

    def prepare_logs(self, log_filename, log_folder=None):
        self.log_filename = log_filename
        if log_folder is not None:
            if not os.path.exists(log_folder):
                os.mkdir(log_folder)
            if hasattr(self, "_repeat") and self._repeat > 0:
                self.log_filename = f"{log_folder}/{self._repeat}_{self.log_filename}"
            else:
                self.log_filename = f"{log_folder}/{self.log_filename}"

    def run(
        self,
        max_epochs=100,
        log_filename=None,
        log_folder=None,
        checkpoint_frequency=1,
        checkpoint_folder="checkpoint_folder",
        allow_n_failures=10,
        verbose=None,
        allow_grad_in_valid=False,
    ):
        """
        Calls the following in a loop:

        - :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`
        - :func:`Trainer.valid_epoch <molearn.trainers.Trainer.valid_epoch>`
        - :func:`Trainer.scheduler_step <molearn.trainers.Trainer.scheduler_step>`
        - :func:`Trainer.checkpoint <molearn.trainers.Trainer.checkpoint>`
        - :func:`Trainer.checkpoint <molearn.trainers.Trainer.checkpoint>`
        - :func:`Trainer.log <molearn.trainers.Trainer.log>`

        :param int max_epochs: (default: 100). run until ``self.epoch`` matches max_epochs
        :param str log_filename: (default: None) If log_filename already exists, all logs are appended to the existing file. Else new log file file is created.
        :param str log_folder: (default: None) If not None log_folder directory is created and the log file is saved within this folder
        :param int checkpoint_frequency: (default: 1) The frequency at which last.ckpt is saved. A checkpoint is saved every epoch if ``'valid_loss'`` is lower else when ``self.epoch`` is divisible by checkpoint_frequency.
        :param str checkpoint_folder: (default: 'checkpoint_folder') Where to save checkpoints.
        :param int allow_n_failures: (default: 10) How many times should training be restarted on error. Each epoch is run in a try except block. If an error is raised training is continued from the best checkpoint.
        :param bool verbose: (default: None) set trainer.verbose. If True, the epoch logs will be printed as well as written to log_filename

        """
        self.get_repeat(checkpoint_folder)
        self.prepare_logs(
            log_filename if log_filename is not None else self.log_filename, log_folder
        )
        # if log_filename is not None:
        #    self.log_filename = log_filename
        #    if log_folder is not None:
        #        if not os.path.exists(log_folder):
        #            os.mkdir(log_folder)
        #        self.log_filename = log_folder+'/'+self.log_filename
        if verbose is not None:
            self.verbose = verbose

        for attempt in range(allow_n_failures):
            try:
                for epoch in range(self.epoch, max_epochs):
                    time1 = time.time()
                    logs = self.train_epoch(epoch)
                    time2 = time.time()
                    if allow_grad_in_valid:
                        logs.update(self.valid_epoch(epoch))
                    else:
                        with torch.no_grad():
                            logs.update(self.valid_epoch(epoch))
                    time3 = time.time()
                    self.scheduler_step(logs)
                    if self.best is None or self.best > logs["valid_loss"]:
                        self.checkpoint(epoch, logs, checkpoint_folder)
                    elif epoch % checkpoint_frequency == 0:
                        self.checkpoint(epoch, logs, checkpoint_folder)
                    time4 = time.time()
                    logs.update(
                        epoch=epoch,
                        train_seconds=time2 - time1,
                        valid_seconds=time3 - time2,
                        checkpoint_seconds=time4 - time3,
                        total_seconds=time4 - time1,
                    )
                    self.log(logs)
                    if np.isnan(logs["valid_loss"]) or np.isnan(logs["train_loss"]):
                        raise TrainingFailure("nan received, failing")
                    self.epoch += 1
            except TrainingFailure:
                if attempt == (allow_n_failures - 1):
                    failure_message = (
                        f"Training Failure due to Nan in attempt {attempt}, end now\n"
                    )
                    self.log({"Failure": failure_message})
                    raise TrainingFailure("nan received, failing")
                failure_message = f"Training Failure due to Nan in attempt {attempt}, try again from best\n"
                self.log({"Failure": failure_message})
                if hasattr(self, "best"):
                    self.load_checkpoint("best", checkpoint_folder)
            else:
                break

    def train_epoch(self, epoch):
        """
        Train one epoch. Called once an epoch from :func:`trainer.run <molearn.trainers.Trainer.run>`
        This method performs the following functions:
        - Sets network to train mode via ``self.autoencoder.train()``
        - for each batch in self.train_dataloader implements typical pytorch training protocol:

          * zero gradients with call ``self.optimiser.zero_grad()``
          * Use training implemented in trainer.train_step ``result = self.train_step(batch)``
          * Determine gradients using keyword ``'loss'`` e.g. ``result['loss'].backward()``
          * Update network gradients. ``self.optimiser.step``

        - All results are aggregated via averaging and returned with ``'train_'`` prepended on the dictionary key

        :param int epoch: The epoch is passed as an argument however epoch number can also be accessed from self.epoch.
        :returns:  Return all results from train_step averaged. These results will be printed and/or logged in :func:`trainer.run() <molearn.trainers.Trainer.run>` via a call to :func:`self.log(results) <molearn.trainers.Trainer.log>`
        :rtype: dict
        """
        self.autoencoder.train()
        N = 0
        results = {}
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            self.optimiser.zero_grad()
            train_result = self.train_step(batch)
            train_result["loss"].backward()
            self.optimiser.step()
            if i == 0:
                results = {
                    key: value.item() * len(batch)
                    for key, value in train_result.items()
                }
            else:
                for key in train_result.keys():
                    results[key] += train_result[key].item() * len(batch)
            N += len(batch)
        return {f"train_{key}": results[key] / N for key in results.keys()}

    def train_step(self, batch):
        """
        Called from :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        """
        results = self.common_step(batch)
        results["loss"] = results["mse_loss"]
        return results

    def common_step(self, batch):
        """
        Called from both train_step and valid_step.
        Calculates the mean squared error loss for self.autoencoder.
        Encoded and decoded frames are saved in self._internal under keys ``encoded`` and ``decoded`` respectively should you wish to use them elsewhere.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms] A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return calculated mse_loss
        :rtype: dict
        """
        self._internal = {}
        encoded = self.autoencoder.encode(batch)
        self._internal["encoded"] = encoded
        decoded = self.autoencoder.decode(encoded)[:, :, : batch.size(2)]
        self._internal["decoded"] = decoded
        return dict(mse_loss=((batch - decoded) ** 2).mean())

    def valid_epoch(self, epoch):
        """
        Called once an epoch from :func:`trainer.run <molearn.trainers.Trainer.run>` within a no_grad context.
        This method performs the following functions:
        - Sets network to eval mode via ``self.autoencoder.eval()``
        - for each batch in ``self.valid_dataloader`` calls :func:`trainer.valid_step <molearn.trainers.Trainer.valid_step>` to retrieve validation loss
        - All results are aggregated via averaging and returned with ``'valid_'`` prepended on the dictionary key

          * The loss with key ``'loss'`` is returned as ``'valid_loss'`` this will be the loss value by which the best checkpoint is determined.

        :param int epoch: The epoch is passed as an argument however epoch number can also be accessed from self.epoch.
        :returns: Return all results from valid_step averaged. These results will be printed and/or logged in :func:`Trainer.run() <molearn.trainers.Trainer.run>` via a call to :func:`self.log(results) <molearn.trainers.Trainer.log>`
        :rtype: dict
        """
        self.autoencoder.eval()
        N = 0
        results = {}
        for i, batch in enumerate(self.valid_dataloader):
            batch = batch[0].to(self.device)
            valid_result = self.valid_step(batch)
            if i == 0:
                results = {
                    key: value.item() * len(batch)
                    for key, value in valid_result.items()
                }
            else:
                for key in valid_result.keys():
                    results[key] += valid_result[key].item() * len(batch)
            N += len(batch)
        return {f"valid_{key}": results[key] / N for key in results.keys()}

    def valid_step(self, batch):
        """
        Called from :func:`Trainer.valid_epoch<molearn.trainer.Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict
        """
        results = self.common_step(batch)
        results["loss"] = results["mse_loss"]
        return results

    def learning_rate_sweep(
        self,
        max_lr=100,
        min_lr=1e-5,
        number_of_iterations=1000,
        checkpoint_folder="checkpoint_sweep",
        train_on="mse_loss",
        save=["loss", "mse_loss"],
    ):
        """
        Deprecated method.
        Performs a sweep of learning rate between ``max_lr`` and ``min_lr`` over ``number_of_iterations``.
        See `Finding Good Learning Rate and The One Cycle Policy <https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6>`_

        :param float max_lr: (default: 100.0) final/maximum learning rate to be used
        :param float min_lr: (default: 1e-5) Starting learning rate
        :param int number_of_iterations: (default: 1000) Number of steps to run sweep over.
        :param str train_on: (default: 'mse_loss') key returned from trainer.train_step(batch) on which to train
        :param list save: (default: ['loss', 'mse_loss']) what loss values to return.
        :returns: array of shape [len(save), min(number_of_iterations, iterations before NaN)] containing loss values defined in `save` key word.
        :rtype: numpy.ndarray
        """
        self.autoencoder.train()

        def cycle(iterable):
            while True:
                for i in iterable:
                    yield i

        init_loss = 0.0
        values = []
        data = iter(cycle(self.train_dataloader))
        for i in range(number_of_iterations):
            lr = min_lr * ((max_lr / min_lr) ** (i / number_of_iterations))
            self.update_optimiser_hyperparameters(lr=lr)
            batch = next(data)[0].to(self.device).float()

            self.optimiser.zero_grad()
            result = self.train_step(batch)
            # result['loss']/=len(batch)
            result[train_on].backward()
            self.optimiser.step()
            values.append((lr,) + tuple((result[name].item() for name in save)))
            if i == 0:
                init_loss = result[train_on].item()
            # if result[train_on].item()>1e6*init_loss:
            #    break
        values = np.array(values)
        print("min value ", values[np.nanargmin(values[:, 1])])
        return values

    def update_optimiser_hyperparameters(self, **kwargs):
        """
        Update optimeser hyperparameter e.g. ``trainer.update_optimiser_hyperparameters(lr = 1e3)``

        :param \*\*kwargs: each key value pair in \*\*kwargs is inserted into ``self.optimiser``
        """
        for g in self.optimiser.param_groups:
            for key, value in kwargs.items():
                g[key] = value

    def checkpoint(self, epoch, valid_logs, checkpoint_folder, loss_key="valid_loss"):
        """
        Checkpoint the current network. The checkpoint will be saved as ``'last.ckpt'``.
        If valid_logs[loss_key] is better than self.best then this checkpoint will replace self.best and ``'last.ckpt'`` will be renamed to ``f'{checkpoint_folder}/checkpoint_epoch{epoch}_loss{valid_loss}.ckpt'`` and the former best (filename saved as ``self.best_name``) will be deleted

        :param int epoch: current epoch, will be saved within the ckpt. Current epoch can usually be obtained with ``self.epoch``
        :param dict valid_logs: results dictionary containing loss_key.
        :param str checkpoint_folder:  The folder in which to save the checkpoint.
        :param str loss_key: (default: 'valid_loss') The key with which to get loss from valid_logs.
        """
        valid_loss = valid_logs[loss_key]
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.autoencoder.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
                "loss": valid_loss,
                "network_kwargs": self._autoencoder_kwargs,
                "atoms": self._data.atoms,
                "std": self.std,
                "mean": self.mean,
            },
            f'{checkpoint_folder}/last{f"_{self._repeat}" if self._repeat > 0 else ""}.ckpt',
        )

        if self.best is None or self.best > valid_loss:
            filename = f'{checkpoint_folder}/checkpoint{f"_{self._repeat}" if self._repeat>0 else ""}_epoch{epoch}_loss{valid_loss}.ckpt'
            shutil.copyfile(
                f'{checkpoint_folder}/last{f"_{self._repeat}" if self._repeat>0 else ""}.ckpt',
                filename,
            )
            if self.best is not None:
                os.remove(self.best_name)
            self.best_name = filename
            self.best_epoch = epoch
            self.best = valid_loss

    def load_checkpoint(
        self, checkpoint_name="best", checkpoint_folder="", load_optimiser=True
    ):
        """
        Load checkpoint.

        :param str checkpoint_name: (default: ``'best'``) if ``'best'`` then checkpoint_folder is searched for all files beginning with ``'checkpoint_'`` and loss values are extracted from the filename by assuming all characters after ``'loss'`` and before ``'.ckpt'`` are a float. The checkpoint with the lowest loss is loaded. checkpoint_name is not ``'best'`` we search for a checkpoint file at ``f'{checkpoint_folder}/{checkpoint_name}'``.
        :param str checkpoint_folder:  Folder whithin which to search for checkpoints.
        :param bool load_optimiser: (default: True) Should optimiser state dictionary be loaded.
        """
        if checkpoint_name == "best":
            if self.best_name is not None:
                _name = self.best_name
            else:
                ckpts = glob.glob(checkpoint_folder + "/checkpoint_*")
                indexs = [x.rfind("loss") for x in ckpts]
                losses = [float(x[y + 4 : -5]) for x, y in zip(ckpts, indexs)]
                _name = ckpts[np.argmin(losses)]
        elif checkpoint_name == "last":
            _name = f"{checkpoint_folder}/last.ckpt"
        else:
            _name = f"{checkpoint_folder}/{checkpoint_name}"
        checkpoint = torch.load(_name, map_location=self.device)
        if not hasattr(self, "autoencoder"):
            raise NotImplementedError(
                "self.autoencoder does not exist, I have no way of knowing what network you want to load checkoint weights into yet, please set the network first"
            )

        self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
        if load_optimiser:
            if not hasattr(self, "optimiser"):
                raise NotImplementedError(
                    "self.optimiser does not exist, I have no way of knowing what optimiser you previously used, please set it first."
                )
            self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        self.epoch = epoch + 1

    def get_repeat(self, checkpoint_folder):
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        if not hasattr(self, "_repeat"):
            self._repeat = 0
            for i in range(1000):
                if not os.path.exists(
                    checkpoint_folder
                    + f'/last{f"_{self._repeat}" if self._repeat>0 else ""}.ckpt'
                ):
                    break  # os.mkdir(checkpoint_folder)
                else:
                    self._repeat += 1
            else:
                raise Exception(
                    "Something went wrong, you surely havnt done 1000 repeats?"
                )


if __name__ == "__main__":
    pass
