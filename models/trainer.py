import torch
import time
import operator
import copy
import json
from tqdm import tqdm
from IPython.display import clear_output

from models.utils import make_multilabel_metrics_readable


class MultilabelTrainer:
    def __init__(self, model: torch.nn.Module, device, labels_mapping):
        self.model = model.to(device)
        self.device = device
        self.labels_mapping = labels_mapping

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        patience_early_stopping,
        metrics,
        labels_mapping,
        val_metric="loss",
        val_metric_goal="min",
        val_interval=1,
        final_training=False,
        keep_test_logs=False,
        qat=False,
        min_epochs=10
    ):
        self.val_metric = val_metric
        self.labels_mapping = labels_mapping

        best_model_snapshot = copy.deepcopy(self.model)
        best_model_epoch = 0
        best_metric = 0.0 if val_metric_goal == "max" else float("inf")
        early_stopping = EarlyStoppingCriterion(
            patience=int(patience_early_stopping), mode=val_metric_goal
        )

        since = time.time()
        val_comparison_operator = (
            operator.lt if val_metric_goal == "min" else operator.gt
        )
        training_history = []

        for epoch in range(n_epochs):

            val_metric_res =None
            train_results = self.train_epoch(
                epoch, train_loader, loss_fn, optimizer, metrics
            )

            if epoch % val_interval == 0 or epoch == n_epochs - 1:
                val_results, _ = self.test_epoch(
                    epoch, val_loader, loss_fn, metrics, phase="val"
                )
                
                val_metric_res = val_results[val_metric]["Labels_Average"] if isinstance(val_results, dict) else val_results[val_metric]

                if val_comparison_operator(val_metric_res, best_metric):
                    best_metric = val_metric_res
                    
                    best_model_snapshot = copy.deepcopy(self.model)
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    best_model_epoch = epoch
                    print(f"New best model found at epoch {epoch} with {val_metric}: {best_metric:.4f}")

                training_history.append(
                    {"epoch": epoch, "train": train_results, "val": val_results}
                )

                scheduler.step(val_metric_res)
                

                if epoch >= min_epochs and early_stopping.step(val_metric_res):
                    print(
                        f"No improvement over last {patience_early_stopping} epochs in validation loss. Early stopping..."
                    )
                    break
            else:
                training_history.append({"epoch": epoch, "train": train_results})

            if qat:
             
                if val_metric_res > 0.8:
                    self.model.apply(torch.ao.quantization.disable_observer)
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)


        print("Training complete. Loading best model weights.")
        
        self.model = best_model_snapshot

        total_training_time = time.time() - since
        print(
            f"Training complete in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s"
        )

        training_details = {
            "chosen_val_metric": val_metric,
            "best_epoch": best_model_epoch,
            "total_training_time": total_training_time,
            "training_history": training_history,
            "other_tests": {}
        }

        test_predictions_log = None
        if not final_training:
            test_results, test_predictions_log = self.test_epoch(0, test_loader, loss_fn, metrics, phase="test", keep_logs=keep_test_logs)
            final_loss = test_results["loss"]
            print(f"Final test loss: {final_loss:.4f}")
            training_details["testing_metrics"] = test_results

        return self.model, training_details, test_predictions_log

    def train_epoch(self, epoch, train_loader, loss_fn, optimizer, metrics):
        print(f"\nTraining Epoch: {epoch}")
        self.reset_metrics(metrics)

        self.model.train()
        epoch_start = time.time()
        epoch_loss_acc = 0.0
        iterator = tqdm(train_loader)

        for batch_idx, (features, labels, _) in enumerate(iterator):
            batch_loss = self.process_train_batch(
                batch_idx, features, labels, loss_fn, optimizer, metrics
            )
            epoch_loss_acc += batch_loss.item()

        epoch_loss = epoch_loss_acc / len(train_loader)
        time_elapsed = time.time() - epoch_start

        results = {
            "time_elapsed": time_elapsed,
            "loss": epoch_loss,
        }

        results = self.update_results(results, metrics)

        results = make_multilabel_metrics_readable(results, self.labels_mapping)

        self.log_epoch_results(epoch, results, phase="train")

        if epoch % 5 == 0:
            clear_output()

        return results

    def process_train_batch(self, batch_idx, features, labels, loss_fn, optimizer, metrics):
        features = features.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad()
        outputs = self.model(features)

        # For BCEWithLogitsLoss, we need float targets
        # print(f"outputs shape: {outputs}, labels shape: {labels.shape}")
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        optimizer.step()

        self.update_metrics(metrics, labels, outputs)
        return loss

    def test_epoch(self, epoch, loader, loss_fn, metrics, phase="val", keep_logs=False):
        print(f"\n{phase.capitalize()} Epoch: {epoch}")
        self.reset_metrics(metrics)
        self.model.eval()

        epoch_start = time.time()
        epoch_loss_acc = 0
        iterator = tqdm(loader)

        test_predictions_log = [] if keep_logs else None

        with torch.no_grad():
            for batch_idx, (features, labels, metadata) in enumerate(iterator):
                batch_loss = self.process_test_batch(
                    batch_idx, features, labels, loss_fn, metrics, test_predictions_log=test_predictions_log, metadata=metadata
                )
                epoch_loss_acc += batch_loss.item()

        epoch_loss = epoch_loss_acc / len(loader)
        time_elapsed = time.time() - epoch_start

        results = {
            "time_elapsed": time_elapsed,
            "loss": epoch_loss,
        }

        results = self.update_results(results, metrics)
        results = make_multilabel_metrics_readable(results, self.labels_mapping)

        self.log_epoch_results(epoch, results, phase=phase)

        return results, test_predictions_log

    def process_test_batch(self, batch_idx, features, labels, loss_fn, metrics, metadata=None, test_predictions_log=None):
        features = features.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(features)

        loss = loss_fn(outputs, labels.float())

        if test_predictions_log is not None:
            batch_size = features.shape[0]
            for i in range(batch_size):
                sample_log = {
                    'Filename': metadata['Filename'][i],
                    'Site': metadata['Site'][i],
                    'N_HF_Call_Types': metadata['N_HF_Call_Types'][i].item()
                }
                probs = torch.sigmoid(outputs[i])
                predictions = (probs >= 0.5).float()
                sample_log["true"] = labels[i].cpu().numpy().tolist()
                sample_log["pred"] = predictions.cpu().numpy().tolist()
                sample_log["probs"] = probs.cpu().numpy().tolist()
                test_predictions_log.append(sample_log)

        self.update_metrics(metrics, labels, outputs)
        return loss


    def reset_metrics(self, metrics):
        for metric in metrics.values():
            metric.reset()

    def update_metrics(self, metrics, labels, outputs):
        for name, metric in metrics.items():
            metric.update(outputs, labels.int())

    # def update_results(self, results, metrics):
    #     for name, metric in metrics.items():
    #         if name != "confusion_matrix":
    #             value = metric.compute().item()
    #             results[name] = value
    #         else:
    #             results[name] = metric.compute().cpu().numpy().tolist()
    #     return results

    def update_results(self, results, metrics):
        for name, metric in metrics.items():
            value = metric.compute()
            if name.startswith('multilabel') or name == "confusion_matrix":
                # For multilabel metrics that return a list/tensor of values per class
                results[name] = value.cpu().numpy().tolist()
            else:
                # For scalar metrics
                results[name] = value.item()
        
        return results

    def log_epoch_results(self, epoch, results, phase):
        log_str = f"{phase.capitalize()} Epoch: {epoch} Results - \n"
        log_str += f"loss: {results['loss']:0.3f}, \n"
        
        for metric_name, value in results.items():
            if metric_name not in ['loss', 'time_elapsed', 'confusion_matrix']:
                log_str += f"{metric_name}: {value}, \n"
        
        print(log_str[:-2])


class EarlyStoppingCriterion(object):
    """
    Stop training when a metric has stopped improving.
    """
    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {"min", "max"}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self._best_score = None
        self.is_improved = None

    def step(self, cur_score):
        if self._best_score is None:
            self._best_score = cur_score
            return False
        else:
            if self.mode == "max":
                self.is_improved = cur_score >= self._best_score + self.min_delta
            else:
                self.is_improved = cur_score <= self._best_score - self.min_delta

            if self.is_improved:
                self._count = 0
                self._best_score = cur_score
            else:
                self._count += 1
            return self._count > self.patience

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


