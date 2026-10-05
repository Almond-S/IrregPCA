"""The joint-loss history is optional: skipping it must not change the fit."""

from __future__ import annotations

import torch

from irregpca import IrregPCA, IrregPCAConfig, fit_irreg_pca


class TestTrackJointLoss:
    def test_default_records_one_entry_per_validation_step(self, small_dataset):
        sample_ids, locations, values = small_dataset
        result = IrregPCA(n_components=1, epochs=4, patience=100, random_state=3).fit(
            sample_ids=sample_ids, locations=locations, values=values
        )
        h = result.history
        n_steps = sum(len(t) for t in h.train_losses)
        assert len(h.joint_train) == n_steps
        assert len(h.joint_valid) == n_steps

    def test_disabled_leaves_history_empty_and_fit_bit_identical(self, small_dataset):
        sample_ids, locations, values = small_dataset
        kw = {
            "sample_ids": sample_ids,
            "locations": locations,
            "values": values,
            "n_components": 2,
            "epochs": 4,
            "patience": 100,
            "random_state": 3,
        }
        with_joint = fit_irreg_pca(**kw, track_joint_loss=True)
        without = fit_irreg_pca(**kw, track_joint_loss=False)

        assert without.history.joint_train == []
        assert without.history.joint_valid == []
        # the per-model histories -- what early stopping reads -- are untouched
        assert without.history.train_losses == with_joint.history.train_losses
        assert without.history.valid_losses == with_joint.history.valid_losses
        assert without.history.best_epochs == with_joint.history.best_epochs

        # under grid integration the joint loss is a pure read-out, so the
        # fitted functions are identical to the last bit
        grid = torch.linspace(0, 1, 20).unsqueeze(-1)
        assert torch.equal(without.mean(grid), with_joint.mean(grid))
        assert torch.equal(without.components(grid), with_joint.components(grid))

    def test_callbacks_receive_none_when_disabled(self, small_dataset):
        events = []
        sample_ids, locations, values = small_dataset
        IrregPCA(
            n_components=1,
            epochs=3,
            patience=100,
            callbacks=[lambda e: events.append(e.copy())],
            track_joint_loss=False,
        ).fit(sample_ids=sample_ids, locations=locations, values=values)
        epoch_events = [e for e in events if e["stage"] == "epoch_end"]
        assert epoch_events
        assert all(e["joint_train_loss"] is None for e in epoch_events)
        assert all(e["joint_valid_loss"] is None for e in epoch_events)

    def test_config_field_round_trips(self, small_dataset):
        sample_ids, locations, values = small_dataset
        cfg = IrregPCAConfig(n_components=1, epochs=2, patience=100, track_joint_loss=False)
        assert cfg.track_joint_loss is False
        result = IrregPCA(config=cfg).fit(
            sample_ids=sample_ids, locations=locations, values=values
        )
        assert result.history.joint_train == []
