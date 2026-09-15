from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Stop epoch-based training when a validation metric stops improving.

    This hook is intentionally small and depends only on metrics already written
    by MMDetection's evaluation hook into ``runner.log_buffer.output``.
    """

    def __init__(self,
                 monitor,
                 rule='greater',
                 patience=40,
                 min_delta=0.0,
                 interval=1):
        if rule not in ('greater', 'less'):
            raise ValueError("rule must be either 'greater' or 'less'")
        if patience < 1:
            raise ValueError('patience must be positive')
        if interval < 1:
            raise ValueError('interval must be positive')
        self.monitor = monitor
        self.rule = rule
        self.patience = patience
        self.min_delta = float(min_delta)
        self.interval = interval
        self.best_score = None
        self.best_epoch = None
        self.bad_epochs = 0

    def _is_better(self, score):
        if self.best_score is None:
            return True
        if self.rule == 'greater':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.interval != 0:
            return

        metrics = getattr(runner.log_buffer, 'output', {})
        if self.monitor not in metrics:
            return

        score = float(metrics[self.monitor])
        if self._is_better(score):
            self.best_score = score
            self.best_epoch = current_epoch
            self.bad_epochs = 0
            runner.logger.info(
                'EarlyStoppingHook: best %s improved to %.6f at epoch %d',
                self.monitor, score, current_epoch)
            return

        self.bad_epochs += 1
        runner.logger.info(
            'EarlyStoppingHook: %s=%.6f did not improve from %.6f '
            'for %d/%d epochs',
            self.monitor, score, self.best_score, self.bad_epochs,
            self.patience)
        if self.bad_epochs >= self.patience:
            runner.logger.info(
                'EarlyStoppingHook: stopping at epoch %d; best %s=%.6f '
                'at epoch %d',
                current_epoch, self.monitor, self.best_score,
                self.best_epoch)
            runner._max_epochs = current_epoch
