"""Bilby dynamic dynesty with native state checkpoints, without maxcall chunks.

This adapter targets the Bilby/dynesty 3 API used by the project. Installed
packages are not modified. Native checkpoints are distinct from legacy Bilby
resume files and require matching package versions and prior transforms.
"""
import datetime
from pathlib import Path
import time

import bilby
import dynesty
from bilby.core.sampler.dynamic_dynesty import DynamicDynesty
from bilby.core.utils import logger


class NativeDynamicDynesty(DynamicDynesty):
    """Keep one dynamic run open; let dynesty save its current execution state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.new_dynesty_api:
            raise RuntimeError('NativeDynamicDynesty currently requires dynesty >= 3')
        legacy_file = Path(self.resume_file)
        self.resume_file = str(Path(self.outdir) / f'{self.label}_native_dynesty.save')
        if self.resume and legacy_file.exists() and not Path(self.resume_file).exists():
            raise ValueError('Legacy Bilby checkpoint cannot be resumed by the native adapter. Use a new label.')
        self._native_restored = False
        self._native_context = dict(
            schema=1, bilby=bilby.__version__, dynesty=dynesty.__version__,
            parameters=list(self.search_parameter_keys),
            priors={key: repr(value) for key, value in self.priors.items()},
            use_ratio=self.use_ratio,
            proposals=self.proposals, naccept=self.naccept, nact=self.nact,
            maxmcmc=self.maxmcmc, bound=self.kwargs['bound'],
            sample=self.kwargs['sample'])

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        files, directories = super().get_expected_outputs(outdir=outdir, label=label)
        legacy = str(Path(outdir) / f'{label}_resume.pickle')
        native = str(Path(outdir) / f'{label}_native_dynesty.save')
        return [native if filename == legacy else filename for filename in files], directories

    def read_saved_state(self, continuing=False):
        """Restore native state; the parent has already initialized worker globals."""
        if not Path(self.resume_file).is_file():
            logger.info(f'No native checkpoint at {self.resume_file}; starting from the prior')
            return False
        restored = dynesty.DynamicNestedSampler.restore(self.resume_file, pool=self.pool)
        if getattr(restored, '_gcfit_context', None) != self._native_context:
            raise ValueError('Native checkpoint versions, priors or proposal settings differ. Use a new label.')
        # dynesty 3 restore() reconnects all internal samplers' pool/mapper.
        # LogLikelihood no longer has a pool: adding one makes checkpoints unpickleable.
        self.sampler = restored
        self._native_restored = True
        # mtime measures time up to the saved state, excluding queue downtime.
        previous = getattr(restored, '_gcfit_elapsed_before', 0.)
        started = getattr(restored, '_gcfit_started_at', Path(self.resume_file).stat().st_mtime)
        seconds = previous + max(0., Path(self.resume_file).stat().st_mtime-started)
        self.sampling_time = datetime.timedelta(seconds=seconds)
        logger.info(f'Restored native dynamic state {restored.internal_state} from {self.resume_file}')
        return True

    def _run_native(self, checkpoint):
        options = self.sampler_function_kwargs.copy()
        # No checkpoint maxcall/maxiter limit is injected. Explicit user limits
        # remain genuine stopping conditions and are not treated as pauses.
        finished = self.sampler.internal_state.name == 'RUN_DONE'
        if self._native_restored and finished:
            logger.info('Native checkpoint is complete; converting saved results without adding batches')
            return self.sampler.results
        options['resume'] = self._native_restored
        self.sampler._gcfit_context = self._native_context
        self.sampler._gcfit_elapsed_before = self.sampling_time.total_seconds()
        self.sampler._gcfit_started_at = time.time()
        if checkpoint:
            options['checkpoint_file'] = self.resume_file
            options['checkpoint_every'] = self.check_point_delta_t
        else:
            options['checkpoint_file'] = None
        logger.info('Running dynamic dynesty in one uninterrupted run_nested call with native checkpointing')
        self.sampler.run_nested(**options)
        if checkpoint:
            self.write_current_state()
        return self.sampler.results

    def _run_external_sampler_with_checkpointing(self):
        return self._run_native(checkpoint=True)

    def _run_external_sampler_without_checkpointing(self):
        return self._run_native(checkpoint=False)

    def write_current_state(self):
        """Use the native format for both periodic and Bilby signal checkpoints."""
        if getattr(self, 'sampler', None) is None:
            return
        self.sampler._gcfit_context = self._native_context
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        self.sampler.save(self.resume_file)
        logger.info(f'Written native dynesty checkpoint {self.resume_file}')
