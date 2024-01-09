"""
Credit: I do not remeber where I copied the code base from
        Maybe here: https://github.com/yueyericardo/pkbar, but not sure
        See https://github.com/yueyericardo/pkbar for usuage.

Author: Rockson Ayeman (rockson.agyeman@aau.at, rocksyne@gmail.com)
        Bernhard Rinner (bernhard.rinner@aau.at)

For:    Pervasive Computing Group (https://nes.aau.at/?page_id=6065)
        Institute of Networked and Embedded Systems (NES)
        University of Klagenfurt, 9020 Klagenfurt, Austria.

Date:   Sometime in 2022 (First authored date)

Documentation:
--------------------------------------------------------------------------------
Shows Keras like progress bar for Pytorch training

TODO:   [x] Do proper documentation
        [x] Search where code base was gotten from and credit appropriately
"""

# system modules import
import time
import sys

# 3rd party imports
import numpy as np


class ProgressBar(object):
    """
    Keras-like progress bar.
    
    · Example display 1: show epoch progress and set prefix text
    
                    Epoch: 1/4
                    Prefix_text: 20/20 [=====] - 2s 120ms/step - T. Loss: 1.7684 - T. Pix. Acc.: 0.0819 - T. mPix. Acc.: 0.0537 - T. mIoU: 0.0125
    
    
    · Example display 2: hide epoch progress and set prefix text
                    
                    Prefix_text: 20/20 [=====] - 2s 120ms/step - T. Loss: 1.7684 - T. Pix. Acc.: 0.0819 - T. mPix. Acc.: 0.0537 - T. mIoU: 0.0125
                    
                    
    · Example display 2: hide epoch progress and hide prefix text
                    
                    20/20 [=====] - 2s 120ms/step - T. Loss: 1.7684 - T. Pix. Acc.: 0.0819 - T. mPix. Acc.: 0.0537 - T. mIoU: 0.0125
    
    Arguments:
            prefix_text: see usuage from example
            show_epoch_progress: see usage from examples
            target: Total number of steps expected, None if unknown.
            epoch: Zero-indexed current epoch.
            num_epochs: Total epochs.
            width: Progress bar width on screen.
            verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
            always_stateful: (Boolean) Whether to set all metrics to be stateful.
            stateful_metrics: Iterable of string names of metrics that
                    should *not* be averaged over time. Metrics in this list
                    will be displayed as-is. All others will be averaged
                    by the progbar before display.
            interval: Minimum visual progress update interval (in seconds).
            unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(self, prefix_text=None, show_epoch_progress=True, target=None, epoch=None, num_epochs=None,
                 width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, always_stateful=False,
                 unit_name='step'):
        self.prefix_text = '' if prefix_text is None else prefix_text+": "
        self.show_epoch_progress = show_epoch_progress
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        self.always_stateful= always_stateful
        if (epoch is not None) and (num_epochs is not None):
            
            if self.show_epoch_progress:
                print('Epoch: %d/%d' % (epoch + 1, num_epochs))
        
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            # if torch tensor, convert it to numpy
            if str(type(v)) == "<class 'torch.Tensor'>":
                v = v.detach().cpu().numpy()

            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics and not self.always_stateful:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = (self.prefix_text + '%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)