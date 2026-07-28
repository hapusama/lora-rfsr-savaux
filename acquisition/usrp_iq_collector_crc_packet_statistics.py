import collections
import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Convert one CRC byte per decoded packet into live packet statistics."""

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="CRC Packet Statistics",
            in_sig=[np.uint8],
            out_sig=[
                np.float32,
                np.float32,
                np.float32,
                np.float32,
                np.float32,
            ],
        )
        self.total = 0
        self.valid = 0
        self.recent = collections.deque(maxlen=10)

    def work(self, input_items, output_items):
        source = input_items[0]
        count = min(
            len(source),
            *(len(output) for output in output_items),
        )
        for index in range(count):
            current = 1.0 if int(source[index]) != 0 else 0.0
            self.total += 1
            self.valid += int(current)
            self.recent.append(current)

            output_items[0][index] = current
            output_items[1][index] = float(self.total)
            output_items[2][index] = float(self.valid)
            output_items[3][index] = 100.0 * self.valid / self.total
            output_items[4][index] = (
                100.0 * sum(self.recent) / len(self.recent)
            )
        return count