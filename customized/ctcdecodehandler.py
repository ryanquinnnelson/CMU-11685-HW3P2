"""
Everything related to CTCDecode.
"""
__author__ = 'ryanquinnnelson'


class CTCDecodeHandler:

    def __init__(self, labels, model_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_id,
                 log_probs_input):
        self.labels = labels
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta
        self.cutoff_top_n = cutoff_top_n
        self.cutoff_prob = cutoff_prob
        self.beam_width = beam_width  # beam_width=1 (greedy search); beam_width>1 (beam search)
        self.num_processes = num_processes
        self.blank_id = blank_id
        self.log_probs_input = log_probs_input

    def get_ctcdecoder(self):
        import ctcdecode
        return ctcdecode.CTCBeamDecoder(self.labels, self.model_path, self.alpha, self.beta, self.cutoff_top_n,
                                        self.cutoff_prob, self.beam_width, self.num_processes, self.blank_id,
                                        self.log_probs_input)
