import torch

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, input_ids=None, encoder_outputs=None):
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty = length_penalty
        self.input_ids = input_ids
        self.encoder_outputs = encoder_outputs

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_log_probs):

        score = sum_log_probs / len(hyp) ** self.length_penalty

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))

            if len(self.beams) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                self.beams.pop(sorted_scores[0][1])
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_log_probs, cur_len=None):

        if len(self.beams) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_log_probs / cur_len ** self.length_penalty
            done = self.worst_score >= cur_score
            return done
    
def expand_inputs(input_ids, input_encodes, beam_size=1):

    batch_size = input_ids.shape[0]
    input_ids_expand_list = []
    input_encodes_expand_list = []

    for i in range(batch_size):
        temp_t1 = input_ids[i, :].unsqueeze(0).repeat(beam_size, 1)
        temp_t2 = input_encodes[i, :, :].unsqueeze(0).repeat(beam_size, 1, 1)
        input_ids_expand_list.append(temp_t1)
        input_encodes_expand_list.append(temp_t2)

    input_expanded_ids = torch.cat(input_ids_expand_list)
    input_expanded_encodes = torch.cat(input_encodes_expand_list)
    return input_expanded_ids, input_expanded_encodes


