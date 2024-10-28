import os
import torch
import torch.nn.functional as F

# Define special tokens
BOS_IDX = 1
EOS_IDX = 58 #69

class Predictor:
    """
    Predictor class for generating predictions using a trained model.
    """
    def __init__(self, config):
        """
        Initialize Predictor object.

        Args:
        - config: Configuration object containing model parameters
        """
        self.config = config
        self.device = torch.device(self.config.device)

        # Get the model
        self.model = self.get_model()
        self.model.to(self.device)

        # Load the best checkpoint
        self.logs_dir = os.path.join(self.config.root_dir, self.config.experiment_name)
        path = os.path.join(self.logs_dir, "best_checkpoint.pth")
        self.model.load_state_dict(torch.load(path)["state_dict"])
        
        # Set the model to evaluation mode
        self.model.eval()
        
    def get_model(self):
        from algorithms.xval_transformers.model.seq2seq import Model
        model = Model(num_encoder_layers=self.config.num_encoder_layers,
                        num_decoder_layers=self.config.num_decoder_layers,
                        emb_size=self.config.embedding_size,
                        nhead=self.config.nhead,
                        src_vocab_size=self.config.src_vocab_size,
                        tgt_vocab_size=self.config.tgt_vocab_size,
                        input_emb_size=self.config.input_emb_size,
                        max_input_points=self.config.max_input_points,
                        )
        
        return model
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def greedy_decode(self, src, num_array, src_mask, max_len, start_symbol, src_padding_mask=None):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        dim = 1

        memory = self.model.encode(src, num_array, src_mask)
        memory = memory.to(self.device)
        dim = 1
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):

            tgt_mask = (self.generate_square_subsequent_mask(ys.size(1), self.device).type(torch.bool)).to(self.device)

            out = self.model.decode(ys, memory, tgt_mask)
            prob = self.model.generator(out[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=dim)
            if next_word == EOS_IDX:
                break

        return ys


    def predict(self, x, num_array):
        self.model.eval()
        src = x
        num_tokens = src.shape[1]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        src_padding_mask = torch.zeros(1, num_tokens).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, num_array, src_mask, max_len=256, start_symbol=BOS_IDX, src_padding_mask=src_padding_mask).flatten()

        return tgt_tokens


class BeamPredictor(Predictor):
    """
    Predictor class for generating predictions using a trained model.
    """
    def __init__(self, config):
        """
        Initialize Predictor object.

        Args:
        - config: Configuration object containing model parameters
        """
        super().__init__(config)


    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def beam_search_decode(self, src, num_array, src_mask, max_len, start_symbol, beam_size=5, num_equations=3, src_padding_mask=None):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)

        memory = self.model.encode(src, num_array, src_mask).to(self.device)
        beams = [(torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device), 0)]  # (sequence, score)

        completed_sequences = []

        for _ in range(max_len - 1):
            new_beams = []
            for ys, score in beams:
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1), self.device).type(torch.bool).to(self.device)
                out = self.model.decode(ys, memory, tgt_mask)
                prob = self.model.generator(out[:, -1])
                log_probs = F.log_softmax(prob, dim=-1)

                # Get top beam_size candidates
                top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)

                for i in range(beam_size):
                    next_word = top_indices[0, i].item()
                    next_score = score + top_log_probs[0, i].item()

                    new_seq = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                    if next_word == EOS_IDX:
                        completed_sequences.append((new_seq, next_score))
                    else:
                        new_beams.append((new_seq, next_score))

            # Sort and prune to retain only top beam_size beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams

            # Stop if enough sequences have been completed
            if len(completed_sequences) >= num_equations:
                break

        # Sort all completed sequences and return top `num_equations`
        completed_sequences.sort(key=lambda x: x[1], reverse=True)
        best_sequences = [seq for seq, score in completed_sequences[:num_equations]]
        return best_sequences

    def predict(self, x, num_array, beam_size=5, num_equations=3):
        self.model.eval()

        src = x
        num_tokens = src.shape[1]

        src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool).to(self.device)
        src_padding_mask = torch.zeros(1, num_tokens).type(torch.bool).to(self.device)
        tgt_tokens = self.beam_search_decode(src, num_array, src_mask, max_len=256, start_symbol=BOS_IDX,
                                                beam_size=beam_size, num_equations=num_equations,
                                                src_padding_mask=src_padding_mask)

        return tgt_tokens
        
