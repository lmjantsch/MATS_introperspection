from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Dict, Callable, Any
import torch
from collections import UserDict

class Collator:
    def __init__(self, collate_fn: Dict[str, Callable[[str, List[Any]], Dict]] = None):
        self.collate_fn = collate_fn or {}

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
            
        keys = batch[0].keys()
        batch_out = {}

        for key in keys:
            column_data = [row[key] for row in batch]

            if key in self.collate_fn:
                batch_out.update(self.collate_fn[key](key, column_data))
                continue
            
            processed_data = self.process_column(key, column_data)
            
            if isinstance(processed_data, (dict, UserDict)):
                batch_out.update(processed_data)
            else:
                batch_out[key] = processed_data

        batch_out['length'] = len(batch)
        return batch_out
    
    def process_column(self, key: str, data: List[Any]) -> Any:
        """
        Overridable method for default column processing.
        Returns either the value to be assigned to 'key', or a dict to merge.
        """
        return data
    

class TensorCollator(Collator):
    def __init__(self, collate_fn: Dict[str, Callable] = None, padding: bool = False, padding_value: int = 0):
        super().__init__(collate_fn)
        self.padding = padding
        self.padding_value = padding_value

    def process_column(self, key: str, data: List[Any]) -> Any:
        if data and isinstance(data[0], torch.Tensor):
            if self.padding:
                return max_pad_sequence(data, padding_value=self.padding_value)
            return torch.stack(data)
        
        return data

        

class TokenizeCollator(TensorCollator):
    """
    Extends TensorCollator to handle tokenization, then falls back to Tensor logic
    for other columns (stacking/padding).
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        text_key: str = 'prompts', 
        collate_fn: Dict[str, Callable] = None,
        padding: bool = True
    ):
        super().__init__(collate_fn, padding=padding) 
        self.tokenizer = tokenizer
        self.text_key = text_key

    def process_column(self, key: str, data: List[Any]) -> Any:
        if key == self.text_key:
            return self.tokenizer(
                data,
                return_tensors='pt',
                padding=True,
                truncation=True 
            )
        return super().process_column(key, data)

def max_pad_sequence(sequence: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """
    Pads a list of tensors to the maximum shape in the batch and stacks them.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in sequence):
        raise TypeError(f"Expected all elements to be tensors")

    if not sequence:
        return torch.tensor([])

    num_dim = len(sequence[0].shape)
    max_shape = []

    for tensor in sequence:
        if len(tensor.shape) != num_dim:
            raise ValueError(f"Expected all tensors to have same number of dimension.")
        max_shape.append(tensor.shape)

    max_shape_tensor = torch.tensor(max_shape)
    target_shape = max_shape_tensor.max(dim=0).values.tolist()

    padded_tensors = []
    for tensor in sequence:
        curr_shape = tensor.shape

        tensor_slice = tuple(slice(0, dim) for dim in curr_shape)
        padded_tensor = torch.full(target_shape, padding_value, dtype=tensor.dtype, device=tensor.device)

        padded_tensor[tensor_slice] = tensor
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=0)