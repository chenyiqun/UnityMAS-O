from __future__ import annotations

import copy
import os

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


class CodeJsonlDataset(RLHFDataset):
    """Dataset adapter for code-task json/jsonl rows.

    Expected raw fields include `problem` and `tests`. The adapter creates the
    chat-style `prompt` field needed by RLHFDataset while preserving all original
    fields for the STAR workflow verifier.
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            elif data_file.endswith((".json", ".jsonl")):
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported file format for CodeJsonlDataset: {data_file}")
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        prompt_key = self.prompt_key

        def row_problem_text(row):
            for key in ("problem", "question", "query"):
                value = row.get(key)
                if value:
                    return str(value)
            prompt_value = row.get(prompt_key) or row.get("prompt")
            if isinstance(prompt_value, str):
                return prompt_value
            if isinstance(prompt_value, list):
                texts = []
                for message in prompt_value:
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, str):
                            texts.append(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    texts.append(str(part.get("text", "")))
                                elif isinstance(part, str):
                                    texts.append(part)
                    else:
                        texts.append(str(message))
                return "\n".join(text for text in texts if text).strip()
            return str(prompt_value or "")

        def normalize_row(row):
            problem = row_problem_text(row)
            uid = str(row.get("uid") or row.get("query_id") or row.get("id") or "")
            if not uid:
                uid = (
                    f"{row.get('source') or row.get('datasource') or row.get('data_source') or 'code'}/"
                    f"{row.get('split', 'unknown')}/{row.get('row_id', '')}"
                )
            row[prompt_key] = [{"role": "user", "content": problem}]
            row["query_id"] = uid
            row["data_source"] = str(row.get("source") or row.get("data_source") or row.get("datasource") or "code")
            if row.get("starter_code") is None:
                row["starter_code"] = ""
            if "extra_info" not in row or row["extra_info"] is None:
                row["extra_info"] = {}
            return row

        self.dataframe = self.dataframe.map(normalize_row, desc="Building code-task prompts")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, data_file in enumerate(data_files):
            expanded = os.path.expanduser(str(data_file))
            self.data_files[i] = copy_to_local(src=expanded, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        if isinstance(data_files, (list, tuple)) or data_files.__class__.__name__ == "ListConfig":
            self.original_data_files = copy.deepcopy(list(data_files))
        else:
            self.original_data_files = copy.deepcopy([data_files])
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
