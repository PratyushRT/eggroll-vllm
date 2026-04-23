import re
import numpy as np
from datasets import load_dataset
from typing import List, Literal, Optional
from egg_img import EGG_IMG, CHICK_IMG

# Alternate confidence prompt/regex for the `<conf>...</conf>` template variant.
# (The DCPO-verbose template + regex live in dcpo_prompt.py and remain default.)
CONF_TAGS_DIRECTIVE = (
    "\n\nPlease provide your reasoning, then output your final answer inside "
    "\\boxed{} and your confidence inside <conf>...</conf> tags "
    "(a float in [0,1]).\n"
    "\nExample:\n"
    "\\boxed{42}\n"
    "<conf>0.83</conf>\n"
)
CONF_TAGS_RE = re.compile(r"<conf>\s*([01]?\.\d+|\d+)\s*</conf>")

def general_get_fitness(task_obj, generations, truncateds, answer, pass_at_k: bool = False):
        if len(generations) == 0:
            # Edge case: no generations (shouldn't happen in normal operation)
            return 0.0, (), np.array([])

        fitnesses, model_answers = zip(*[task_obj.get_fitness_single_sample(g, t, answer) for g, t in zip(generations, truncateds)])
        fitnesses = np.array(fitnesses)
        if pass_at_k:
            fitness = np.max(fitnesses)
        else:
            fitness = np.mean(fitnesses)
        return fitness, model_answers, fitnesses, {}
        
def extract_model_answer(text, ans_format="none"):
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore =[
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        if ans_format == "none":
            match = re.findall(regex_pattern, text)
            if match:
                match = match[-1] # take the last regex match
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                text = match.strip()

                for regex in regexes_to_ignore:
                    text = re.sub(regex, "", text)
                return text, "answer extracted"
            else:
                # print("NO REGEX MATCH FOUND")
                return None, "No regex match found"

        elif ans_format == "boxed":
            splits = text.split("boxed{")
            if len(splits) < 2:
                return None, "No `boxed{` found"
            else:
                text = splits[-1].strip() # take the last `boxed{`
                
                match = re.findall(regex_pattern, text)
                if match:
                    match = match[0] # take the first regex match
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    text = match.strip()

                    for regex in regexes_to_ignore:
                        text = re.sub(regex, "", text)
                    return text, "answer extracted"
                else:
                    return None, "No regex match found"
        elif ans_format == "answer_tags":
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                
                for regex in regexes_to_ignore:
                    text = re.sub(regex, "", text)
            
                return text, "answer extracted"
            else:
                return None, "No `<answer>` tags found"
        else:
            raise ValueError(f"Unknown {ans_format=}")

class ZerosTask:
    """Debug task where model rewarded for outputting zeros."""

    def __init__(self, batch_size, max_tokens):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompts = [
            "Hello, my name is",
            "Write some random numbers:",
            "Output 3 numbers and then stop:",
            # "Output zeros:",
        ]

    def get_batch(self):
        indices = np.arange(self.batch_size) % len(self.prompts)
        batch_prompts = [self.prompts[i] for i in indices]
        return batch_prompts, [None for _ in batch_prompts]
       
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        return sum(c == "0" for c in generation)/self.max_tokens, None
    
class RandomTask:
    """Debug task where model is rewarded for guessing a random number.
    Useful for testing pass@k objective."""
    def __init__(self, batch_size, max_random_number, seed, answer_format="none"):
        self.batch_size = batch_size
        self.prompt = "Pick a random number between 1 and " + str(max_random_number) + " (inclusive)."
        self.ans_format = answer_format
        if self.ans_format == "none":
            pass
        elif self.ans_format == "boxed":
            self.prompt += " Format your pick in \\boxed{}."
        else:
            raise ValueError(f"Unknown {self.ans_format=}")
        self.prompt = f"User: {self.prompt}\n\nAssistant:"
        self.max_random_number = max_random_number
        self.rng = np.random.default_rng(seed)

    def get_batch(self):
        batch_prompts = [self.prompt for _ in range(self.batch_size)]
        batch_answers = self.rng.integers(1, self.max_random_number+1, size=self.batch_size).tolist()
        return batch_prompts, batch_answers
    
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        model_answer, _ = extract_model_answer(generation, ans_format=self.ans_format)
        try:
            model_answer = int(model_answer)
        except:
            model_answer = None
        is_correct = (model_answer is not None) and (model_answer == int(answer))
        return 1.0 if is_correct else 0.0, model_answer

from gem.utils.math_grader import extract_answer, grade

def boxed_reward_fn(model_answer, gt_answer, fast=False,):
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    return is_correct

class MathTask:
    def __init__(self, batch_size, seed, tokenizer=None, dataset_name="gsm8k", datset_size=None, apply_chat_template=False, answer_format="none"):
        self.dataset_name = dataset_name
        dataset_names_dict = {
            "gsm8k": ("axon-rl/GSM-8k", "train", True),
            "asdiv2k": ("axon-rl/ASDIV-2k", "train", True),
            "math12k": ("axon-rl/MATH-12k", "train", True),
            "orz57k": ("axon-rl/ORZ-57k", "train", True),
            "deepscaler40k": ("axon-rl/DeepScaleR-40K", "train", True),
            "math-eval": ("axon-rl/math-eval", ["math", "amc", "olympiad_bench", "minerva", "aime24"], False),
        }
        assert dataset_name.lower() in dataset_names_dict, f"Unknown dataset_name {dataset_name}. Supported: {list(dataset_names_dict.keys())}"
        dataset_name, splits, is_train = dataset_names_dict[dataset_name.lower()]
        self.is_train = is_train
        if is_train:
            self.dataset = load_dataset(dataset_name, split=splits)
            self.dataset = self.dataset.shuffle(seed=seed)
            if datset_size is not None:
                self.dataset = self.dataset.select(range(datset_size))
        else:
            self.split_names = splits
            self.dataset = load_dataset(dataset_name)
            # Add gsm8k and asdiv subsets for math-eval
            if dataset_name == "axon-rl/math-eval":
                gsm8k_subset = load_dataset("axon-rl/GSM-8k", split="train").shuffle(seed=seed).select(range(500))
                self.dataset['gsm8k'] = gsm8k_subset
                self.split_names.append('gsm8k')
                asdiv_subset = load_dataset("axon-rl/ASDIV-2k", split="train").shuffle(seed=seed).select(range(500))
                self.dataset['asdiv'] = asdiv_subset
                self.split_names.append('asdiv')
                aime25_set = load_dataset("math-ai/aime25", split="test").shuffle(seed=seed)
                self.dataset['aime25'] = aime25_set
                self.split_names.append('aime25')
            for split in self.split_names:
                self.dataset[split] = self.dataset[split].shuffle(seed=seed)
        self.apply_chat_template = apply_chat_template
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ans_format = answer_format
        if is_train:
            self.idx = 0

    @staticmethod
    def check_correct(generation: str, gt_answer: str, ans_format: str = "none") -> bool:
        """Check if the action is correct."""
        # get correct answers from the dataset entry
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = gt_answer
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        # check against all possible correct answers
        if ans_format == "answer_tags":
            model_answer, _ = extract_model_answer(generation, ans_format = ans_format)
        else:
            model_answer = extract_answer(generation)
        if model_answer is None:
            is_correct = False
        else:
            for correct_answer in correct_answers:
                is_correct = boxed_reward_fn(model_answer, correct_answer, fast=True)
                if is_correct:
                    break
        return is_correct, model_answer
    
    def _format_conversation(self, example):
        if self.ans_format == "answer_tags":
            instruction_str = "Please reason step-by-step concisely."
        else:
            instruction_str = "Please reason step-by-step concisely, and put your final answer within \\boxed{ }."
        
        problem = f"{example['problem']}\n{instruction_str}"
        if self.apply_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {problem}\nAssistant: <think"
        
    def _format_examples(self, examples):
        batch_prompts = [self._format_conversation(example) for example in examples]
        batch_answers = [example["answer"] for example in examples]    
        return batch_prompts, batch_answers

    def get_batch(self):
        assert self.is_train, f"get_batch can only be called on a train dataset, not on {self.dataset_name=}."
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples = [self.dataset[i] for i in indices]
        return self._format_examples(examples)
        
    def get_eval_batch(self):
        assert self.is_train == False, f"get_eval_batch can only be called in eval mode, not on {self.dataset_name=}."
        indices = np.arange(self.batch_size)
        examples = []
        for split in self.split_names:
            split_dataset = self.dataset[split]
            split_length = len(split_dataset)
            examples.extend([split_dataset[i % split_length] for i in indices])
        return self._format_examples(examples)
    
    def get_fitness(self, generations, truncateds, gt_answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, truncateds, gt_answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, truncated, gt_answer):
        if truncated:
            return 0.0, None
        is_correct, model_answer = self.check_correct(generation, gt_answer, ans_format = self.ans_format)
        return 1.0 if is_correct else 0.0, model_answer


# ---------------------------------------------------------------------------
# DCPO-compatible calibrated math task
#
# Adds verbalized confidence to the rollout (matches the DCPO reference
# implementation at https://github.com/icip-cas/DCPO). Three reward variants
# are supported, selected via `reward_variant`:
#
#   "instance" — DCPO-I:  R = C - 0.3 * (C - q)^2        (sample-level Brier)
#   "hybrid"   — DCPO full: R = C - 0.5 * (0.5*C + 0.5*mean(C_group) - q)^2
#                          (requires samples_per_prompt >= 2; uses group mean)
#   "rlcr"     — ESvPG analytical form: R = C - λ * (q - C)^2
#
# Correctness uses DCPO's own grader (`dcpo_grader.compute_score`) so that
# judgments match the DCPO baseline table 1-for-1.
#
# Confidence is parsed from "CONFIDENCE: <float>" in the generation using the
# DCPO regex (dcpo_prompt.CONF_RE). Parse failures during training default to
# q=0.0 (matches DCPO reward managers).
# ---------------------------------------------------------------------------
try:
    from dcpo_prompt import (
        CONF_RE,
        DCPO_CONFIDENCE_DIRECTIVE,
        parse_conf_for_training,
        strip_conf,
    )
    from dcpo_grader import compute_score as dcpo_compute_score
    _HAVE_DCPO_HELPERS = True
except Exception as _dcpo_import_err:
    _HAVE_DCPO_HELPERS = False
    _DCPO_IMPORT_ERR = _dcpo_import_err


def _parse_conf_with_regex(text: str, regex) -> tuple[float, str]:
    """Like dcpo_prompt.parse_conf_for_training but uses an arbitrary regex.

    Returns (conf, status) where status is "ok" iff the regex matched and
    float() succeeded. Callers use status=="ok" to distinguish a true parse
    from the q=0.0 fallback.
    """
    match = regex.search(text)
    if match:
        try:
            conf = float(match.group(1))
        except ValueError:
            return 0.0, "parse_fail"
        return max(0.0, min(1.0, conf)), "ok"
    return 0.0, "no_match"


class CalibratedMathTask(MathTask):
    """DCPO-compatible calibrated math task for ES training.

    Differences from MathTask:
      * Appends the DCPO verbalized-confidence directive to every prompt.
      * When `apply_chat_template=True` and the tokenizer supports it, passes
        `enable_thinking=False` (Qwen3-style) so the model doesn't burn tokens
        on <think>...</think> before the CONFIDENCE line.
      * Reward blends correctness and a Brier-style calibration term.
      * `get_fitness` (not just `get_fitness_single_sample`) is overridden so
        that the "hybrid" variant can use the group-mean correctness over the
        samples_per_prompt rollouts for a single prompt — this is the analog
        of DCPO's G=8 group in ES.
    """

    _REWARD_VARIANTS = {
        "instance",
        "hybrid",
        "rlcr",
        "rlcr_hybrid_loo",
        # A1/A2 triage: adds anchor-aware retention + wrong-confidence-anchor
        # penalties + an explicit truncation penalty on top of rlcr_hybrid_loo.
        # A1 = rho=0.8 (hybrid LOO target); A2 = rho=1.0 (strict per-sample Brier).
        "rlcr_hybrid_loo_retention",
    }

    _PROMPT_TEMPLATES = {"dcpo_verbose", "conf_tags"}

    def __init__(
        self,
        batch_size,
        seed,
        tokenizer=None,
        dataset_name: str = "deepscaler40k",
        datset_size=None,
        apply_chat_template: bool = True,
        reward_variant: str = "hybrid",
        lambda_cal: float = 0.5,          # λ for RLCR; weight for Hybrid
        instance_weight: float = 0.3,     # DCPO-I weight (fixed in their code)
        enable_thinking: bool = False,    # Qwen3 chat-template flag
        format_reward_enabled: bool = True,  # RLCR-only: +/- bonus on parseable format
        prompt_template: Literal["dcpo_verbose", "conf_tags"] = "dcpo_verbose",
        # rlcr_hybrid_loo knobs
        rho: float = 0.5,                 # soft target mix: T = rho*C + (1-rho)*C_bar_{-i}
        gamma_fmt: float = 0.05,          # format bonus (valid format)
        gamma_bad: float = 1.0,           # invalid penalty (truncated or unparseable)
        # rlcr_hybrid_loo_retention (A1/A2 triage) knobs — applied only on prompts
        # flagged is_anchor=True (base model solves ≥threshold of K samples).
        lambda_retention: float = 0.20,         # penalty on (1-C) for anchor prompts
        lambda_wrong_conf_anchor: float = 0.50, # extra penalty on q^2 when wrong & anchor
        lambda_trunc: float = 0.25,             # per-sample truncation penalty (any prompt)
        # Anchor batch mixing: when both `anchor_indices` and `anchor_frac`>0 are
        # provided, `get_batch()` returns an `anchor_frac`-fraction of anchor
        # prompts sampled uniformly from the anchor pool, with the remainder
        # drawn via the usual round-robin through the shuffled dataset.
        anchor_indices=None,                    # iterable[int] — indices into shuffled dataset
        anchor_frac: float = 0.0,               # [0, 1] — fraction of each batch to force-anchor
        anchor_rng_seed: int = 0,               # seed for the anchor sampler RNG
    ):
        if not _HAVE_DCPO_HELPERS:
            raise ImportError(
                f"CalibratedMathTask requires dcpo_prompt + dcpo_grader modules: "
                f"{_DCPO_IMPORT_ERR}"
            )
        if reward_variant not in self._REWARD_VARIANTS:
            raise ValueError(
                f"reward_variant must be one of {self._REWARD_VARIANTS}, got {reward_variant!r}"
            )
        if prompt_template not in self._PROMPT_TEMPLATES:
            raise ValueError(
                f"prompt_template must be one of {self._PROMPT_TEMPLATES}, got {prompt_template!r}"
            )
        # Force boxed answer format; the DCPO directive requires \boxed{}.
        super().__init__(
            batch_size=batch_size,
            seed=seed,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            datset_size=datset_size,
            apply_chat_template=apply_chat_template,
            answer_format="boxed",
        )
        self.reward_variant = reward_variant
        self.lambda_cal = float(lambda_cal)
        self.instance_weight = float(instance_weight)
        self.enable_thinking = bool(enable_thinking)
        self.format_reward_enabled = bool(format_reward_enabled)
        self.prompt_template = prompt_template
        self.rho = float(rho)
        self.gamma_fmt = float(gamma_fmt)
        self.gamma_bad = float(gamma_bad)
        self.lambda_retention = float(lambda_retention)
        self.lambda_wrong_conf_anchor = float(lambda_wrong_conf_anchor)
        self.lambda_trunc = float(lambda_trunc)

        # Anchor batch mixing state.
        if anchor_indices is None:
            self._anchor_indices = np.array([], dtype=np.int64)
            self._anchor_set = set()
        else:
            arr = np.asarray(list(anchor_indices), dtype=np.int64)
            # Guard against indices past the end of the dataset.
            max_idx = len(self.dataset) if self.is_train else 0
            if max_idx > 0:
                arr = arr[arr < max_idx]
            self._anchor_indices = arr
            self._anchor_set = set(int(x) for x in arr)
        self.anchor_frac = float(anchor_frac)
        if self.anchor_frac < 0.0 or self.anchor_frac > 1.0:
            raise ValueError(f"anchor_frac must be in [0, 1], got {self.anchor_frac}")
        if self.anchor_frac > 0.0 and self._anchor_indices.size == 0:
            raise ValueError(
                "anchor_frac > 0 but no anchor_indices provided — refuse to silently"
                " fall back to ordinary batches."
            )
        self._anchor_rng = np.random.default_rng(int(anchor_rng_seed))

        # Resolve active directive + regex once at construction time.
        if prompt_template == "conf_tags":
            self._conf_directive = CONF_TAGS_DIRECTIVE
            self._conf_regex = CONF_TAGS_RE
        else:
            self._conf_directive = DCPO_CONFIDENCE_DIRECTIVE
            self._conf_regex = CONF_RE

    # ----- batch assembly with anchor mixing --------------------------

    def get_batch(self):
        """Return a batch of (prompts, answers) plus a per-example `is_anchor` mask.

        When `anchor_frac > 0`, the first `round(anchor_frac * batch_size)`
        slots are filled with prompts sampled uniformly from the anchor pool,
        and the remainder are drawn via the usual round-robin through the
        shuffled dataset (inherited from MathTask). Callers that don't care
        about anchors can ignore the 3rd return value — existing tasks.py
        callers of MathTask.get_batch (plain 2-tuple) are unaffected because
        this override lives on CalibratedMathTask only.
        """
        assert self.is_train, (
            f"get_batch can only be called on a train dataset, not on {self.dataset_name=}."
        )

        bs = int(self.batch_size)
        n_anchor = int(round(self.anchor_frac * bs))
        # Never exceed the pool; never return all-anchor when caller asked for < 1.0.
        n_anchor = min(n_anchor, bs)

        n_ordinary = bs - n_anchor
        # Ordinary slots: round-robin as in MathTask.
        ordinary_indices = (np.arange(self.idx, self.idx + n_ordinary) %
                            len(self.dataset)).tolist() if n_ordinary > 0 else []
        self.idx += n_ordinary

        # Anchor slots: sample without replacement from the pool (with
        # replacement if pool is smaller than n_anchor).
        if n_anchor > 0 and self._anchor_indices.size > 0:
            replace = self._anchor_indices.size < n_anchor
            picks = self._anchor_rng.choice(
                self._anchor_indices, size=n_anchor, replace=replace
            )
            anchor_indices_sampled = [int(x) for x in picks]
        else:
            anchor_indices_sampled = []

        all_indices = ordinary_indices + anchor_indices_sampled
        examples = [self.dataset[i] for i in all_indices]
        prompts, answers = self._format_examples(examples)
        # True iff the underlying shuffled-dataset index is in the anchor set.
        # (An "ordinary" prompt can happen to fall on an anchor index — honour
        # that, since the reward maths cares about *whether the base model
        # solves it*, not about which batch slot it came from.)
        is_anchor = [int(i) in self._anchor_set for i in all_indices]
        return prompts, answers, is_anchor

    # ----- prompt construction ----------------------------------------

    def _format_conversation(self, example):
        problem = f"{example['problem']}{self._conf_directive}"
        if self.apply_chat_template:
            if self.tokenizer is None:
                raise RuntimeError(
                    "CalibratedMathTask with apply_chat_template=True requires a tokenizer."
                )
            # Qwen3 chat template accepts enable_thinking; other templates ignore extra kwargs.
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Tokenizer template doesn't accept enable_thinking — fall back.
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"User: {problem}\nAssistant:"

    # ----- correctness via DCPO grader (not gem-llm) ------------------

    @staticmethod
    def _dcpo_is_correct(generation: str, gt_answer) -> tuple[bool, Optional[str]]:
        """Grade a generation against the ground-truth using DCPO's is_equiv.

        Returns (is_correct, extracted_boxed_answer_or_None).
        """
        from dcpo_grader import last_boxed_only_string, remove_boxed

        # Strip both possible confidence markers so the grader sees a clean answer
        # regardless of which prompt template was used.
        cleaned = strip_conf(generation)
        cleaned = re.sub(r"<conf>\s*[0-9.]+\s*</conf>", "", cleaned).strip()
        try:
            boxed = last_boxed_only_string(cleaned)
            model_answer = remove_boxed(boxed) if boxed is not None else None
        except Exception:
            model_answer = None

        if isinstance(gt_answer, (list, tuple)):
            is_correct = any(
                dcpo_compute_score(cleaned, str(gt)) > 0.5 for gt in gt_answer
            )
        else:
            gt_str = str(gt_answer) if not isinstance(gt_answer, str) else gt_answer
            is_correct = dcpo_compute_score(cleaned, gt_str) > 0.5
        return is_correct, model_answer

    # ----- single-sample reward (used by instance + rlcr) -------------

    def _single_sample_reward(self, C: float, q: float) -> float:
        if self.reward_variant == "instance":
            return C - self.instance_weight * (C - q) ** 2
        if self.reward_variant == "rlcr":
            return C - self.lambda_cal * (q - C) ** 2
        raise RuntimeError("hybrid reward must be computed in get_fitness")

    def _has_boxed(self, generation: str) -> bool:
        """Detect a parseable \\boxed{...} answer in the generation."""
        from dcpo_grader import last_boxed_only_string, remove_boxed
        try:
            boxed = last_boxed_only_string(generation)
            if boxed is None:
                return False
            return remove_boxed(boxed) is not None
        except Exception:
            return False

    def did_parse_confidence(self, text: str) -> bool:
        """Return True iff the active confidence regex matched in `text`
        (not the q=0.0 fallback path)."""
        _, status = _parse_conf_with_regex(text, self._conf_regex)
        return status == "ok"

    def did_parse_answer(self, text: str) -> bool:
        """Return True iff a \\boxed{...} answer is parseable from `text`."""
        return self._has_boxed(text)

    def _rlcr_format_bonus(self, generation: str, parsed_ok: bool, truncated: bool) -> float:
        """RLCR format reward: +0.1 if BOTH boxed and confidence parsed,
        -1.0 otherwise. Truncated rollouts count as malformed."""
        if not self.format_reward_enabled:
            return 0.0
        if truncated:
            return -1.0
        if parsed_ok and self._has_boxed(generation):
            return 0.1
        return -1.0

    def get_fitness_single_sample(self, generation, truncated, gt_answer):
        """Per-rollout reward. For hybrid, the group-mean term isn't available
        at this level — callers should invoke `get_fitness` instead. Falls back
        to instance-style reward if someone calls this in hybrid mode (e.g. for
        ad-hoc inspection)."""
        if truncated:
            # truncation → C=0, q=0 → R = 0 under all three variants.
            return 0.0, None

        q, status = _parse_conf_with_regex(generation, self._conf_regex)
        parsed_ok = (status == "ok")
        is_correct, model_answer = self._dcpo_is_correct(generation, gt_answer)
        C = 1.0 if is_correct else 0.0

        if self.reward_variant == "hybrid":
            # Degrade gracefully: with a group of size 1, mean_C = C_i, so
            # (0.5*C + 0.5*C - q)^2 = (C - q)^2, reward weight stays 0.5.
            R = C - self.lambda_cal * (C - q) ** 2
        elif self.reward_variant in ("rlcr_hybrid_loo", "rlcr_hybrid_loo_retention"):
            # Single-sample path: no cross-member LOO info available — fall
            # back to self-target (T = C) so R = C - lambda*(q-C)^2 + format.
            R = C - self.lambda_cal * (q - C) ** 2
            if self.format_reward_enabled:
                valid_fmt = parsed_ok and self._has_boxed(generation)
                if valid_fmt:
                    R += self.gamma_fmt
                else:
                    R -= self.gamma_bad
            return R, model_answer
        else:
            R = self._single_sample_reward(C, q)

        if self.reward_variant == "rlcr":
            R = R + self._rlcr_format_bonus(generation, parsed_ok, truncated=False)
        return R, model_answer

    # ----- group-aware aggregation (hybrid uses this) -----------------

    def get_fitness(self, generations, truncateds, gt_answer, pass_at_k: bool = False):
        """Override general_get_fitness so hybrid can use the group-mean of
        C_i across the `samples_per_prompt` rollouts for this single prompt.

        Returns (fitness, model_answers, fitnesses, metadata) — the 4-tuple
        contract expected by the training loop.
        """
        n = len(generations)
        if n == 0:
            return 0.0, (), np.array([]), {}

        # Pass 1: grade every rollout and parse confidence.
        C_vals = np.zeros(n, dtype=np.float64)
        q_vals = np.zeros(n, dtype=np.float64)
        parsed_ok = np.zeros(n, dtype=np.bool_)
        boxed_ok = np.zeros(n, dtype=np.bool_)
        trunc_mask = np.zeros(n, dtype=np.bool_)
        model_answers: list = [None] * n
        for i, (g, t) in enumerate(zip(generations, truncateds)):
            if t:
                trunc_mask[i] = True
                # C=0, q=0 for truncated rollouts → R=0 under all variants.
                continue
            q, status = _parse_conf_with_regex(g, self._conf_regex)
            q_vals[i] = q
            parsed_ok[i] = (status == "ok")
            boxed_ok[i] = self._has_boxed(g)
            is_correct, model_answer = self._dcpo_is_correct(g, gt_answer)
            C_vals[i] = 1.0 if is_correct else 0.0
            model_answers[i] = model_answer

        # Pass 2: compute per-rollout reward given the variant.
        if self.reward_variant == "hybrid":
            # Group = all non-truncated rollouts for this prompt (DCPO analog of G).
            valid = ~trunc_mask
            if valid.any():
                mean_C_group = float(C_vals[valid].mean())
            else:
                mean_C_group = 0.0
            target = 0.5 * C_vals + 0.5 * mean_C_group
            R = C_vals - self.lambda_cal * (target - q_vals) ** 2
        elif self.reward_variant == "instance":
            R = C_vals - self.instance_weight * (C_vals - q_vals) ** 2
        elif self.reward_variant == "rlcr":
            R = C_vals - self.lambda_cal * (q_vals - C_vals) ** 2
        elif self.reward_variant in ("rlcr_hybrid_loo", "rlcr_hybrid_loo_retention"):
            # Placeholder: final reward requires cross-member LOO group mean.
            # The caller is expected to pass the raw data (returned in meta)
            # to `finalize_loo_fitness` and overwrite per-member fitnesses.
            # Here we return mean(C) as a sensible fallback fitness so the
            # training loop still has a valid scalar if finalize is skipped.
            R = C_vals.copy()
        else:
            raise RuntimeError(f"unreachable: {self.reward_variant}")

        # RLCR format bonus: +0.1 on fully-parseable rollouts, -1.0 otherwise
        # (truncated rollouts count as malformed if they reach this code path,
        # but we zero them out below — so they contribute 0, not -1.0).
        format_bonus = np.zeros(n, dtype=np.float64)
        if self.reward_variant == "rlcr" and self.format_reward_enabled:
            fully_parsed = parsed_ok & boxed_ok & (~trunc_mask)
            format_bonus = np.where(fully_parsed, 0.1, -1.0)
            # truncated → zero (truncation path already zeroes R below).
            format_bonus[trunc_mask] = 0.0
            R = R + format_bonus

        # Zero-out truncated rollouts (they had C=q=0 — reward should be 0, not tiny-negative noise).
        R[trunc_mask] = 0.0

        fitness = float(np.max(R) if pass_at_k else np.mean(R))
        valid_mask = ~trunc_mask
        any_valid = bool(valid_mask.any())
        # Parse-rate / avg-confidence over the parseable subset so
        # `avg_verbal_confidence` reflects the distribution the model
        # actually produces (not a 0-fill on malformed outputs).
        parseable_q = q_vals[parsed_ok & valid_mask]

        # Confidence histogram entropy over 10 bins in [0,1] (over parseable q).
        if parseable_q.size > 0:
            hist, _ = np.histogram(parseable_q, bins=10, range=(0.0, 1.0))
            total = hist.sum()
            if total > 0:
                p = hist.astype(np.float64) / float(total)
                nz = p[p > 0]
                conf_entropy = float(-(nz * np.log(nz)).sum())
            else:
                conf_entropy = 0.0
            conf_std = float(parseable_q.std())
            mean_verbal_conf = float(parseable_q.mean())
        else:
            conf_entropy = 0.0
            conf_std = 0.0
            mean_verbal_conf = 0.0

        # Wrong-answer high-confidence diagnostic. Key DCPO/RLCR failure mode:
        # the model stays high-conf even when wrong. Over non-truncated rollouts,
        # report (a) mean q on wrong rollouts, (b) fraction of wrong rollouts
        # with q >= 0.8, and the mirror diagnostics for correct rollouts.
        wrong_mask = valid_mask & (C_vals < 0.5)
        right_mask = valid_mask & (C_vals >= 0.5)
        if wrong_mask.any():
            q_wrong = q_vals[wrong_mask]
            wrong_conf_mean = float(q_wrong.mean())
            wrong_frac_q_ge_0p8 = float((q_wrong >= 0.8).mean())
        else:
            wrong_conf_mean = 0.0
            wrong_frac_q_ge_0p8 = 0.0
        if right_mask.any():
            q_right = q_vals[right_mask]
            correct_conf_mean = float(q_right.mean())
            correct_frac_q_ge_0p8 = float((q_right >= 0.8).mean())
        else:
            correct_conf_mean = 0.0
            correct_frac_q_ge_0p8 = 0.0

        # 10-bin confidence histogram (over parseable q). Stored as per-bin
        # rates so WandB line-plots show the distribution migration over steps.
        hist_rates = np.zeros(10, dtype=np.float64)
        if parseable_q.size > 0:
            hist, _ = np.histogram(parseable_q, bins=10, range=(0.0, 1.0))
            total = float(hist.sum())
            if total > 0:
                hist_rates = hist.astype(np.float64) / total

        meta = {
            "correct_rate": float(C_vals[valid_mask].mean()) if any_valid else 0.0,
            "mean_q": float(q_vals[valid_mask].mean()) if any_valid else 0.0,
            "truncation_rate": float(trunc_mask.mean()),
            "parse_success_rate": float(parsed_ok[valid_mask].mean()) if any_valid else 0.0,
            "mean_brier_component": float(((q_vals[valid_mask] - C_vals[valid_mask]) ** 2).mean())
                if any_valid else 0.0,
            # Per-step confidence logging hooks (expert rec for wandb):
            "avg_verbal_confidence": float(parseable_q.mean()) if parseable_q.size else 0.0,
            "confidence_parse_rate": float(parsed_ok[valid_mask].mean()) if any_valid else 0.0,
            "boxed_parse_rate": float(boxed_ok[valid_mask].mean()) if any_valid else 0.0,
            # New aggregates for <conf>-tags / rlcr_hybrid_loo runs:
            "mean_verbal_conf": mean_verbal_conf,
            "confidence_std": conf_std,
            "confidence_entropy": conf_entropy,
            # Wrong-answer-high-conf diagnostics (paper_rlcr primary failure mode):
            "wrong_conf_mean": wrong_conf_mean,
            "wrong_frac_q_ge_0p8": wrong_frac_q_ge_0p8,
            "correct_conf_mean": correct_conf_mean,
            "correct_frac_q_ge_0p8": correct_frac_q_ge_0p8,
            # 10-bin confidence histogram rates (bin i covers [i/10, (i+1)/10]).
            "conf_hist_bin0_rate": float(hist_rates[0]),
            "conf_hist_bin1_rate": float(hist_rates[1]),
            "conf_hist_bin2_rate": float(hist_rates[2]),
            "conf_hist_bin3_rate": float(hist_rates[3]),
            "conf_hist_bin4_rate": float(hist_rates[4]),
            "conf_hist_bin5_rate": float(hist_rates[5]),
            "conf_hist_bin6_rate": float(hist_rates[6]),
            "conf_hist_bin7_rate": float(hist_rates[7]),
            "conf_hist_bin8_rate": float(hist_rates[8]),
            "conf_hist_bin9_rate": float(hist_rates[9]),
        }
        if self.reward_variant == "rlcr" and self.format_reward_enabled:
            meta["format_bonus_mean"] = float(format_bonus[valid_mask].mean()) if any_valid else 0.0

        # For rlcr_hybrid_loo, expose the raw per-rollout data so the caller
        # can compute the cross-population LOO group mean and call
        # `finalize_loo_fitness`. Keys are namespaced under `_loo_raw` to
        # avoid interfering with the scalar-averaging pattern in the caller
        # (which does `np.mean(v)` over task_info values).
        if self.reward_variant in ("rlcr_hybrid_loo", "rlcr_hybrid_loo_retention"):
            meta["_loo_raw"] = {
                "C": C_vals.copy(),
                "q": q_vals.copy(),
                "parsed_ok": parsed_ok.copy(),
                "boxed_ok": boxed_ok.copy(),
                "trunc": trunc_mask.copy(),
            }
        return fitness, tuple(model_answers), R, meta

    # ----- cross-population LOO aggregation (rlcr_hybrid_loo) ---------

    def finalize_loo_fitness(self, per_member_prompt_raw, prompt_is_anchor=None):
        """Compute LOO cross-population fitnesses for rlcr_hybrid_loo[_retention].

        Args:
            per_member_prompt_raw: nested list indexed as [i][j], where i is
                the population-member index and j is the prompt index. Each
                entry is the `_loo_raw` dict returned by `get_fitness`
                (with keys "C", "q", "parsed_ok", "boxed_ok", "trunc", each
                a 1-D np.ndarray of length samples_per_prompt).
            prompt_is_anchor: optional list[bool] of length num_prompts. When
                the variant is `rlcr_hybrid_loo_retention`, anchor prompts
                incur two extra penalties:
                  - lambda_retention · (1 - C)      [penalise regression on
                    prompts the base model already solves]
                  - lambda_wrong_conf_anchor · (1-C) · q^2   [extra penalty for
                    high confidence when wrong on an anchor]
                A per-sample truncation penalty `lambda_trunc` is always
                applied to truncated rollouts (for both LOO variants) rather
                than zeroing the reward, so truncation is an actively bad
                event rather than a neutral one.
                When `prompt_is_anchor is None` (or all-False), the extra
                anchor penalties vanish and this reduces to the plain LOO
                formula (with the truncation penalty still in effect for the
                _retention variant only).

        Returns:
            (fitnesses, info) where
              fitnesses is a nested list [i][j] of float scalar fitnesses
                  u_ij = mean_k r_ijk
              info is a dict of aggregate metrics for logging.
        """
        pop_size = len(per_member_prompt_raw)
        if pop_size == 0:
            return [], {}
        num_prompts = len(per_member_prompt_raw[0])

        # Normalise the anchor flag list.
        if prompt_is_anchor is None:
            anchor_flags = [False] * num_prompts
        else:
            anchor_flags = [bool(x) for x in prompt_is_anchor]
            if len(anchor_flags) != num_prompts:
                raise ValueError(
                    f"prompt_is_anchor length {len(anchor_flags)} != num_prompts {num_prompts}"
                )

        use_retention = self.reward_variant == "rlcr_hybrid_loo_retention"

        fitnesses = [[0.0 for _ in range(num_prompts)] for _ in range(pop_size)]
        all_R = []
        all_fmt_bonus = []
        all_bad_penalty = []
        all_retention_penalty = []
        all_wrong_conf_anchor_penalty = []
        all_trunc_penalty = []
        # Anchor-only aggregates for logging (wrong_conf_excess_over_0p2).
        all_anchor_wrong_q = []
        # Anchor-weighted retention bookkeeping for `base_solved_retention_rate`.
        anchor_correct_count = 0
        anchor_total_count = 0

        for j in range(num_prompts):
            # Stack per-member (samples_per_prompt) arrays for this prompt.
            C_stack = [per_member_prompt_raw[i][j]["C"] for i in range(pop_size)]
            q_stack = [per_member_prompt_raw[i][j]["q"] for i in range(pop_size)]
            parsed_stack = [per_member_prompt_raw[i][j]["parsed_ok"] for i in range(pop_size)]
            boxed_stack = [per_member_prompt_raw[i][j]["boxed_ok"] for i in range(pop_size)]
            trunc_stack = [per_member_prompt_raw[i][j]["trunc"] for i in range(pop_size)]

            is_anchor_j = anchor_flags[j]

            # Pre-compute per-member sums of C over samples for LOO mean.
            C_sums = np.array([c.sum() for c in C_stack], dtype=np.float64)
            C_counts = np.array([c.size for c in C_stack], dtype=np.float64)
            total_sum = C_sums.sum()
            total_count = C_counts.sum()

            for i in range(pop_size):
                C_i = C_stack[i]
                q_i = q_stack[i]
                parsed_i = parsed_stack[i]
                boxed_i = boxed_stack[i]
                trunc_i = trunc_stack[i]

                # LOO mean of C over (i' != i, all k).
                other_sum = total_sum - C_sums[i]
                other_count = total_count - C_counts[i]
                if other_count > 0:
                    C_bar_loo = float(other_sum / other_count)
                else:
                    # Single-member population fallback: use self-mean.
                    C_bar_loo = float(C_i.mean()) if C_i.size else 0.0

                # Soft target T_ijk = rho*C + (1-rho)*C_bar_loo.
                # At rho=1.0 this collapses to T=C (A2: strict per-sample Brier).
                T = self.rho * C_i + (1.0 - self.rho) * C_bar_loo

                # Brier-style calibration penalty.
                cal = self.lambda_cal * (q_i - T) ** 2

                # Format / invalid mutually-exclusive indicators.
                valid_fmt = parsed_i & boxed_i & (~trunc_i)
                invalid = (~parsed_i) | (~boxed_i) | trunc_i
                # Ensure mutual exclusivity (invalid is the complement of valid_fmt here).
                fmt_bonus = self.gamma_fmt * valid_fmt.astype(np.float64) if self.format_reward_enabled else np.zeros_like(C_i)
                bad_penalty = self.gamma_bad * invalid.astype(np.float64) if self.format_reward_enabled else np.zeros_like(C_i)

                # --- A1/A2 extras (rlcr_hybrid_loo_retention only) ---
                retention_penalty = np.zeros_like(C_i)
                wrong_conf_anchor_penalty = np.zeros_like(C_i)
                trunc_penalty = np.zeros_like(C_i)

                if use_retention:
                    # Explicit per-sample truncation penalty: truncation is
                    # actively bad, not neutral (the non-retention LOO variant
                    # just zeros truncated rewards, which can be gamed).
                    trunc_penalty = self.lambda_trunc * trunc_i.astype(np.float64)

                    if is_anchor_j:
                        wrong_mask = (1.0 - C_i)  # 1 when wrong, 0 when correct
                        # Retention: punish each wrong sample on an anchor prompt.
                        retention_penalty = self.lambda_retention * wrong_mask
                        # Wrong-conf-anchor: extra sting for high q when wrong on
                        # an anchor (the exact pathology localised in the
                        # stage-1 forensics).
                        wrong_conf_anchor_penalty = (
                            self.lambda_wrong_conf_anchor * wrong_mask * (q_i ** 2)
                        )
                        # Log anchor-wrong q for the wrong_conf_excess_over_0p2
                        # aggregate (only valid — non-truncated — samples).
                        valid_i = (~trunc_i)
                        wrong_valid = valid_i & (C_i < 0.5)
                        if wrong_valid.any():
                            all_anchor_wrong_q.append(q_i[wrong_valid].copy())

                        # Retention bookkeeping for `base_solved_retention_rate`:
                        # on anchor prompts, track what fraction of (population
                        # member × sample) rollouts are still correct.
                        valid_anchor = valid_i
                        anchor_correct_count += int(C_i[valid_anchor].sum())
                        anchor_total_count += int(valid_anchor.sum())

                r_ijk = (
                    C_i
                    - cal
                    + fmt_bonus
                    - bad_penalty
                    - retention_penalty
                    - wrong_conf_anchor_penalty
                    - trunc_penalty
                )

                # Per-member per-prompt fitness u_ij = mean_k r_ijk.
                u_ij = float(r_ijk.mean()) if r_ijk.size else 0.0
                fitnesses[i][j] = u_ij
                all_R.append(r_ijk)
                all_fmt_bonus.append(fmt_bonus)
                all_bad_penalty.append(bad_penalty)
                all_retention_penalty.append(retention_penalty)
                all_wrong_conf_anchor_penalty.append(wrong_conf_anchor_penalty)
                all_trunc_penalty.append(trunc_penalty)

        if all_R:
            R_all = np.concatenate(all_R)
            fmt_all = np.concatenate(all_fmt_bonus)
            bad_all = np.concatenate(all_bad_penalty)
            ret_all = np.concatenate(all_retention_penalty)
            wca_all = np.concatenate(all_wrong_conf_anchor_penalty)
            trunc_pen_all = np.concatenate(all_trunc_penalty)
            info = {
                "loo_reward_mean": float(R_all.mean()),
                "loo_reward_std": float(R_all.std()),
                "loo_format_bonus_mean": float(fmt_all.mean()),
                "loo_invalid_penalty_mean": float(bad_all.mean()),
            }
            if use_retention:
                info["loo_retention_penalty_mean"] = float(ret_all.mean())
                info["loo_wrong_conf_anchor_penalty_mean"] = float(wca_all.mean())
                info["loo_trunc_penalty_mean"] = float(trunc_pen_all.mean())
                info["anchor_fraction_in_batch"] = (
                    float(sum(anchor_flags) / max(len(anchor_flags), 1))
                )
                if all_anchor_wrong_q:
                    anchor_wrong_q = np.concatenate(all_anchor_wrong_q)
                    info["anchor_wrong_conf_mean"] = float(anchor_wrong_q.mean())
                    # The key triage metric: mean of max(q - 0.2, 0) | wrong & anchor.
                    info["anchor_wrong_conf_excess_over_0p2"] = float(
                        np.maximum(anchor_wrong_q - 0.2, 0.0).mean()
                    )
                    info["anchor_wrong_frac_q_ge_0p8"] = float(
                        (anchor_wrong_q >= 0.8).mean()
                    )
                else:
                    info["anchor_wrong_conf_mean"] = 0.0
                    info["anchor_wrong_conf_excess_over_0p2"] = 0.0
                    info["anchor_wrong_frac_q_ge_0p8"] = 0.0
                if anchor_total_count > 0:
                    info["base_solved_retention_rate"] = float(
                        anchor_correct_count / anchor_total_count
                    )
                else:
                    info["base_solved_retention_rate"] = 0.0
        else:
            info = {}
        return fitnesses, info


class CountdownTask:
    def __init__(self, batch_size, seed, datset_size=None, end_token: Optional[str] = None):
        data_path = "countdown.json"
        self.dataset = load_dataset("json", data_files=data_path, split="train")
        self.dataset = self.dataset.shuffle(seed=seed)
        print(f"{self.dataset=}")
        if datset_size is not None:
            self.dataset = self.dataset.select(range(datset_size))
        assert batch_size <= len(self.dataset), f"{batch_size=} must be <= {len(self.dataset)=}"
        self.batch_size = batch_size
        self.end_token = end_token
        self.idx = 0

    def get_batch(self):
        """Returns a list of prompt and answer strings of length batch_size."""
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        examples = [self.dataset[i] for i in indices]
        self.idx += self.batch_size
        batch_prompts = [example["context"] for example in examples]
        batch_answers = [(example["numbers"], example["target"]) for example in examples]
        return batch_prompts, batch_answers

    @staticmethod
    def _format_reward_function(response: str, end_token: Optional[str] = None) -> float:
        """
        Checks if the response follows the format <think>...</think><answer>...</answer>
        """
        # Strip end token if present
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_regex = r"<think>.*?<\/think>"
        answer_regex = r"<answer>.*?<\/answer>"
        full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

        think_match = re.search(think_regex, response, re.DOTALL)
        answer_match = re.search(answer_regex, response, re.DOTALL)
        full_format_match = re.match(full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0
        reward = 0.0
        if think_match:
            reward += 0.1
        if answer_match:
            reward += 0.5
        return reward

    @staticmethod
    def _answer_reward_function(response: str, numbers: List[int] = None, target: int = None) -> float:
        """
        Checks if the last <answer>...</answer> uses all numbers exactly once and evaluates to the target.
        Returns 1.0 if the last one is correct, else 0.0.
        """
        answer_regex = r"<answer>(.*?)<\/answer>"
        all_matches = re.findall(answer_regex, response, re.DOTALL)

        if not all_matches:
            return 0.0, None

        # Only check the last answer
        answer_content = all_matches[-1]
        
        allowed_chars = r"^[0-9+\-*/() ]+$"

        if not answer_content:
            return 0.0, answer_content
        if not re.match(allowed_chars, answer_content):
            return 0.0, answer_content

        # Check numbers used
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0, answer_content

        # Try evaluating
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0, answer_content
        except:
            return 0.0, answer_content

        return 0.0, answer_content
    
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        numbers, target = answer
        format_reward = self._format_reward_function("<think>" + generation, self.end_token)
        answer_reward, model_answer = self._answer_reward_function(generation, numbers, target)
        reward = format_reward * 0.1 + answer_reward
        return reward, model_answer
    

