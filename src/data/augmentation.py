"""
Data augmentation for bias measurement - specifically gender swapping.
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SwapPair:
    """A pair of original and swapped text."""
    original: str
    swapped: str
    swaps_made: List[Tuple[str, str]]
    swap_type: str  # "male_to_female" or "female_to_male"


class GenderSwapper:
    """
    Swap gendered words in text for bias measurement.

    Implements the gender flipping procedure described in the project:
    - He/She -> She/He
    - Male/Female names swap
    - Pronoun and possessive swaps
    """

    # Default gender swap mappings
    DEFAULT_PRONOUN_MAP = {
        # Subject pronouns
        "he": "she",
        "she": "he",
        # Object pronouns
        "him": "her",
        "her": "him",  # Note: "her" can be object or possessive
        # Possessive determiners
        "his": "her",
        # Possessive pronouns - note her->his already covered
        # Reflexive pronouns
        "himself": "herself",
        "herself": "himself",
        # Common nouns
        "man": "woman",
        "woman": "man",
        "men": "women",
        "women": "men",
        "male": "female",
        "female": "male",
        "boy": "girl",
        "girl": "boy",
        "boys": "girls",
        "girls": "boys",
        "sir": "ma'am",
        "ma'am": "sir",
        "mr.": "ms.",
        "ms.": "mr.",
        "mr": "ms",
        "ms": "mr",
        "mrs.": "mr.",
        "mrs": "mr",
        # Professions with gender connotations
        "father": "mother",
        "mother": "father",
        "fathers": "mothers",
        "mothers": "fathers",
        "dad": "mom",
        "mom": "dad",
        "son": "daughter",
        "daughter": "son",
        "sons": "daughters",
        "daughters": "sons",
        "brother": "sister",
        "sister": "brother",
        "brothers": "sisters",
        "sisters": "brothers",
        "uncle": "aunt",
        "aunt": "uncle",
        "uncles": "aunts",
        "aunts": "uncles",
        "grandfather": "grandmother",
        "grandmother": "grandfather",
        "husband": "wife",
        "wife": "husband",
        "husbands": "wives",
        "wives": "husbands",
        "king": "queen",
        "queen": "king",
        "prince": "princess",
        "princess": "prince",
        "gentleman": "lady",
        "lady": "gentleman",
        "gentlemen": "ladies",
        "ladies": "gentlemen",
    }

    # Common male/female first names for name swapping
    DEFAULT_MALE_NAMES = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
        "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
        "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
        "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan", "Jacob",
        "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott",
        "Brandon", "Benjamin", "Samuel", "Frank", "Gregory", "Raymond", "Alexander",
        "Patrick", "Jack", "Henry", "Oliver", "Arthur", "Peter", "Roger", "Adam"
    ]

    DEFAULT_FEMALE_NAMES = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
        "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra",
        "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol",
        "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura", "Sharon",
        "Cynthia", "Kathleen", "Amy", "Shirley", "Angela", "Helen", "Anna", "Brenda",
        "Pamela", "Nicole", "Samantha", "Katherine", "Emma", "Ruth", "Christine",
        "Catherine", "Debra", "Rachel", "Carolyn", "Janet", "Virginia", "Maria"
    ]

    def __init__(
        self,
        pronoun_map: Optional[Dict[str, str]] = None,
        male_names: Optional[List[str]] = None,
        female_names: Optional[List[str]] = None,
        names_file: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the gender swapper.

        Args:
            pronoun_map: Custom pronoun mappings (overrides defaults)
            male_names: List of male names to swap (overrides defaults)
            female_names: List of female names to swap (overrides defaults)
            names_file: JSON file with male/female name lists
            random_seed: Random seed for name selection
        """
        self.pronoun_map = pronoun_map or self.DEFAULT_PRONOUN_MAP.copy()
        self.reverse_pronoun_map = {v: k for k, v in self.pronoun_map.items()}

        random.seed(random_seed)

        # Load or create name mappings
        if names_file and Path(names_file).exists():
            with open(names_file, 'r') as f:
                name_data = json.load(f)
                self.male_names = name_data.get("male", self.DEFAULT_MALE_NAMES)
                self.female_names = name_data.get("female", self.DEFAULT_FEMALE_NAMES)
        else:
            self.male_names = male_names or self.DEFAULT_MALE_NAMES
            self.female_names = female_names or self.DEFAULT_FEMALE_NAMES

        # Create bidirectional name mapping
        self.name_map = {}
        min_len = min(len(self.male_names), len(self.female_names))
        for i in range(min_len):
            self.name_map[self.male_names[i]] = self.female_names[i]
            self.name_map[self.female_names[i]] = self.male_names[i]

        # Compile regex patterns for word boundaries
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Create case-insensitive patterns for pronouns
        self.pronoun_patterns = {}
        for word, replacement in self.pronoun_map.items():
            # Match word boundaries, case-insensitive
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            self.pronoun_patterns[word] = (pattern, replacement)

        # Name patterns (case-sensitive for names)
        self.name_patterns = {}
        for name, replacement in self.name_map.items():
            pattern = re.compile(r'\b' + re.escape(name) + r'\b')
            self.name_patterns[name] = (pattern, replacement)

    def swap_gender(self, text: str, swap_names: bool = True) -> SwapPair:
        """
        Swap gendered words in text.

        Args:
            text: Input text
            swap_names: Whether to swap names as well

        Returns:
            SwapPair with original, swapped text, and list of changes
        """
        if not text:
            return SwapPair(original=text, swapped=text, swaps_made=[], swap_type="none")

        swaps_made = []
        swapped = text

        # Track swap direction
        male_to_female = 0
        female_to_male = 0

        # Swap pronouns and common gendered words
        for word, (pattern, replacement) in self.pronoun_patterns.items():
            matches = pattern.findall(swapped)
            if matches:
                swapped = pattern.sub(replacement, swapped)
                for match in matches:
                    swaps_made.append((match, replacement))
                    # Track direction
                    if word.lower() in ["he", "him", "his", "himself", "man", "men", "male", "boy", "father", "dad", "son", "brother", "uncle", "grandfather", "husband", "king", "prince", "gentleman"]:
                        male_to_female += 1
                    else:
                        female_to_male += 1

        # Swap names
        if swap_names:
            for name, (pattern, replacement) in self.name_patterns.items():
                matches = pattern.findall(swapped)
                if matches:
                    swapped = pattern.sub(replacement, swapped)
                    for match in matches:
                        swaps_made.append((match, replacement))
                        # Track direction based on name lists
                        if match in self.male_names:
                            male_to_female += 1
                        elif match in self.female_names:
                            female_to_male += 1

        # Determine swap type
        if male_to_female > female_to_male:
            swap_type = "male_to_female"
        elif female_to_male > male_to_female:
            swap_type = "female_to_male"
        else:
            swap_type = "balanced"

        return SwapPair(
            original=text,
            swapped=swapped,
            swaps_made=swaps_made,
            swap_type=swap_type
        )

    def create_bias_test_pair(self, text: str) -> Tuple[str, str]:
        """
        Create a male/female variant pair from text.

        If text contains male-coded words, returns (male_version, female_version).
        If text contains female-coded words, returns (female_version, male_version).
        """
        swap_result = self.swap_gender(text)

        if swap_result.swap_type == "male_to_female":
            return (text, swap_result.swapped)
        elif swap_result.swap_type == "female_to_male":
            return (swap_result.swapped, text)
        else:
            # If balanced, randomly decide which is "male" version
            if random.random() < 0.5:
                return (text, swap_result.swapped)
            else:
                return (swap_result.swapped, text)

    def batch_swap(self, texts: List[str]) -> List[SwapPair]:
        """Apply gender swapping to a batch of texts."""
        return [self.swap_gender(text) for text in texts]

    def subsample_for_bias(self, dataset: List[Dict], n_samples: int) -> List[Dict]:
        """
        Subsample dataset and create gender-swapped pairs for bias testing.

        Returns list of dicts with original and swapped versions.
        """
        if len(dataset) <= n_samples:
            sampled = dataset
        else:
            sampled = random.sample(dataset, n_samples)

        bias_pairs = []
        for item in sampled:
            text = item.get("text", "")
            if not text:
                continue

            # Create gender-swapped variant
            swap_pair = self.swap_gender(text)

            bias_pairs.append({
                "original": item,
                "male_variant": swap_pair.original if swap_pair.swap_type == "male_to_female" else swap_pair.swapped,
                "female_variant": swap_pair.swapped if swap_pair.swap_type == "male_to_female" else swap_pair.original,
                "swaps_made": swap_pair.swaps_made,
                "swap_type": swap_pair.swap_type,
            })

        return bias_pairs

    def has_gendered_content(self, text: str) -> bool:
        """Check if text contains gendered words that can be swapped."""
        swap_result = self.swap_gender(text)
        return len(swap_result.swaps_made) > 0

    def extract_gendered_samples(
        self,
        dataset: List[Dict],
        min_swaps: int = 1
    ) -> List[Dict]:
        """Filter dataset to only samples with gendered content."""
        gendered = []
        for item in dataset:
            text = item.get("text", "")
            swap_result = self.swap_gender(text)
            if len(swap_result.swaps_made) >= min_swaps:
                gendered.append(item)
        return gendered


def create_gender_augmented_dataset(
    dataset: List[Dict],
    swapper: Optional[GenderSwapper] = None,
    n_augment: Optional[int] = None
) -> List[Dict]:
    """
    Augment dataset with gender-swapped variants.

    Each sample with gendered content is replaced by two samples:
    - Original with male-coded language
    - Swapped with female-coded language
    """
    if swapper is None:
        swapper = GenderSwapper()

    augmented = []

    for item in dataset:
        text = item.get("text", "")
        swap_pair = swapper.swap_gender(text)

        if len(swap_pair.swaps_made) > 0:
            # Add male version
            male_item = item.copy()
            male_item["text"] = swap_pair.original if swap_pair.swap_type == "male_to_female" else swap_pair.swapped
            male_item["gender_coding"] = "male"
            augmented.append(male_item)

            # Add female version
            female_item = item.copy()
            female_item["text"] = swap_pair.swapped if swap_pair.swap_type == "male_to_female" else swap_pair.original
            female_item["gender_coding"] = "female"
            augmented.append(female_item)
        else:
            # Keep original if no gendered content
            augmented.append(item)

        if n_augment and len(augmented) >= n_augment * 2:
            break

    return augmented
