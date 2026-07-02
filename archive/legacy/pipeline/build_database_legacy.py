import os
import sys
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# NOTE: Remember we're in pipeline/, adjust paths accordingly for future use

# Path Configuration
# We're in pipeline/, so data/ is one level up at project root
SOURCE_DATA_DIR = "../data/perva-data"    # Source directory containing images
DEVICE = "cuda"                            # Device for model inference (cuda/cpu)
MODEL_PATH = "openbmb/MiniCPM-o-2_6"      # MiniCPM model path

# Dataset Split Configuration (following R2P workflow)
DATASET_SPLIT = "train"    # Options: "train", "test", "all"
IMAGES_PER_CONCEPT = 1      # Options: 1, 3, 5, "all"
                            # How many images to use per concept for fingerprinting
                            # R2P original uses 1 image per concept

# Run Settings
DEBUG_MODE = True         # Set to False to process the entire dataset
DEBUG_LIMIT = 5            # Number of concepts to process in debug mode
USE_CLIP_CATEGORY = True   # True = Let R2P detect category via CLIP; False = Pass None/Generic
USE_CLIP_SELECTION = True  # True = Select images closest to CLIP centroid (R2P); False = Use first N images
IGNORE_LAION = True        # Ignore 'laion' subdirectories (R2P training data)
SEED = 42                  # Random seed for reproducible CLIP selection (R2P uses seed for K-means)

# SDXL Prompt Generation Strategy
# Options: 'simple', 'optimized', 'gemini'
SDXL_PROMPT_STRATEGY = 'gemini'  # 'simple' = Natural description + style suffix (baseline)
                                  # 'optimized' = Hierarchical tags with weights (R2P enhanced)
                                  # 'gemini' = Brand-first, ultra-concise (SOTA personalization)


# ============================================================================
# 2. SETUP & IMPORTS
# ============================================================================

# Path Configuration: We're in pipeline/, r2p_core is at the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from pipeline/
core_dir = os.path.join(project_root, "r2p_core")
utils_dir = os.path.join(core_dir, "utils")
models_dir = os.path.join(core_dir, "models")
database_dir = os.path.join(project_root, "database")
eval_dir = os.path.join(core_dir, "evaluators")

for path in [project_root, core_dir, utils_dir, models_dir, database_dir, eval_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Import SDXL prompt templates from prompts module
from pipeline.prompts import (
    SYSTEM_PROMPT_SIMPLE,
    SYSTEM_PROMPT_GEMINI,
    SYSTEM_PROMPT_OPTIMIZED,
    HARDCODED_STYLE
)

try:
    from database.mini_cpm_info import MiniCPMDescription
    from database.create_train_test_perva_split import CLIPImageProcessor
    
    class MockArgs:
        """Mock arguments for R2P model interface."""
        def __init__(self):
            self.user_defined = False
            self.template_based = True
            
except ImportError as e:
    print(f"❌ Error importing R2P modules: {e}")
    print("Ensure 'r2p_core' contains the 'src' files from the original repo.")
    sys.exit(1)


# ============================================================================
# 3. DATABASE BUILDER CLASS
# ============================================================================

class DatabaseBuilder:
    """
    Builds a database of image fingerprints and SDXL prompts following R2P workflow.
    
    This class processes images at the CONCEPT level (not individual images):
    - Each concept = a unique physical object with multiple views
    - Extracts fingerprints from ONE representative image per concept
    - Stores ALL selected images for each concept in the database
    
    R2P Dataset Structure:
        data/perva-data/
        ├── train/
        │   ├── bag/
        │   │   ├── alx/  ← concept (unique object)
        │   │   │   ├── 1.jpg, 2.jpg, ..., 8.jpg  ← multi-view
        │   │   │   └── laion/ (ignored - training data)
        │   │   ├── ash/
        │   │   └── ...
        │   └── ...
        └── test/ (same structure)
    
    Attributes:
        source_dir (str): Base directory containing train/test splits
        output_path (str): Where to save the JSON database
        device (str): Compute device ('cuda' or 'cpu')
        model_path (str): MiniCPM model identifier
        dataset_split (str): Which split to process ('train', 'test', 'all')
        images_per_concept (int|str): How many images to use (1, 3, 5, 'all')
        debug_mode (bool): If True, process only debug_limit concepts
        debug_limit (int): Number of concepts to process in debug mode
        use_clip_category (bool): If True, auto-detect category with CLIP
        ignore_laion (bool): If True, skip 'laion' subdirectories
    """
    
    # Valid image file extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Mapping: R2P fingerprint fields -> Human-readable labels for LLM
    FIELD_MAP = [
        ('category', 'Category'),
        ('general', 'Base Description'),
        ('shape', 'Shape'),
        ('material', 'Material'),
        ('color', 'Color'),
        ('pattern', 'Pattern'),
        ('brand/text', 'Visible Text'),
        ('distinct features', 'Distinctive Flaws/Details')
    ]
    
    # Phrases indicating invalid/missing data that should be filtered out
    NEGATIVE_TRIGGERS = [
        "no visible", "none", "n/a", "not readable", 
        "unknown", "no brand", "no text"
    ]
    
    def __init__(self, source_dir, output_path, device="cuda", 
                 model_path="openbmb/MiniCPM-o-2_6",
                 dataset_split="train", images_per_concept=1,
                 debug_mode=False, debug_limit=5, use_clip_category=True,
                 use_clip_selection=True, ignore_laion=True, seed=42,
                 prompt_strategy='gemini', system_prompt=None, hardcoded_style=None):
        """
        Initialize the DatabaseBuilder following R2P workflow.
        
        Args:
            source_dir: Directory containing images to process
            output_path: Where to save the resulting JSON database
            device: Device for model inference ('cuda' or 'cpu')
            model_path: Path or identifier for the MiniCPM model
            dataset_split: Which split to process ('train', 'test', 'all')
            images_per_concept: Number of images per concept (1, 3, 5, 'all')
            debug_mode: If True, process only a subset of concepts
            debug_limit: Number of concepts to process in debug mode
            use_clip_category: If True, detect category via CLIP
            use_clip_selection: If True, select images closest to CLIP centroid
            ignore_laion: If True, ignore 'laion' subdirectories
            seed: Random seed for reproducible image selection (R2P compatibility)
            prompt_strategy: Which prompt template to use ('simple'/'optimized'/'gemini')
                            'simple' = Natural description + hardcoded style (baseline)
                            'optimized' = Hierarchical tags with weights (R2P enhanced)
                            'gemini' = Brand-first ultra-concise (SOTA personalization)
            system_prompt: Custom prompt for LLM (uses global default based on strategy)
            hardcoded_style: Custom style suffix (uses global default if None)
        """
        self.source_dir = source_dir
        self.output_path = output_path
        self.device = device
        self.model_path = model_path
        self.dataset_split = dataset_split
        self.images_per_concept = images_per_concept
        self.debug_mode = debug_mode
        self.debug_limit = debug_limit
        self.use_clip_category = use_clip_category
        self.use_clip_selection = use_clip_selection
        self.ignore_laion = ignore_laion
        self.seed = seed
        self.prompt_strategy = prompt_strategy.lower()
        
        # Select appropriate prompt template if not provided
        if system_prompt is None:
            if self.prompt_strategy == 'gemini':
                self.system_prompt = SYSTEM_PROMPT_GEMINI
            elif self.prompt_strategy == 'optimized':
                self.system_prompt = SYSTEM_PROMPT_OPTIMIZED
            else:  # 'simple'
                self.system_prompt = SYSTEM_PROMPT_SIMPLE
        else:
            self.system_prompt = system_prompt
            
        self.hardcoded_style = hardcoded_style or HARDCODED_STYLE
        
        # Will be initialized during build()
        self.extractor = None
        self.mock_args = None
        self.clip_processor = None
        self.database_data = {
            "concept_dict": {},
            "path_to_concept": {}
        }
    
    def _get_concepts(self):
        """
        Discover all concepts in the dataset following R2P structure.
        
        Scans the dataset and identifies:
        - Categories (bag, pillow, plant, etc.)
        - Concepts within categories (alx, ash, bkq, etc.)
        - Images for each concept
        
        Returns:
            list: List of dicts, each with:
                - 'category': Category name
                - 'concept_id': Unique concept identifier
                - 'concept_path': Full path to concept directory
                - 'images': Sorted list of image paths
                - 'split': Which split this concept belongs to
        """
        abs_path = os.path.abspath(self.source_dir)
        print(f"   🔍 Looking in: {abs_path}")
        print(f"   📊 Split: {self.dataset_split}")
        
        # Automatic path correction
        target_dir = self.source_dir
        if not os.path.exists(self.source_dir):
            print(f"   ⚠️ Path '{self.source_dir}' not found.")
            if "_" in self.source_dir and os.path.exists(self.source_dir.replace("_", "-")):
                target_dir = self.source_dir.replace("_", "-")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            elif "-" in self.source_dir and os.path.exists(self.source_dir.replace("-", "_")):
                target_dir = self.source_dir.replace("-", "_")
                print(f"   ✅ Found '{target_dir}' instead. Switching.")
            else:
                return []
        
        concepts = []
        
        # Determine which splits to process
        splits_to_process = []
        if self.dataset_split == "all":
            splits_to_process = ["train", "test"]
        else:
            splits_to_process = [self.dataset_split]
        
        # Scan for concepts
        for split in splits_to_process:
            split_dir = os.path.join(target_dir, split)
            if not os.path.exists(split_dir):
                print(f"   ⚠️ Split directory '{split}' not found, skipping...")
                continue
            
            # Iterate over categories (bag, pillow, etc.)
            for category in sorted(os.listdir(split_dir)):
                category_path = os.path.join(split_dir, category)
                if not os.path.isdir(category_path):
                    continue
                
                # Iterate over concepts (alx, ash, etc.)
                for concept_id in sorted(os.listdir(category_path)):
                    concept_path = os.path.join(category_path, concept_id)
                    if not os.path.isdir(concept_path):
                        continue
                    
                    # Skip 'laion' directories if configured
                    if self.ignore_laion and concept_id.lower() == "laion":
                        continue
                    
                    # Find all images in this concept (skip subdirectories)
                    images = []
                    for item in sorted(os.listdir(concept_path)):
                        item_path = os.path.join(concept_path, item)
                        
                        # Skip subdirectories (like laion/)
                        if os.path.isdir(item_path):
                            continue
                        
                        ext = os.path.splitext(item)[1]
                        if ext in self.VALID_EXTENSIONS:
                            images.append(item_path)
                    
                    if images:
                        concepts.append({
                            'category': category,
                            'concept_id': concept_id,
                            'concept_path': concept_path,
                            'images': images,  # Already sorted
                            'split': split
                        })
        
        return concepts
    
    def _select_images_for_concept(self, images):
        """
        Select N images from a concept's image list.
        
        Two selection strategies:
        1. CLIP-based (R2P enhanced): 
           - Select top-N images closest to CLIP feature centroid
           - Shuffle them with seed for robust random selection
           - This avoids bias from arbitrary ordering
        2. Simple: Use first N images (sorted numerically)
        
        Args:
            images: List of image paths for a concept
            
        Returns:
            list: Selected image paths (shuffled if CLIP-based)
        """
        if self.images_per_concept == "all":
            return images
        
        num_to_select = min(self.images_per_concept, len(images))
        
        # Strategy 1: CLIP-based selection with robust random sampling
        if self.use_clip_selection and self.clip_processor is not None:
            try:
                # Extract CLIP features for all images
                clip_features = self.clip_processor.extract_clip_features(images)
                
                # Get images closest to centroid (mean feature)
                _, closest_indices = self.clip_processor.get_closest_to_mean_features(
                    clip_features, top_n=num_to_select
                )
                
                selected = [images[idx] for idx in closest_indices]
                
                # Shuffle with seed for robust selection
                # This ensures the "representative" image (first element) is 
                # randomly chosen among the top-N, avoiding view-angle bias
                np.random.seed(self.seed)
                np.random.shuffle(selected)
                
                return selected
                
            except Exception as e:
                print(f"⚠️ CLIP selection failed: {e}, falling back to simple selection")
        
        # Strategy 2: Simple selection (first N images)
        return images[:num_to_select]
    
    def _collect_features_safely(self, info_dict):
        """
        Aggregate fingerprint fields into a human-readable technical sheet.
        
        Filters out empty, invalid, or negative-indicator fields.
        
        Args:
            info_dict: Dictionary containing extracted fingerprints
            
        Returns:
            str: Formatted technical sheet (one field per line)
        """
        lines = []
        
        for key, label in self.FIELD_MAP:
            val = info_dict.get(key)
            
            if val and isinstance(val, str):
                clean_val = val.strip()
                val_lower = clean_val.lower()
                
                # Skip if too short
                if len(clean_val) < 2:
                    continue
                
                # Skip if contains negative indicators
                if any(trigger in val_lower for trigger in self.NEGATIVE_TRIGGERS):
                    continue
                
                lines.append(f"{label}: {clean_val}")
        
        return "\n".join(lines)
    
    def _generate_sdxl_prompt(self, full_info_dict):
        """
        Generate SDXL-compatible prompt from fingerprint data.
        
        Three strategies available (controlled by SDXL_PROMPT_STRATEGY):
        
        1. SIMPLE (baseline): Natural language description + hardcoded style suffix
           - More readable, traditional approach
           - Style applied as separate suffix
        
        2. OPTIMIZED (R2P enhanced): Hierarchical tags with weights
           - (Concept:1.3) emphasis for identity preservation
           - Discriminative traits placed first
           - Target 60-70 tokens
        
        3. GEMINI (SOTA personalization): Brand-first ultra-concise
           - (Brand logo:1.4) if present, maximum brand emphasis
           - Material+texture fused descriptors
           - Target 65-70 tokens, maximum identity preservation
        
        Args:
            full_info_dict: Dictionary with extracted fingerprints
            
        Returns:
            str: Complete SDXL prompt ready for image generation
        """
        features_text = self._collect_features_safely(full_info_dict)
        concept_name = full_info_dict.get('category', 'object')
        
        # Fallback if no valid features found
        if not features_text:
            if self.prompt_strategy == 'simple':
                return f"A photorealistic image of a {concept_name}, studio lighting."
            else:  # optimized or gemini
                return f"({concept_name}:1.3), photorealistic, studio lighting, 8k, sharp focus"
        
        # Build query with appropriate prompt template
        query = f"{self.system_prompt}\n{features_text}"
        
        try:
            msgs = [{"role": "user", "content": query}]
            res = self.extractor.model.chat(msgs=msgs, tokenizer=self.extractor.tokenizer)
            
            # Handle tuple/list response
            if isinstance(res, (tuple, list)):
                description = res[0]
            else:
                description = res
                
            # Clean output
            clean_desc = description.strip().strip('"').strip("'")
            
            # Apply post-processing based on strategy
            if self.prompt_strategy == 'simple':
                # SIMPLE: Append hardcoded style suffix
                final_prompt = f"{clean_desc}{self.hardcoded_style}"
            else:
                # OPTIMIZED or GEMINI: Validate emphasis weight presence
                if f"({concept_name}:" not in clean_desc and f"({concept_name.lower()}:" not in clean_desc:
                    # Add emphasis if LLM forgot
                    final_prompt = f"({concept_name}:1.3), {clean_desc}"
                else:
                    final_prompt = clean_desc
            
            return final_prompt
                       
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate SDXL prompt: {e}")
            # Fallback based on strategy
            if self.prompt_strategy == 'simple':
                return f"{features_text}{self.hardcoded_style}"
            else:
                return f"({concept_name}:1.3), {features_text}, photorealistic, 8k"
    
    def _process_concept(self, concept_data):
        """
        Process a concept following R2P workflow (enhanced):
        1. Select N most representative images (CLIP centroid + random shuffle with seed)
        2. Extract fingerprints from FIRST image (randomly chosen among top-N)
        3. Generate SDXL prompt from fingerprints
        4. Store ALL selected images in database
        
        Args:
            concept_data: Dict with 'category', 'concept_id', 'images', etc.
            
        Returns:
            tuple: (concept_key, entry_dict) for database
        """
        category = concept_data['category']
        concept_id = concept_data['concept_id']
        all_images = concept_data['images']
        
        # Select N images for this concept
        selected_images = self._select_images_for_concept(all_images)
        
        if not selected_images:
            raise ValueError(f"No images selected for concept {concept_id}")
        
        # Extract fingerprints from FIRST image (R2P approach)
        representative_image = selected_images[0]
        
        cat_arg = None if self.use_clip_category else category
        
        # Extract fingerprints using MiniCPM
        json_str = self.extractor.generate_caption(
            image_file=representative_image,
            cat=cat_arg,
            concept_identifier=concept_id, 
            args=self.mock_args
        )
        
        # Clean and parse JSON response
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        item_info = json.loads(json_str)
        
        # Generate SDXL prompt and add to info
        sdxl_prompt = self._generate_sdxl_prompt(item_info)
        item_info["sdxl_prompt"] = sdxl_prompt
        
        # Build concept key (R2P format)
        concept_key = f"<{concept_id}>"
        
        # Build database entry
        entry = {
            "name": concept_id,
            "image": selected_images,  # Store ALL selected images
            "info": item_info,
            "category": category
        }
        
        return concept_key, entry
    
    def _select_target_concepts(self, all_concepts):
        """
        Select which concepts to process based on debug mode.
        
        Args:
            all_concepts: Complete list of discovered concepts
            
        Returns:
            list: Filtered list of concepts to process
        """
        if not self.debug_mode:
            return all_concepts
        
        print(f"⚠️ DEBUG MODE ACTIVE: Selecting {self.debug_limit} concepts...")
        return all_concepts[:self.debug_limit]
    
    def _initialize_models(self):
        """Load the MiniCPM model and CLIP processor, initialize mock arguments."""
        print("\n[1/5] Loading Models...")
        print("   - MiniCPM for fingerprint extraction")
        self.extractor = MiniCPMDescription(model_path=self.model_path, device=self.device)
        self.mock_args = MockArgs()
        
        # Load CLIP for image selection if enabled
        if self.use_clip_selection:
            print("   - CLIP for image selection (closest to centroid)")
            self.clip_processor = CLIPImageProcessor(device=self.device)
        else:
            print("   - Using simple selection (first N images)")
            self.clip_processor = None
    
    def _save_database(self):
        """Save the database to JSON file."""
        print(f"\n[4/5] Saving database to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, "w") as f:
            json.dump(self.database_data, f, indent=4)
    
    def extract_single_image(self, image_path: str, category: str = None) -> dict:
        """
        Extract fingerprints from a single image.
        
        This is a convenience method for processing individual images
        without going through the full database pipeline.
        
        Args:
            image_path: Path to the image file
            category: Optional category hint (e.g., 'bag', 'shoe')
            
        Returns:
            dict: Fingerprints including sdxl_prompt, or None if failed
        """
        # Initialize extractor if not already done
        if self.extractor is None:
            print("   📦 Loading MiniCPM for extraction...")
            from r2p_core.database.mini_cpm_info import MiniCPMDescription
            self.extractor = MiniCPMDescription(model_path=self.model_path, device=self.device)
            self.mock_args = MockArgs()
        
        try:
            # Extract fingerprints
            cat_arg = None if self.use_clip_category else category
            concept_id = os.path.splitext(os.path.basename(image_path))[0]
            
            json_str = self.extractor.generate_caption(
                image_file=image_path,
                cat=cat_arg,
                concept_identifier=concept_id,
                args=self.mock_args
            )
            
            # Clean and parse JSON
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            fingerprints = json.loads(json_str)
            
            # Generate SDXL prompt
            sdxl_prompt = self._generate_sdxl_prompt(fingerprints)
            fingerprints["sdxl_prompt"] = sdxl_prompt
            
            return fingerprints
            
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            return None
    
    def build_database(self):
        """
        Execute the complete database building pipeline following R2P workflow.
        
        Workflow:
        1. Initialize models (MiniCPM)
        2. Discover concepts (not individual images)
        3. For each concept:
           - Select N images
           - Extract fingerprints from first image
           - Generate SDXL prompt
           - Store all selected images
        4. Save database to JSON
        
        Returns:
            dict: Statistics with keys:
                - success_count: Number of successfully processed concepts
                - total_concepts: Total number of concepts attempted
                - total_images: Total number of images stored
                - database_path: Absolute path to the saved database file
        """
        print(f"🚀 Starting Build Database (R2P Workflow)")
        print(f"📂 Source: {self.source_dir}")
        print(f"💾 Output: {self.output_path}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Split: {self.dataset_split}")
        print(f"🖼️  Images per concept: {self.images_per_concept}")
        print(f"🎯 CLIP Selection: {self.use_clip_selection}")
        print(f"🌱 Seed: {self.seed}")
        print(f"✨ SDXL Prompt: {self.prompt_strategy.upper()}")
        print(f"🧪 Debug Mode: {self.debug_mode}")
        
        # Step 1: Initialize models
        self._initialize_models()
        
        # Step 2: Discover concepts
        print("\n[2/5] Discovering concepts...")
        all_concepts = self._get_concepts()
        
        if not all_concepts:
            print(f"❌ No concepts found in {self.source_dir}")
            return {"success_count": 0, "total_concepts": 0, "total_images": 0}
        
        print(f"Found {len(all_concepts)} concepts across categories.")
        
        # Step 3: Select target concepts (debug mode filter)
        target_concepts = self._select_target_concepts(all_concepts)
        print(f"Processing {len(target_concepts)} concepts...")
        
        # Step 4: Process concepts
        print("\n[3/5] Extracting Fingerprints & Generating Prompts...")
        
        success_count = 0
        total_images_stored = 0
        
        for concept_data in tqdm(target_concepts, desc="Processing concepts"):
            try:
                concept_key, entry = self._process_concept(concept_data)
                
                # Add to database
                self.database_data["concept_dict"][concept_key] = entry
                
                # Map each image path to its concept
                for img_path in entry["image"]:
                    self.database_data["path_to_concept"][img_path] = concept_key
                
                success_count += 1
                total_images_stored += len(entry["image"])
                
            except Exception as e:
                concept_id = concept_data.get('concept_id', 'unknown')
                print(f"\n❌ Error processing concept '{concept_id}': {e}")
                continue
        
        # Step 5: Save database
        self._save_database()
        
        print(f"\n✅ Done! Processed {success_count}/{len(target_concepts)} concepts.")
        print(f"   Total images stored: {total_images_stored}")
        print(f"   Database saved at: {os.path.abspath(self.output_path)}")
        
        return {
            "success_count": success_count,
            "total_concepts": len(target_concepts),
            "total_images": total_images_stored,
            "database_path": os.path.abspath(self.output_path)
        }


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Script entry point.
    
    Creates a DatabaseBuilder instance with global configuration
    and executes the build process following R2P workflow.
    
    Output filename format: database_perva_{split}_{num_images}_{selection}.json
    Examples:
        - database_perva_train_1_clip.json      (CLIP-based selection)
        - database_perva_train_3_simple.json    (Simple first-N selection)
        - database_perva_test_all_clip.json     (All images, test split)
        - database_perva_all_5_simple.json      (Both splits, 5 images, simple)
    """
    # Calculate absolute paths based on script location
    # This works whether called from pipeline/ or root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_main = os.path.dirname(script_dir)  # Go up from pipeline/ to root
    
    # Change working directory to project root
    # This ensures all relative paths (e.g., example_database/) work correctly
    os.chdir(project_root_main)
    
    # Source data is always at project_root/data/perva-data
    source_data_absolute = os.path.join(project_root_main, "data", "perva-data")
    
    # Generate dynamic output filename based on configuration
    img_count_str = str(IMAGES_PER_CONCEPT) if isinstance(IMAGES_PER_CONCEPT, int) else IMAGES_PER_CONCEPT
    selection_method = "clip" if USE_CLIP_SELECTION else "simple"
    output_filename = f"database_perva_{DATASET_SPLIT}_{img_count_str}_{selection_method}.json"
    output_path = os.path.join(project_root_main, "database", output_filename)
    
    print(f"\n" + "="*70)
    print(f"DATABASE BUILDER - R2P Workflow")
    print(f"="*70)
    print(f"Configuration:")
    print(f"  - Dataset Split: {DATASET_SPLIT}")
    print(f"  - Images per Concept: {IMAGES_PER_CONCEPT}")
    print(f"  - CLIP Selection: {USE_CLIP_SELECTION}")
    print(f"  - Seed: {SEED}")
    strategy_names = {'simple': 'SIMPLE (baseline)', 'optimized': 'OPTIMIZED (R2P)', 'gemini': 'GEMINI (SOTA)'}
    print(f"  - SDXL Prompt: {strategy_names.get(SDXL_PROMPT_STRATEGY, SDXL_PROMPT_STRATEGY)}")
    print(f"  - Ignore Laion: {IGNORE_LAION}")
    print(f"  - Output: {output_path}")
    print(f"="*70 + "\n")
    
    builder = DatabaseBuilder(
        source_dir=source_data_absolute,
        output_path=output_path,
        device=DEVICE,
        model_path=MODEL_PATH,
        dataset_split=DATASET_SPLIT,
        images_per_concept=IMAGES_PER_CONCEPT,
        debug_mode=DEBUG_MODE,
        debug_limit=DEBUG_LIMIT,
        use_clip_category=USE_CLIP_CATEGORY,
        use_clip_selection=USE_CLIP_SELECTION,
        ignore_laion=IGNORE_LAION,
        seed=SEED,
        prompt_strategy=SDXL_PROMPT_STRATEGY
    )
    
    stats = builder.build_database()
    return stats


if __name__ == "__main__":
    main()
