import re
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import docx
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import json

# Add ML module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
from src.models.multitask_text_model import MultiTaskTextModel
from src.data.datasets import Vocab, encode_text

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ScriptAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the script analyzer.
        
        Args:
            model_path: Path to trained multitask model for genre prediction
        """
        self.stop_words = set(stopwords.words('english'))
        self.model_path = model_path
        self.genre_classifier = None
        self.abstractive_tokenizer = None
        self.abstractive_model = None
        
        # Load custom model if available
        if model_path and os.path.exists(model_path):
            self.genre_classifier = self._load_custom_model(model_path)

    def load_abstractive_model(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """Load a small local summarization model (downloads weights once)."""
        try:
            self.abstractive_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"Failed to load abstractive model: {e}")
            return False
    
    def _load_custom_model(self, model_path: str):
        """Load custom trained model for genre prediction"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(model_path, map_location=device)
            
            # Rebuild vocab
            vocab_list = ckpt["vocab"]
            vocab = Vocab()
            vocab.itos = list(vocab_list)
            vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
            
            # Get label classes
            label_classes = ckpt["label_classes"]
            cfg = ckpt.get("config", {})
            
            # Create model
            model = MultiTaskTextModel(
                vocab_size=len(vocab),
                embed_dim=cfg.get("embed_dim", 128),
                encoder_hidden=cfg.get("encoder_hidden", 256),
                num_layers=cfg.get("num_layers", 2),
                sentiment_classes=len(label_classes["sentiment"]),
                genre_classes=len(label_classes["genre"]),
                emotion_classes=len(label_classes["emotion"]),
                encoder_type=cfg.get("encoder_type", "transformer"),
                max_len=cfg.get("max_len", 256),
            ).to(device)
            
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            
            return {
                'model': model,
                'vocab': vocab,
                'label_classes': label_classes,
                'device': device,
                'max_len': cfg.get("max_len", 256)
            }
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return None
    
    def load_script(self, file_path: str) -> str:
        """
        Load script from .docx or .txt file
        
        Args:
            file_path: Path to script file
            
        Returns:
            Raw text content
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.docx':
            return self._load_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_docx(self, file_path: Path) -> str:
        """Load content from .docx file"""
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    
    def _load_txt(self, file_path: Path) -> str:
        """Load content from .txt file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_characters(self, script_text: str) -> List[Tuple[str, int, int]]:
        """
        Extract characters and rank by importance (dialogue count + presence)
        
        Args:
            script_text: Raw script text
            
        Returns:
            List of (character_name, dialogue_count, total_mentions) tuples
        """
        # Pattern to match character names (ALL CAPS, often at start of line)
        character_pattern = r'^([A-Z][A-Z\s]+?)(?:\s*\([^)]*\))?\s*$'
        
        # Find all character mentions
        character_mentions = []
        lines = script_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with character name
            match = re.match(character_pattern, line)
            if match:
                char_name = match.group(1).strip()
                # Clean up character name
                char_name = re.sub(r'\s+', ' ', char_name)
                character_mentions.append(char_name)
        
        # Count dialogue lines and total mentions
        char_stats = defaultdict(lambda: {'dialogue': 0, 'total': 0})
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for character name at start of line (dialogue)
            match = re.match(character_pattern, line)
            if match:
                char_name = match.group(1).strip()
                char_name = re.sub(r'\s+', ' ', char_name)
                char_stats[char_name]['dialogue'] += 1
            
            # Count all mentions of character names in the line
            for char in char_stats.keys():
                if char.lower() in line.lower():
                    char_stats[char]['total'] += 1
        
        # Calculate importance score and sort
        character_importance = []
        for char, stats in char_stats.items():
            if stats['dialogue'] > 0:  # Only include characters with dialogue
                importance_score = stats['dialogue'] * 2 + stats['total']
                character_importance.append((char, stats['dialogue'], stats['total'], importance_score))
        
        # Sort by importance score (descending)
        character_importance.sort(key=lambda x: x[3], reverse=True)
        
        return [(char, dialogue, total) for char, dialogue, total, _ in character_importance]
    
    def extract_locations(self, script_text: str) -> List[Tuple[str, int]]:
        """
        Extract locations from script
        
        Args:
            script_text: Raw script text
            
        Returns:
            List of (location_name, mention_count) tuples
        """
        # Pattern to match scene headings (INT./EXT. LOCATION - TIME)
        scene_pattern = r'^(INT\.|EXT\.)\s+([^-]+?)(?:\s*-\s*[^-]+)?\s*$'
        
        locations = []
        lines = script_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(scene_pattern, line, re.IGNORECASE)
            if match:
                location = match.group(2).strip()
                # Clean up location name
                location = re.sub(r'\s+', ' ', location)
                locations.append(location)
        
        # Count occurrences
        location_counts = Counter(locations)
        
        # Sort by frequency (descending)
        return sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    
    def generate_summary(self, script_text: str, max_sentences: int = 25, abstractive: bool = False) -> str:
        """
        Generate a comprehensive, detailed script synopsis that tells the complete story and plot.
        
        Args:
            script_text: Raw script text
            max_sentences: Target number of sentences in summary (default 25 for detailed synopsis)
            abstractive: Whether to use abstractive summarization
            
        Returns:
            Generated detailed story-focused summary text
        """
        cleaned = self._clean_script_text(script_text)
        
        # Try abstractive summarization first if requested
        if abstractive:
            abs_sum = self._summarize_abstractive(cleaned, target_sentences=max_sentences)
            if abs_sum and len(abs_sum.strip()) > 50:  # Ensure meaningful output
                # Validate abstractive output; if it looks generic/off-topic, fall through to detailed pipeline
                scenes_for_validation = self._split_into_scenes(cleaned)
                narrative_for_validation = self._extract_detailed_narrative_elements(scenes_for_validation) if scenes_for_validation else {}
                if not self._is_boilerplate_or_off_topic(abs_sum, cleaned, narrative_for_validation):
                    return self._enhance_summary_narrative(abs_sum)
        
        # Enhanced story-focused summarization with detailed analysis
        scenes = self._split_into_scenes(cleaned)

        if not scenes:
            return self._generate_detailed_story_summary(cleaned, max_sentences=max_sentences)
        
        # Extract comprehensive narrative elements and story structure
        narrative_elements = self._extract_detailed_narrative_elements(scenes)
        story_arc = self._analyze_detailed_story_arc(scenes)
        
        # Generate comprehensive scene summaries with full story details
        detailed_scene_summaries = []
        for i, scene_text in enumerate(scenes):
            scene_story = self._extract_detailed_scene_story(scene_text, i, len(scenes))
            if scene_story:
                detailed_scene_summaries.append(scene_story)
        
        # Create comprehensive, detailed plot summary
        detailed_plot_summary = self._create_detailed_plot_summary(detailed_scene_summaries, narrative_elements, story_arc, max_sentences)
        
        # Enhance with extensive character and conflict details
        enhanced_summary = self._enhance_with_detailed_story_details(detailed_plot_summary, narrative_elements, story_arc, max_sentences)
        
        # Final comprehensive narrative polish
        polished = self._polish_detailed_narrative_summary(enhanced_summary, max_sentences)
        
        # Safety: if polished still looks boilerplate/off-topic, expand and force include key names
        if self._is_boilerplate_or_off_topic(polished, cleaned, narrative_elements):
            forced = self._force_include_key_elements(polished, cleaned, narrative_elements, target_sentences=max(30, max_sentences))
            return self._polish_detailed_narrative_summary(forced, max(30, max_sentences))
        
        return polished

    def _summarize_abstractive(self, text: str, target_sentences: int = 7) -> Optional[str]:
        """Enhanced abstractive summarization via seq2seq with intelligent chunking."""
        try:
            if self.abstractive_model is None or self.abstractive_tokenizer is None:
                ok = self.load_abstractive_model()
                if not ok:
                    return None
            
            tokenizer = self.abstractive_tokenizer
            model = self.abstractive_model
            
            # Clean and preprocess text for better summarization
            cleaned_text = self._preprocess_for_abstractive(text)
            
            # Intelligent chunking based on scenes and content
            chunks = self._create_smart_chunks(cleaned_text, tokenizer)
            
            if not chunks:
                return None
            
            # Generate summaries for each chunk
            chunk_summaries = []
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                try:
                    inputs = tokenizer(
                        [chunk],
                        max_length=1024,
                        truncation=True,
                        return_tensors="pt",
                        padding=True
                    )
                    with torch.no_grad():
                        summary_ids = model.generate(
                            **inputs,
                            max_length=min(150 + target_sentences * 20, 300),
                            min_length=max(30, target_sentences * 8),
                            length_penalty=2.0,
                            num_beams=4,
                            early_stopping=True,
                            do_sample=False,
                            temperature=0.7,
                            no_repeat_ngram_size=3
                        )
                        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        if summary and len(summary.strip()) > 20:
                            chunk_summaries.append(summary.strip())
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            
            if not chunk_summaries:
                return None
            
            # Combine chunk summaries intelligently
            if len(chunk_summaries) == 1:
                return chunk_summaries[0]
            
            # If multiple chunks, create a final summary
            combined_text = " ".join(chunk_summaries)
            
            # Final summarization pass
            try:
                inputs = tokenizer(
                    [combined_text],
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    final_ids = model.generate(
                        **inputs,
                        max_length=min(200 + target_sentences * 25, 400),
                        min_length=max(40, target_sentences * 10),
                        length_penalty=2.5,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        temperature=0.8,
                        no_repeat_ngram_size=3
                    )
                final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
                
                # Post-process the summary
                return self._postprocess_abstractive_summary(final_summary, target_sentences)
                
            except Exception as e:
                print(f"Error in final summarization: {e}")
                # Fallback to simple combination
                return self._combine_chunk_summaries(chunk_summaries)
                
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            return None
    
    def _preprocess_for_abstractive(self, text: str) -> str:
        """Preprocess text for better abstractive summarization."""
        # Remove excessive formatting
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Focus on dialogue and key action
        lines = text.split('.')
        important_lines = []
        
        for line in lines:
            line = line.strip()
            if (len(line) > 10 and 
                any(word in line.lower() for word in ['says', 'tells', 'asks', 'replies', 'responds', 'shouts', 'whispers'])):
                important_lines.append(line)
        
        if important_lines:
            return '. '.join(important_lines) + '.'
        
        return text
    
    def _create_smart_chunks(self, text: str, tokenizer) -> List[str]:
        """Create intelligent chunks for abstractive summarization."""
        max_tokens = 768
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=True))
            
            # If adding this sentence exceeds limit, save current chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        # Ensure chunks aren't too small
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) >= 20:  # At least 20 words
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _extract_narrative_elements(self, scenes: List[str]) -> Dict[str, Any]:
        """Extract key narrative elements from scenes."""
        narrative_elements = {
            'main_characters': set(),
            'conflicts': [],
            'events': [],
            'locations': set(),
            'relationships': [],
            'themes': [],
            'tone': 'neutral'
        }
        
        for i, scene in enumerate(scenes):
            # Extract characters
            characters = self._extract_characters_from_scene(scene)
            narrative_elements['main_characters'].update(characters)
            
            # Extract conflicts and events
            if self._contains_conflict(scene):
                conflict_desc = self._describe_conflict(scene)
                narrative_elements['conflicts'].append(f"Scene {i+1}: {conflict_desc}")
            
            # Extract key events
            events = self._extract_key_events(scene)
            for event in events:
                narrative_elements['events'].append(f"Scene {i+1}: {event}")
            
            # Extract locations
            locations = self._extract_locations_from_scene(scene)
            narrative_elements['locations'].update(locations)
            
            # Determine tone
            scene_tone = self._analyze_scene_tone(scene)
            if scene_tone != 'neutral':
                narrative_elements['tone'] = scene_tone
        
        # Convert sets to lists for JSON serialization
        narrative_elements['main_characters'] = list(narrative_elements['main_characters'])
        narrative_elements['locations'] = list(narrative_elements['locations'])
        
        return narrative_elements
    
    def _analyze_story_arc(self, scenes: List[str]) -> Dict[str, Any]:
        """Analyze the overall story arc and structure."""
        story_arc = {
            'setup': [],
            'confrontation': [],
            'resolution': [],
            'turning_points': [],
            'climax_scene': -1,
            'total_scenes': len(scenes)
        }
        
        # Divide scenes into acts (rough approximation)
        setup_end = len(scenes) // 4
        confrontation_end = len(scenes) * 3 // 4
        
        for i, scene in enumerate(scenes):
            if i <= setup_end:
                story_arc['setup'].append(i)
            elif i <= confrontation_end:
                story_arc['confrontation'].append(i)
            else:
                story_arc['resolution'].append(i)
            
            # Look for turning points and climax
            if self._contains_climax(scene) or self._is_turning_point(scene, i, len(scenes)):
                story_arc['turning_points'].append(i)
                if self._contains_climax(scene):
                    story_arc['climax_scene'] = i
        
        return story_arc
    
    def _extract_scene_story(self, scene_text: str, scene_index: int, total_scenes: int) -> str:
        """Extract the actual story content from a scene."""
        # Extract dialogue content
        dialogue_content = self._extract_meaningful_dialogue(scene_text)
        
        # Extract action content
        action_content = self._extract_meaningful_action(scene_text)
        
        # Extract character interactions
        character_interactions = self._extract_character_interactions(scene_text)
        
        # Combine into story description
        story_parts = []
        
        if action_content:
            story_parts.append(action_content)
        
        if dialogue_content:
            story_parts.append(dialogue_content)
        
        if character_interactions:
            story_parts.append(character_interactions)
        
        # Create narrative description
        if story_parts:
            story_desc = " ".join(story_parts)
            # Add scene context
            position = "beginning" if scene_index < total_scenes * 0.3 else "middle" if scene_index < total_scenes * 0.7 else "end"
            return f"In the {position} of the story, {story_desc.lower()}"
        
        return ""
    
    def _create_plot_summary(self, scene_summaries: List[str], narrative_elements: Dict, story_arc: Dict, max_sentences: int) -> str:
        """Create a comprehensive plot summary from scene summaries and narrative elements."""
        if not scene_summaries:
            return "Unable to extract plot from script content."
        
        # Start with main characters and setting
        plot_intro = self._create_plot_introduction(narrative_elements)
        
        # Add story progression
        story_progression = self._create_story_progression(scene_summaries, story_arc)
        
        # Add conflicts and resolution
        conflicts_resolution = self._create_conflicts_resolution(narrative_elements, story_arc)
        
        # Combine all parts
        plot_parts = [plot_intro, story_progression, conflicts_resolution]
        plot_parts = [part for part in plot_parts if part.strip()]
        
        combined_plot = " ".join(plot_parts)
        
        # Summarize to target length
        return self._summarize_to_target_length(combined_plot, max_sentences)
    
    def _enhance_with_story_details(self, plot_summary: str, narrative_elements: Dict, max_sentences: int) -> str:
        """Enhance plot summary with character and conflict details."""
        if not plot_summary:
            return plot_summary
        
        # Add character details
        character_details = self._extract_character_details(narrative_elements)
        
        # Add conflict details
        conflict_details = self._extract_conflict_details(narrative_elements)
        
        # Enhance the summary
        enhanced_parts = [plot_summary]
        
        if character_details:
            enhanced_parts.append(character_details)
        
        if conflict_details:
            enhanced_parts.append(conflict_details)
        
        enhanced_summary = " ".join(enhanced_parts)
        
        # Ensure it doesn't exceed target length
        sentences = sent_tokenize(enhanced_summary)
        if len(sentences) > max_sentences:
            enhanced_summary = " ".join(sentences[:max_sentences])
        
        return enhanced_summary
    
    def _enhance_summary_narrative(self, summary: str) -> str:
        """Enhance abstractive summary to be more narrative-focused."""
        if not summary:
            return summary
        
        # Add narrative elements to make it more story-focused
        enhanced = summary
        
        # Replace generic terms with more specific narrative language
        replacements = {
            'characters': 'protagonists',
            'story': 'narrative',
            'events': 'plot developments',
            'situation': 'dramatic situation',
            'conflict': 'central conflict',
            'resolution': 'climactic resolution'
        }
        
        for generic, specific in replacements.items():
            enhanced = enhanced.replace(generic, specific)
        
        return enhanced
    
    def _generate_story_summary(self, text: str, max_sentences: int = 7) -> str:
        """Generate story-focused summary for text without clear scenes."""
        # Extract key story elements
        characters = self._extract_characters_from_text(text)
        conflicts = self._identify_conflicts_in_text(text)
        events = self._extract_events_from_text(text)
        
        # Create story description
        story_parts = []
        
        if characters:
            story_parts.append(f"The story follows {', '.join(characters[:3])}")
        
        if events:
            story_parts.append(f"Key events include: {'; '.join(events[:2])}")
        
        if conflicts:
            story_parts.append(f"Central conflicts involve: {'; '.join(conflicts[:2])}")
        
        if story_parts:
            story_summary = ". ".join(story_parts) + "."
            return self._summarize_to_target_length(story_summary, max_sentences)
        
        # Fallback to enhanced summarization
        return self._summarize_enhanced(text, max_sentences)
    
    def _polish_narrative_summary(self, summary: str, max_sentences: int) -> str:
        """Polish narrative summary for better storytelling."""
        if not summary or len(summary.strip()) < 20:
            return "Unable to generate a coherent story summary from the script content."
        
        # Ensure proper sentence structure and flow
        sentences = sent_tokenize(summary)
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Improve narrative flow
        polished_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                # Add narrative transitions
                if i > 0 and not sentence.startswith(('Then', 'Next', 'Meanwhile', 'However', 'Later', 'Finally')):
                    if 'conflict' in sentence.lower() or 'problem' in sentence.lower():
                        sentence = f"However, {sentence.lower()}"
                    elif 'resolution' in sentence.lower() or 'ends' in sentence.lower():
                        sentence = f"Finally, {sentence.lower()}"
                
                # Ensure proper ending
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                polished_sentences.append(sentence)
        
        final_summary = " ".join(polished_sentences)
        
        # Add narrative conclusion if needed
        if not any(word in final_summary.lower() for word in ['concludes', 'ends', 'resolves', 'finishes']):
            if len(polished_sentences) > 1:
                final_summary += " The story concludes with these dramatic developments."
        
        return final_summary
    
    # Helper methods for narrative extraction
    def _extract_characters_from_scene(self, scene: str) -> List[str]:
        """Extract character names from a scene."""
        characters = []
        lines = scene.split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', line):
                char_name = re.sub(r'\s+', ' ', line).strip()
                if len(char_name) > 2 and char_name not in characters:
                    characters.append(char_name)
        
        return characters[:5]  # Limit to main characters
    
    def _describe_conflict(self, scene: str) -> str:
        """Describe the conflict in a scene."""
        conflict_words = ['argue', 'fight', 'disagree', 'conflict', 'tension', 'angry', 'frustrated', 'attack', 'challenge']
        scene_lower = scene.lower()
        
        for word in conflict_words:
            if word in scene_lower:
                if 'argue' in scene_lower:
                    return "characters engage in heated argument"
                elif 'fight' in scene_lower:
                    return "physical confrontation occurs"
                elif 'disagree' in scene_lower:
                    return "characters disagree on important matters"
                elif 'tension' in scene_lower:
                    return "tension builds between characters"
        
        return "conflict emerges"
    
    def _extract_key_events(self, scene: str) -> List[str]:
        """Extract key events from a scene."""
        events = []
        scene_lower = scene.lower()
        
        event_indicators = {
            'discovers': 'character makes a discovery',
            'reveals': 'important information is revealed',
            'learns': 'character learns something significant',
            'meets': 'characters meet for the first time',
            'leaves': 'character departs or exits',
            'arrives': 'character arrives at location',
            'finds': 'character finds something important',
            'realizes': 'character has a realization'
        }
        
        for indicator, description in event_indicators.items():
            if indicator in scene_lower:
                events.append(description)
        
        return events[:2]  # Limit to most important events
    
    def _extract_locations_from_scene(self, scene: str) -> List[str]:
        """Extract locations from scene headings."""
        locations = []
        scene_pattern = r'^(INT\.|EXT\.)\s+([^-]+?)(?:\s*-\s*[^-]+)?\s*$'
        
        for line in scene.split('\n'):
            match = re.match(scene_pattern, line.strip(), re.IGNORECASE)
            if match:
                location = match.group(2).strip()
                if location and location not in locations:
                    locations.append(location)
        
        return locations
    
    def _analyze_scene_tone(self, scene: str) -> str:
        """Analyze the emotional tone of a scene."""
        scene_lower = scene.lower()
        
        if any(word in scene_lower for word in ['happy', 'joy', 'laugh', 'smile', 'celebrate']):
            return 'positive'
        elif any(word in scene_lower for word in ['sad', 'cry', 'depressed', 'lonely', 'grief']):
            return 'sad'
        elif any(word in scene_lower for word in ['angry', 'rage', 'furious', 'mad', 'hostile']):
            return 'angry'
        elif any(word in scene_lower for word in ['scared', 'afraid', 'terrified', 'fear', 'panic']):
            return 'tense'
        elif any(word in scene_lower for word in ['romantic', 'love', 'kiss', 'passion', 'intimate']):
            return 'romantic'
        
        return 'neutral'
    
    def _is_turning_point(self, scene: str, scene_index: int, total_scenes: int) -> bool:
        """Determine if a scene is a turning point in the story."""
        # Look for turning point indicators
        turning_indicators = ['suddenly', 'unexpectedly', 'shocking', 'reveals', 'discovers', 'realizes']
        scene_lower = scene.lower()
        
        has_indicators = any(indicator in scene_lower for indicator in turning_indicators)
        
        # Also consider position (middle of story is often where turning points occur)
        is_middle_scene = 0.3 < (scene_index / total_scenes) < 0.7
        
        return has_indicators or (is_middle_scene and len(scene.split()) > 100)
    
    def _extract_meaningful_dialogue(self, scene: str) -> str:
        """Extract meaningful dialogue content."""
        dialogue_lines = self._extract_dialogue(scene)
        if not dialogue_lines:
            return ""
        
        # Focus on dialogue that reveals plot or character
        meaningful_dialogue = []
        for line in dialogue_lines:
            if len(line.strip()) > 10 and any(word in line.lower() for word in ['will', 'going', 'must', 'need', 'want', 'why', 'how', 'what']):
                meaningful_dialogue.append(line.strip())
        
        if meaningful_dialogue:
            return f"characters discuss: '{'; '.join(meaningful_dialogue[:2])}'"
        
        return ""
    
    def _extract_meaningful_action(self, scene: str) -> str:
        """Extract meaningful action descriptions."""
        action_lines = self._extract_action(scene)
        if not action_lines:
            return ""
        
        # Focus on significant actions
        significant_actions = []
        for line in action_lines:
            if len(line.strip()) > 15 and any(word in line.lower() for word in ['runs', 'walks', 'enters', 'leaves', 'grabs', 'throws', 'falls', 'stands', 'looks', 'turns']):
                significant_actions.append(line.strip())
        
        if significant_actions:
            return f"key actions include: {significant_actions[0]}"
        
        return ""
    
    def _extract_character_interactions(self, scene: str) -> str:
        """Extract character interaction descriptions."""
        characters = self._extract_characters_from_scene(scene)
        if len(characters) < 2:
            return ""
        
        # Look for interaction indicators
        scene_lower = scene.lower()
        interactions = []
        
        if 'meets' in scene_lower or 'encounters' in scene_lower:
            interactions.append(f"{characters[0]} meets {characters[1]}")
        elif 'talks' in scene_lower or 'speaks' in scene_lower:
            interactions.append(f"{characters[0]} and {characters[1]} have a conversation")
        elif 'argues' in scene_lower:
            interactions.append(f"{characters[0]} and {characters[1]} argue")
        
        return "; ".join(interactions)
    
    def _create_plot_introduction(self, narrative_elements: Dict) -> str:
        """Create plot introduction with main characters and setting."""
        intro_parts = []
        
        if narrative_elements.get('main_characters'):
            main_chars = narrative_elements['main_characters'][:3]
            intro_parts.append(f"The story centers around {', '.join(main_chars)}")
        
        if narrative_elements.get('locations'):
            main_locations = narrative_elements['locations'][:2]
            intro_parts.append(f"set primarily in {', '.join(main_locations)}")
        
        if intro_parts:
            return ". ".join(intro_parts) + "."
        
        return ""
    
    def _create_story_progression(self, scene_summaries: List[str], story_arc: Dict) -> str:
        """Create story progression description."""
        if not scene_summaries:
            return ""
        
        progression_parts = []
        
        # Beginning setup
        setup_scenes = [scene_summaries[i] for i in story_arc.get('setup', []) if i < len(scene_summaries)]
        if setup_scenes:
            progression_parts.append(f"The story begins as {setup_scenes[0].lower()}")
        
        # Middle confrontation
        confrontation_scenes = [scene_summaries[i] for i in story_arc.get('confrontation', []) if i < len(scene_summaries)]
        if confrontation_scenes:
            progression_parts.append(f"As the plot develops, {confrontation_scenes[0].lower()}")
        
        # Resolution
        resolution_scenes = [scene_summaries[i] for i in story_arc.get('resolution', []) if i < len(scene_summaries)]
        if resolution_scenes:
            progression_parts.append(f"The story concludes when {resolution_scenes[0].lower()}")
        
        return " ".join(progression_parts)
    
    def _create_conflicts_resolution(self, narrative_elements: Dict, story_arc: Dict) -> str:
        """Create conflicts and resolution description."""
        conflict_parts = []
        
        if narrative_elements.get('conflicts'):
            main_conflicts = narrative_elements['conflicts'][:2]
            conflict_parts.append(f"Central conflicts include: {'; '.join(main_conflicts)}")
        
        if story_arc.get('climax_scene', -1) >= 0:
            conflict_parts.append("The story builds to a climactic moment that resolves the main tensions")
        
        return " ".join(conflict_parts)
    
    def _extract_character_details(self, narrative_elements: Dict) -> str:
        """Extract character details for enhancement."""
        if not narrative_elements.get('main_characters'):
            return ""
        
        characters = narrative_elements['main_characters'][:2]
        return f"The narrative explores the relationships and motivations of {', '.join(characters)}"
    
    def _extract_conflict_details(self, narrative_elements: Dict) -> str:
        """Extract conflict details for enhancement."""
        if not narrative_elements.get('conflicts'):
            return ""
        
        return "The story examines themes of conflict, resolution, and personal growth"
    
    def _summarize_to_target_length(self, text: str, max_sentences: int) -> str:
        """Summarize text to target sentence length."""
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Use enhanced summarization to condense
        return self._summarize_enhanced(text, max_sentences)
    
    def _extract_characters_from_text(self, text: str) -> List[str]:
        """Extract characters from general text."""
        characters = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', line):
                char_name = re.sub(r'\s+', ' ', line).strip()
                if len(char_name) > 2 and char_name not in characters:
                    characters.append(char_name)
        
        return characters[:3]
    
    def _identify_conflicts_in_text(self, text: str) -> List[str]:
        """Identify conflicts in general text."""
        conflicts = []
        text_lower = text.lower()
        
        conflict_patterns = [
            ('argue', 'argument between characters'),
            ('fight', 'physical confrontation'),
            ('disagree', 'disagreement on important matters'),
            ('conflict', 'internal or external conflict'),
            ('tension', 'building tension')
        ]
        
        for pattern, description in conflict_patterns:
            if pattern in text_lower:
                conflicts.append(description)
        
        return conflicts[:2]
    
    def _extract_events_from_text(self, text: str) -> List[str]:
        """Extract key events from general text."""
        events = []
        text_lower = text.lower()
        
        event_patterns = [
            ('discovers', 'character makes important discovery'),
            ('reveals', 'important information revealed'),
            ('meets', 'characters meet'),
            ('leaves', 'character departure'),
            ('arrives', 'character arrival')
        ]
        
        for pattern, description in event_patterns:
            if pattern in text_lower:
                events.append(description)
        
        return events[:2]
    
    # Detailed summarization methods for comprehensive synopsis
    def _extract_detailed_narrative_elements(self, scenes: List[str]) -> Dict[str, Any]:
        """Extract comprehensive narrative elements from scenes with detailed analysis."""
        narrative_elements = {
            'main_characters': set(),
            'secondary_characters': set(),
            'conflicts': [],
            'events': [],
            'locations': set(),
            'relationships': [],
            'themes': [],
            'tone': 'neutral',
            'dialogue_themes': [],
            'character_arcs': {},
            'emotional_journey': [],
            'plot_twists': [],
            'symbols': [],
            'time_progression': []
        }
        
        for i, scene in enumerate(scenes):
            # Extract characters with detailed analysis
            characters = self._extract_characters_from_scene(scene)
            main_chars = characters[:3]  # Top 3 are main characters
            secondary_chars = characters[3:]  # Rest are secondary
            
            narrative_elements['main_characters'].update(main_chars)
            narrative_elements['secondary_characters'].update(secondary_chars)
            
            # Extract detailed conflicts and events
            if self._contains_conflict(scene):
                conflict_desc = self._describe_detailed_conflict(scene, i)
                narrative_elements['conflicts'].append(conflict_desc)
            
            # Extract comprehensive events
            events = self._extract_detailed_events(scene, i)
            narrative_elements['events'].extend(events)
            
            # Extract locations with context
            locations = self._extract_locations_from_scene(scene)
            narrative_elements['locations'].update(locations)
            
            # Analyze dialogue themes
            dialogue_themes = self._analyze_dialogue_themes(scene)
            narrative_elements['dialogue_themes'].extend(dialogue_themes)
            
            # Track character development
            character_arcs = self._track_character_development(scene, characters, i)
            for char, arc in character_arcs.items():
                if char not in narrative_elements['character_arcs']:
                    narrative_elements['character_arcs'][char] = []
                narrative_elements['character_arcs'][char].append(arc)
            
            # Track emotional journey
            emotional_state = self._analyze_emotional_state(scene)
            narrative_elements['emotional_journey'].append(f"Scene {i+1}: {emotional_state}")
            
            # Look for plot twists
            if self._contains_plot_twist(scene):
                twist_desc = self._describe_plot_twist(scene, i)
                narrative_elements['plot_twists'].append(twist_desc)
            
            # Determine tone with more nuance
            scene_tone = self._analyze_detailed_scene_tone(scene)
            if scene_tone != 'neutral':
                narrative_elements['tone'] = scene_tone
        
        # Convert sets to lists for JSON serialization
        narrative_elements['main_characters'] = list(narrative_elements['main_characters'])
        narrative_elements['secondary_characters'] = list(narrative_elements['secondary_characters'])
        narrative_elements['locations'] = list(narrative_elements['locations'])
        
        return narrative_elements
    
    def _analyze_detailed_story_arc(self, scenes: List[str]) -> Dict[str, Any]:
        """Analyze the detailed story arc with comprehensive structure analysis."""
        story_arc = {
            'setup': [],
            'inciting_incident': [],
            'rising_action': [],
            'climax': [],
            'falling_action': [],
            'resolution': [],
            'turning_points': [],
            'climax_scene': -1,
            'total_scenes': len(scenes),
            'act_breaks': [],
            'pacing_analysis': [],
            'tension_points': []
        }
        
        # More sophisticated act division
        setup_end = len(scenes) // 4
        inciting_end = len(scenes) // 6
        rising_end = len(scenes) * 3 // 4
        falling_end = len(scenes) * 5 // 6
        
        for i, scene in enumerate(scenes):
            # Detailed act classification
            if i <= inciting_end:
                story_arc['setup'].append(i)
            elif i <= setup_end:
                story_arc['inciting_incident'].append(i)
            elif i <= rising_end:
                story_arc['rising_action'].append(i)
            elif i <= falling_end:
                story_arc['climax'].append(i)
            elif i < len(scenes) - 2:
                story_arc['falling_action'].append(i)
            else:
                story_arc['resolution'].append(i)
            
            # Look for turning points and climax with more detail
            if self._contains_climax(scene) or self._is_detailed_turning_point(scene, i, len(scenes)):
                story_arc['turning_points'].append(i)
                if self._contains_climax(scene):
                    story_arc['climax_scene'] = i
            
            # Analyze pacing
            pacing = self._analyze_scene_pacing(scene, i)
            story_arc['pacing_analysis'].append(pacing)
            
            # Identify tension points
            if self._contains_tension(scene):
                story_arc['tension_points'].append(i)
        
        return story_arc
    
    def _extract_detailed_scene_story(self, scene_text: str, scene_index: int, total_scenes: int) -> str:
        """Extract comprehensive story content from a scene with full detail."""
        # Extract all dialogue content with context
        dialogue_content = self._extract_comprehensive_dialogue(scene_text)
        
        # Extract all action content with detail
        action_content = self._extract_comprehensive_action(scene_text)
        
        # Extract character interactions with relationship context
        character_interactions = self._extract_detailed_character_interactions(scene_text)
        
        # Extract emotional and thematic content
        emotional_content = self._extract_emotional_content(scene_text)
        
        # Extract setting and atmosphere
        setting_content = self._extract_setting_content(scene_text)
        
        # Combine into comprehensive story description
        story_parts = []
        
        if setting_content:
            story_parts.append(setting_content)
        
        if action_content:
            story_parts.append(action_content)
        
        if dialogue_content:
            story_parts.append(dialogue_content)
        
        if character_interactions:
            story_parts.append(character_interactions)
        
        if emotional_content:
            story_parts.append(emotional_content)
        
        # Create detailed narrative description
        if story_parts:
            story_desc = " ".join(story_parts)
            # Add detailed scene context
            position = self._determine_detailed_scene_position(scene_index, total_scenes)
            return f"In {position}, {story_desc.lower()}"
        
        return ""
    
    def _create_detailed_plot_summary(self, scene_summaries: List[str], narrative_elements: Dict, story_arc: Dict, max_sentences: int) -> str:
        """Create a comprehensive, detailed plot summary from scene summaries and narrative elements."""
        if not scene_summaries:
            return "Unable to extract detailed plot from script content."
        
        # Create comprehensive story structure
        plot_sections = []
        
        # 1. Detailed introduction with characters and setting
        detailed_intro = self._create_detailed_plot_introduction(narrative_elements, story_arc)
        if detailed_intro:
            plot_sections.append(detailed_intro)
        
        # 2. Comprehensive story progression with all scenes
        detailed_progression = self._create_detailed_story_progression(scene_summaries, story_arc)
        if detailed_progression:
            plot_sections.append(detailed_progression)
        
        # 3. Detailed character development and relationships
        character_development = self._create_character_development_summary(narrative_elements)
        if character_development:
            plot_sections.append(character_development)
        
        # 4. Comprehensive conflict analysis
        conflict_analysis = self._create_detailed_conflicts_analysis(narrative_elements, story_arc)
        if conflict_analysis:
            plot_sections.append(conflict_analysis)
        
        # 5. Emotional journey and thematic elements
        thematic_analysis = self._create_thematic_analysis(narrative_elements)
        if thematic_analysis:
            plot_sections.append(thematic_analysis)
        
        # 6. Resolution and conclusion details
        resolution_details = self._create_detailed_resolution(narrative_elements, story_arc)
        if resolution_details:
            plot_sections.append(resolution_details)
        
        # Combine all sections
        comprehensive_plot = " ".join(plot_sections)
        
        # Expand to target length if needed
        return self._expand_to_target_length(comprehensive_plot, max_sentences)
    
    def _enhance_with_detailed_story_details(self, plot_summary: str, narrative_elements: Dict, story_arc: Dict, max_sentences: int) -> str:
        """Enhance plot summary with extensive character, conflict, and thematic details."""
        if not plot_summary:
            return plot_summary
        
        # Add comprehensive character analysis
        character_analysis = self._create_comprehensive_character_analysis(narrative_elements)
        
        # Add detailed conflict resolution analysis
        conflict_resolution = self._create_conflict_resolution_analysis(narrative_elements, story_arc)
        
        # Add thematic and symbolic analysis
        thematic_analysis = self._create_symbolic_thematic_analysis(narrative_elements)
        
        # Add emotional and psychological analysis
        psychological_analysis = self._create_psychological_analysis(narrative_elements)
        
        # Enhance the summary
        enhanced_parts = [plot_summary]
        
        if character_analysis:
            enhanced_parts.append(character_analysis)
        
        if conflict_resolution:
            enhanced_parts.append(conflict_resolution)
        
        if thematic_analysis:
            enhanced_parts.append(thematic_analysis)
        
        if psychological_analysis:
            enhanced_parts.append(psychological_analysis)
        
        enhanced_summary = " ".join(enhanced_parts)
        
        # Ensure comprehensive coverage
        return self._ensure_comprehensive_coverage(enhanced_summary, max_sentences)
    
    def _polish_detailed_narrative_summary(self, summary: str, max_sentences: int) -> str:
        """Polish detailed narrative summary for comprehensive storytelling."""
        if not summary or len(summary.strip()) < 50:
            return "Unable to generate a comprehensive story summary from the script content."
        
        # Ensure proper sentence structure and flow for detailed narrative
        sentences = sent_tokenize(summary)
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Improve detailed narrative flow with better transitions
        polished_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:  # Longer sentences for detailed summary
                # Add sophisticated narrative transitions
                if i > 0:
                    sentence = self._add_sophisticated_transition(sentence, i, sentences)
                
                # Ensure proper ending
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                polished_sentences.append(sentence)
        
        final_summary = " ".join(polished_sentences)
        
        # Add comprehensive narrative conclusion
        if not any(word in final_summary.lower() for word in ['concludes', 'ends', 'resolves', 'finishes', 'culminates']):
            if len(polished_sentences) > 3:
                final_summary += " The narrative culminates in a satisfying resolution that ties together all the story threads and character arcs."
        
        return final_summary

    def _is_boilerplate_or_off_topic(self, summary: str, original_text: str, narrative_elements: Dict) -> bool:
        """Detects boilerplate, templated, or off-topic summaries that don't describe the story."""
        if not summary:
            return True
        s = summary.lower()
        # Heuristics for boilerplate phrases that we've observed
        boilerplate_markers = [
            'sample screenplay collection includes scenes',
            'the script is set in an apartment in the middle of the night',
            'the main narrative is set on the beach at the end of the day',
            'the narrative is based on the protagonists\' lives and experiences',
            'the movie is set for release on',
            'for confidential support call the samaritans',
        ]
        if any(marker in s for marker in boilerplate_markers):
            return True
        
        # If it mentions release dates, directors lists, or city lists excessively, likely off-topic metadata
        meta_markers = ['directed by', 'release on', 'usa, london', 'new york city']
        meta_hits = sum(1 for m in meta_markers if m in s)
        if meta_hits >= 2:
            return True
        
        # Require presence of at least one main character name if we have them
        main_chars = [c.lower() for c in narrative_elements.get('main_characters', [])] if narrative_elements else []
        if main_chars and not any(c in s for c in main_chars):
            return True
        
        # Require some story verbs
        story_verbs = ['decides', 'discovers', 'reveals', 'confronts', 'tries', 'fails', 'learns', 'changes', 'agrees', 'betrays', 'escapes']
        if not any(v in s for v in story_verbs):
            # Allow if the original text is extremely short
            if len(original_text.split()) > 100:
                return True
        
        return False

    def _force_include_key_elements(self, summary: str, original_text: str, narrative_elements: Dict, target_sentences: int = 30) -> str:
        """Expand and inject concrete story details (characters, settings, conflicts) into the summary."""
        parts = [summary.strip()]
        
        main_chars = narrative_elements.get('main_characters', []) if narrative_elements else []
        locations = narrative_elements.get('locations', []) if narrative_elements else []
        conflicts = narrative_elements.get('conflicts', []) if narrative_elements else []
        events = narrative_elements.get('events', []) if narrative_elements else []
        
        if main_chars:
            parts.append(f"Key characters include {', '.join(main_chars[:4])}, whose choices and relationships actively drive the plot forward.")
        if locations:
            parts.append(f"Significant locations such as {', '.join(locations[:3])} frame the action and emotional stakes of each scene.")
        if conflicts:
            parts.append(f"Persistent sources of conflict shape the narrative trajectory: {'; '.join(conflicts[:3])}.")
        if events:
            parts.append(f"Pivotal developments punctuate the story: {'; '.join(events[:4])}.")
        
        # Ensure length
        expanded = ' '.join(parts)
        sentences = sent_tokenize(expanded)
        while len(sentences) < target_sentences:
            sentences.append("As tensions escalate, characters are forced to confront their vulnerabilities, making consequential decisions that alter the course of the story.")
        return ' '.join(sentences[:target_sentences])
    
    def _generate_detailed_story_summary(self, text: str, max_sentences: int = 25) -> str:
        """Generate comprehensive story-focused summary for text without clear scenes."""
        # Extract comprehensive story elements
        characters = self._extract_characters_from_text(text)
        conflicts = self._identify_conflicts_in_text(text)
        events = self._extract_events_from_text(text)
        themes = self._identify_themes_in_text(text)
        emotional_arc = self._analyze_emotional_arc(text)
        
        # Create detailed story description
        story_parts = []
        
        if characters:
            story_parts.append(f"The narrative follows the journey of {', '.join(characters[:3])} as they navigate through a complex series of events and relationships.")
        
        if events:
            story_parts.append(f"The story unfolds through several key developments: {'; '.join(events[:3])}.")
        
        if conflicts:
            story_parts.append(f"Central to the narrative are the conflicts that arise: {'; '.join(conflicts[:3])}.")
        
        if themes:
            story_parts.append(f"The story explores themes of {', '.join(themes[:2])}, adding depth to the character development.")
        
        if emotional_arc:
            story_parts.append(f"The emotional journey of the characters evolves from {emotional_arc[0]} to {emotional_arc[-1]}, creating a compelling arc.")
        
        if story_parts:
            detailed_story_summary = " ".join(story_parts)
            return self._expand_to_target_length(detailed_story_summary, max_sentences)
        
        # Fallback to enhanced summarization with more detail
        return self._summarize_enhanced(text, max_sentences)
    
    # Helper methods for detailed analysis
    def _describe_detailed_conflict(self, scene: str, scene_index: int) -> str:
        """Describe conflicts in detail with context."""
        conflict_words = ['argue', 'fight', 'disagree', 'conflict', 'tension', 'angry', 'frustrated', 'attack', 'challenge', 'oppose']
        scene_lower = scene.lower()
        
        for word in conflict_words:
            if word in scene_lower:
                if 'argue' in scene_lower:
                    return f"Scene {scene_index+1}: Characters engage in a heated argument that reveals deeper tensions and conflicting perspectives"
                elif 'fight' in scene_lower:
                    return f"Scene {scene_index+1}: Physical confrontation occurs, escalating the dramatic tension"
                elif 'disagree' in scene_lower:
                    return f"Scene {scene_index+1}: Characters disagree on fundamental issues, creating ideological conflict"
                elif 'tension' in scene_lower:
                    return f"Scene {scene_index+1}: Tension builds between characters, creating suspense and emotional stakes"
        
        return f"Scene {scene_index+1}: Conflict emerges through character interactions and opposing motivations"
    
    def _extract_detailed_events(self, scene: str, scene_index: int) -> List[str]:
        """Extract detailed events from a scene."""
        events = []
        scene_lower = scene.lower()
        
        detailed_event_indicators = {
            'discovers': f'Scene {scene_index+1}: A character makes a crucial discovery that changes the course of the narrative',
            'reveals': f'Scene {scene_index+1}: Important information is revealed, providing new insight into the story',
            'learns': f'Scene {scene_index+1}: A character learns something significant that impacts their understanding',
            'meets': f'Scene {scene_index+1}: Characters meet for the first time, establishing new relationships',
            'leaves': f'Scene {scene_index+1}: A character departs, creating separation and potential conflict',
            'arrives': f'Scene {scene_index+1}: A character arrives at a crucial location, advancing the plot',
            'finds': f'Scene {scene_index+1}: A character finds something important that drives the story forward',
            'realizes': f'Scene {scene_index+1}: A character has a moment of realization that changes their perspective',
            'decides': f'Scene {scene_index+1}: A character makes an important decision that affects the narrative',
            'confronts': f'Scene {scene_index+1}: A character confronts another, escalating dramatic tension'
        }
        
        for indicator, description in detailed_event_indicators.items():
            if indicator in scene_lower:
                events.append(description)
        
        return events[:3]  # Limit to most important events
    
    def _analyze_dialogue_themes(self, scene: str) -> List[str]:
        """Analyze themes present in dialogue."""
        themes = []
        scene_lower = scene.lower()
        
        theme_indicators = {
            'love': 'romantic themes',
            'family': 'family relationships',
            'work': 'professional themes',
            'money': 'financial concerns',
            'power': 'power dynamics',
            'justice': 'moral themes',
            'truth': 'truth and deception',
            'freedom': 'freedom and constraint',
            'death': 'mortality themes',
            'hope': 'hope and despair'
        }
        
        for indicator, theme in theme_indicators.items():
            if indicator in scene_lower:
                themes.append(theme)
        
        return themes
    
    def _track_character_development(self, scene: str, characters: List[str], scene_index: int) -> Dict[str, str]:
        """Track character development throughout the story."""
        character_arcs = {}
        
        for char in characters[:3]:  # Focus on main characters
            char_arc = self._analyze_character_growth(scene, char, scene_index)
            if char_arc:
                character_arcs[char] = char_arc
        
        return character_arcs
    
    def _analyze_character_growth(self, scene: str, character: str, scene_index: int) -> str:
        """Analyze how a character grows or changes in a scene."""
        scene_lower = scene.lower()
        
        growth_indicators = {
            'learns': f'{character} learns something important',
            'realizes': f'{character} has a moment of realization',
            'changes': f'{character} undergoes a change',
            'grows': f'{character} shows personal growth',
            'develops': f'{character} develops new understanding',
            'transforms': f'{character} transforms in some way'
        }
        
        for indicator, description in growth_indicators.items():
            if indicator in scene_lower:
                return description
        
        return ""
    
    def _analyze_emotional_state(self, scene: str) -> str:
        """Analyze the emotional state of a scene."""
        scene_lower = scene.lower()
        
        emotional_states = {
            'happy': 'joy and celebration',
            'sad': 'sorrow and melancholy',
            'angry': 'anger and frustration',
            'scared': 'fear and anxiety',
            'excited': 'excitement and anticipation',
            'confused': 'confusion and uncertainty',
            'hopeful': 'hope and optimism',
            'desperate': 'desperation and urgency',
            'peaceful': 'peace and tranquility',
            'tense': 'tension and suspense'
        }
        
        for state, description in emotional_states.items():
            if state in scene_lower:
                return description
        
        return "neutral emotional tone"
    
    def _contains_plot_twist(self, scene: str) -> bool:
        """Check if scene contains a plot twist."""
        twist_indicators = ['suddenly', 'unexpectedly', 'shocking', 'reveals', 'discovers', 'realizes', 'shocking truth', 'unexpected turn']
        scene_lower = scene.lower()
        return any(indicator in scene_lower for indicator in twist_indicators)
    
    def _describe_plot_twist(self, scene: str, scene_index: int) -> str:
        """Describe a plot twist in detail."""
        return f"Scene {scene_index+1}: A shocking revelation or unexpected turn of events that dramatically changes the story's direction"
    
    def _analyze_detailed_scene_tone(self, scene: str) -> str:
        """Analyze scene tone with more nuance."""
        scene_lower = scene.lower()
        
        if any(word in scene_lower for word in ['happy', 'joy', 'laugh', 'smile', 'celebrate', 'triumph']):
            return 'positive'
        elif any(word in scene_lower for word in ['sad', 'cry', 'depressed', 'lonely', 'grief', 'mourn']):
            return 'melancholic'
        elif any(word in scene_lower for word in ['angry', 'rage', 'furious', 'mad', 'hostile', 'violent']):
            return 'aggressive'
        elif any(word in scene_lower for word in ['scared', 'afraid', 'terrified', 'fear', 'panic', 'horror']):
            return 'tense'
        elif any(word in scene_lower for word in ['romantic', 'love', 'kiss', 'passion', 'intimate', 'tender']):
            return 'romantic'
        elif any(word in scene_lower for word in ['mysterious', 'secret', 'hidden', 'unknown', 'enigmatic']):
            return 'mysterious'
        elif any(word in scene_lower for word in ['dramatic', 'intense', 'climactic', 'crucial', 'pivotal']):
            return 'dramatic'
        
        return 'neutral'
    
    def _is_detailed_turning_point(self, scene: str, scene_index: int, total_scenes: int) -> bool:
        """Determine if a scene is a detailed turning point."""
        turning_indicators = ['suddenly', 'unexpectedly', 'shocking', 'reveals', 'discovers', 'realizes', 'decision', 'choice', 'moment']
        scene_lower = scene.lower()
        
        has_indicators = any(indicator in scene_lower for indicator in turning_indicators)
        
        # More sophisticated position analysis
        is_middle_scene = 0.25 < (scene_index / total_scenes) < 0.75
        is_significant_length = len(scene.split()) > 150
        
        return has_indicators and (is_middle_scene or is_significant_length)
    
    def _analyze_scene_pacing(self, scene: str, scene_index: int) -> str:
        """Analyze the pacing of a scene."""
        word_count = len(scene.split())
        
        if word_count < 50:
            return f"Scene {scene_index+1}: Fast-paced, quick action"
        elif word_count < 150:
            return f"Scene {scene_index+1}: Moderate pacing with dialogue"
        else:
            return f"Scene {scene_index+1}: Slow-paced, detailed exploration"
    
    def _contains_tension(self, scene: str) -> bool:
        """Check if scene contains tension."""
        tension_words = ['tension', 'suspense', 'dramatic', 'intense', 'climactic', 'crucial', 'pivotal', 'urgent']
        scene_lower = scene.lower()
        return any(word in scene_lower for word in tension_words)
    
    def _extract_comprehensive_dialogue(self, scene: str) -> str:
        """Extract comprehensive dialogue content."""
        dialogue_lines = self._extract_dialogue(scene)
        if not dialogue_lines:
            return ""
        
        # Focus on dialogue that reveals plot, character, or themes
        meaningful_dialogue = []
        for line in dialogue_lines:
            if len(line.strip()) > 15 and any(word in line.lower() for word in ['will', 'going', 'must', 'need', 'want', 'why', 'how', 'what', 'when', 'where', 'because', 'if', 'then']):
                meaningful_dialogue.append(line.strip())
        
        if meaningful_dialogue:
            return f"characters engage in meaningful conversation, discussing: '{'; '.join(meaningful_dialogue[:3])}'"
        
        return ""
    
    def _extract_comprehensive_action(self, scene: str) -> str:
        """Extract comprehensive action descriptions."""
        action_lines = self._extract_action(scene)
        if not action_lines:
            return ""
        
        # Focus on significant actions that drive the plot
        significant_actions = []
        for line in action_lines:
            if len(line.strip()) > 20 and any(word in line.lower() for word in ['runs', 'walks', 'enters', 'leaves', 'grabs', 'throws', 'falls', 'stands', 'looks', 'turns', 'opens', 'closes', 'pushes', 'pulls', 'reaches', 'moves']):
                significant_actions.append(line.strip())
        
        if significant_actions:
            return f"key actions unfold as: {significant_actions[0]}"
        
        return ""
    
    def _extract_detailed_character_interactions(self, scene: str) -> str:
        """Extract detailed character interaction descriptions."""
        characters = self._extract_characters_from_scene(scene)
        if len(characters) < 2:
            return ""
        
        # Look for detailed interaction indicators
        scene_lower = scene.lower()
        interactions = []
        
        if 'meets' in scene_lower or 'encounters' in scene_lower:
            interactions.append(f"{characters[0]} meets {characters[1]} for the first time, establishing a new relationship dynamic")
        elif 'talks' in scene_lower or 'speaks' in scene_lower:
            interactions.append(f"{characters[0]} and {characters[1]} engage in a meaningful conversation that reveals their relationship")
        elif 'argues' in scene_lower:
            interactions.append(f"{characters[0]} and {characters[1]} argue, exposing tensions in their relationship")
        elif 'helps' in scene_lower:
            interactions.append(f"{characters[0]} helps {characters[1]}, showing their supportive relationship")
        elif 'confronts' in scene_lower:
            interactions.append(f"{characters[0]} confronts {characters[1]}, creating dramatic tension")
        
        return "; ".join(interactions)
    
    def _extract_emotional_content(self, scene: str) -> str:
        """Extract emotional content from scene."""
        emotional_words = ['happy', 'sad', 'angry', 'scared', 'excited', 'confused', 'hopeful', 'desperate', 'peaceful', 'tense']
        scene_lower = scene.lower()
        
        emotions_found = [emotion for emotion in emotional_words if emotion in scene_lower]
        
        if emotions_found:
            return f"the emotional tone shifts to {', '.join(emotions_found[:2])}"
        
        return ""
    
    def _extract_setting_content(self, scene: str) -> str:
        """Extract setting and atmosphere content."""
        setting_indicators = ['dark', 'bright', 'quiet', 'loud', 'crowded', 'empty', 'warm', 'cold', 'stormy', 'peaceful']
        scene_lower = scene.lower()
        
        settings_found = [setting for setting in setting_indicators if setting in scene_lower]
        
        if settings_found:
            return f"the scene is set in a {', '.join(settings_found[:2])} environment"
        
        return ""
    
    def _determine_detailed_scene_position(self, scene_index: int, total_scenes: int) -> str:
        """Determine detailed scene position in the story."""
        ratio = scene_index / max(total_scenes - 1, 1)
        
        if ratio < 0.15:
            return "the opening moments of the story"
        elif ratio < 0.3:
            return "the early stages of the narrative"
        elif ratio < 0.5:
            return "the first half of the story"
        elif ratio < 0.7:
            return "the middle section of the narrative"
        elif ratio < 0.85:
            return "the latter part of the story"
        else:
            return "the concluding moments of the narrative"
    
    def _create_detailed_plot_introduction(self, narrative_elements: Dict, story_arc: Dict) -> str:
        """Create detailed plot introduction."""
        intro_parts = []
        
        if narrative_elements.get('main_characters'):
            main_chars = narrative_elements['main_characters'][:3]
            intro_parts.append(f"The story centers around {', '.join(main_chars)}, each bringing their own unique perspective and motivations to the narrative")
        
        if narrative_elements.get('locations'):
            main_locations = narrative_elements['locations'][:3]
            intro_parts.append(f"primarily set in {', '.join(main_locations)}, which serve as the backdrop for the unfolding drama")
        
        if narrative_elements.get('tone'):
            tone = narrative_elements['tone']
            intro_parts.append(f"establishing a {tone} tone that permeates throughout the story")
        
        if intro_parts:
            return ". ".join(intro_parts) + "."
        
        return ""
    
    def _create_detailed_story_progression(self, scene_summaries: List[str], story_arc: Dict) -> str:
        """Create detailed story progression."""
        if not scene_summaries:
            return ""
        
        progression_parts = []
        
        # Beginning setup
        setup_scenes = [scene_summaries[i] for i in story_arc.get('setup', []) if i < len(scene_summaries)]
        if setup_scenes:
            progression_parts.append(f"The narrative begins as {setup_scenes[0].lower()}")
        
        # Inciting incident
        inciting_scenes = [scene_summaries[i] for i in story_arc.get('inciting_incident', []) if i < len(scene_summaries)]
        if inciting_scenes:
            progression_parts.append(f"An inciting incident occurs when {inciting_scenes[0].lower()}")
        
        # Rising action
        rising_scenes = [scene_summaries[i] for i in story_arc.get('rising_action', []) if i < len(scene_summaries)]
        if rising_scenes:
            progression_parts.append(f"As the story develops, {rising_scenes[0].lower()}")
        
        # Climax
        climax_scenes = [scene_summaries[i] for i in story_arc.get('climax', []) if i < len(scene_summaries)]
        if climax_scenes:
            progression_parts.append(f"The story reaches its climax when {climax_scenes[0].lower()}")
        
        # Resolution
        resolution_scenes = [scene_summaries[i] for i in story_arc.get('resolution', []) if i < len(scene_summaries)]
        if resolution_scenes:
            progression_parts.append(f"The story concludes as {resolution_scenes[0].lower()}")
        
        return " ".join(progression_parts)
    
    def _create_character_development_summary(self, narrative_elements: Dict) -> str:
        """Create character development summary."""
        if not narrative_elements.get('character_arcs'):
            return ""
        
        char_development = []
        for char, arcs in narrative_elements['character_arcs'].items():
            if arcs:
                char_development.append(f"{char} undergoes significant development throughout the story, experiencing moments of growth and realization")
        
        if char_development:
            return " ".join(char_development) + "."
        
        return ""
    
    def _create_detailed_conflicts_analysis(self, narrative_elements: Dict, story_arc: Dict) -> str:
        """Create detailed conflicts analysis."""
        conflict_parts = []
        
        if narrative_elements.get('conflicts'):
            main_conflicts = narrative_elements['conflicts'][:3]
            conflict_parts.append(f"Central conflicts drive the narrative forward: {'; '.join(main_conflicts)}")
        
        if narrative_elements.get('plot_twists'):
            twists = narrative_elements['plot_twists'][:2]
            conflict_parts.append(f"Plot twists add complexity: {'; '.join(twists)}")
        
        if story_arc.get('tension_points'):
            tension_count = len(story_arc['tension_points'])
            conflict_parts.append(f"The story maintains tension through {tension_count} key dramatic moments")
        
        return " ".join(conflict_parts)
    
    def _create_thematic_analysis(self, narrative_elements: Dict) -> str:
        """Create thematic analysis."""
        thematic_parts = []
        
        if narrative_elements.get('dialogue_themes'):
            themes = narrative_elements['dialogue_themes'][:3]
            thematic_parts.append(f"The story explores themes of {', '.join(set(themes))}")
        
        if narrative_elements.get('emotional_journey'):
            emotional_range = narrative_elements['emotional_journey'][:2]
            thematic_parts.append(f"The emotional journey of the characters evolves through various states: {'; '.join(emotional_range)}")
        
        return " ".join(thematic_parts)
    
    def _create_detailed_resolution(self, narrative_elements: Dict, story_arc: Dict) -> str:
        """Create detailed resolution summary."""
        resolution_parts = []
        
        if story_arc.get('climax_scene', -1) >= 0:
            resolution_parts.append("The story builds to a climactic moment that resolves the central tensions")
        
        if narrative_elements.get('conflicts'):
            resolution_parts.append("bringing closure to the various conflicts that have driven the narrative")
        
        if narrative_elements.get('character_arcs'):
            resolution_parts.append("and providing satisfying conclusions to the character development arcs")
        
        return " ".join(resolution_parts)
    
    def _create_comprehensive_character_analysis(self, narrative_elements: Dict) -> str:
        """Create comprehensive character analysis."""
        if not narrative_elements.get('main_characters'):
            return ""
        
        characters = narrative_elements['main_characters']
        return f"The narrative provides deep exploration of the protagonists' inner lives, motivations, and relationships, particularly focusing on {', '.join(characters[:2])} and their complex interactions throughout the story"
    
    def _create_conflict_resolution_analysis(self, narrative_elements: Dict, story_arc: Dict) -> str:
        """Create conflict resolution analysis."""
        return "The story examines how conflicts arise from character motivations and circumstances, and how these tensions are ultimately resolved through character growth and decisive action"
    
    def _create_symbolic_thematic_analysis(self, narrative_elements: Dict) -> str:
        """Create symbolic and thematic analysis."""
        return "The narrative incorporates symbolic elements and explores universal themes that resonate with the human experience, adding depth and meaning to the story"
    
    def _create_psychological_analysis(self, narrative_elements: Dict) -> str:
        """Create psychological analysis."""
        return "The story delves into the psychological dimensions of its characters, examining their fears, desires, and the internal conflicts that drive their actions"
    
    def _expand_to_target_length(self, text: str, max_sentences: int) -> str:
        """Expand text to target length with additional detail."""
        sentences = sent_tokenize(text)
        if len(sentences) >= max_sentences:
            return text
        
        # Add more detail to reach target length
        current_length = len(sentences)
        needed = max_sentences - current_length
        
        # Add transitional sentences and additional detail
        enhanced_sentences = sentences[:]
        
        for i in range(needed):
            if i < len(sentences):
                # Add detail to existing sentences
                enhanced_sentences.append(f"This development further complicates the narrative and adds depth to the character relationships.")
        
        return " ".join(enhanced_sentences[:max_sentences])
    
    def _ensure_comprehensive_coverage(self, summary: str, max_sentences: int) -> str:
        """Ensure comprehensive coverage of all story elements."""
        sentences = sent_tokenize(summary)
        if len(sentences) <= max_sentences:
            return summary
        
        return " ".join(sentences[:max_sentences])
    
    def _add_sophisticated_transition(self, sentence: str, index: int, all_sentences: List[str]) -> str:
        """Add sophisticated transitions between sentences."""
        transitions = [
            "Furthermore,", "Additionally,", "Moreover,", "Meanwhile,", "However,", 
            "Consequently,", "Subsequently,", "In contrast,", "On the other hand,", 
            "As the story progresses,", "Building upon this,", "This development leads to,"
        ]
        
        if index < len(transitions):
            return f"{transitions[index % len(transitions)]} {sentence.lower()}"
        
        return sentence
    
    def _identify_themes_in_text(self, text: str) -> List[str]:
        """Identify themes in general text."""
        themes = []
        text_lower = text.lower()
        
        theme_patterns = [
            ('love', 'love and relationships'),
            ('family', 'family dynamics'),
            ('work', 'professional life'),
            ('power', 'power and control'),
            ('justice', 'justice and morality'),
            ('truth', 'truth and deception'),
            ('freedom', 'freedom and constraint'),
            ('hope', 'hope and despair'),
            ('death', 'mortality and loss'),
            ('growth', 'personal growth')
        ]
        
        for pattern, theme in theme_patterns:
            if pattern in text_lower:
                themes.append(theme)
        
        return themes[:3]
    
    def _analyze_emotional_arc(self, text: str) -> List[str]:
        """Analyze emotional arc of the text."""
        emotions = []
        text_lower = text.lower()
        
        emotion_patterns = [
            ('happy', 'joy'),
            ('sad', 'sorrow'),
            ('angry', 'anger'),
            ('scared', 'fear'),
            ('excited', 'excitement'),
            ('confused', 'confusion'),
            ('hopeful', 'hope'),
            ('desperate', 'desperation'),
            ('peaceful', 'peace'),
            ('tense', 'tension')
        ]
        
        for pattern, emotion in emotion_patterns:
            if pattern in text_lower:
                emotions.append(emotion)
        
        if len(emotions) >= 2:
            return [emotions[0], emotions[-1]]
        
        return emotions
    
    def _postprocess_abstractive_summary(self, summary: str, target_sentences: int) -> str:
        """Post-process abstractive summary for better quality."""
        if not summary:
            return ""
        
        # Clean up the summary
        summary = summary.strip()
        
        # Ensure proper sentence endings
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Split into sentences and limit length
        sentences = sent_tokenize(summary)
        if len(sentences) > target_sentences:
            sentences = sentences[:target_sentences]
        
        # Ensure sentences are coherent
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Remove very short fragments
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                final_sentences.append(sentence)
        
        return ' '.join(final_sentences)
    
    def _combine_chunk_summaries(self, chunk_summaries: List[str]) -> str:
        """Fallback method to combine chunk summaries."""
        if not chunk_summaries:
            return ""
        
        # Simple combination with basic coherence
        combined = " ".join(chunk_summaries)
        
        # Clean up
        combined = re.sub(r'\s+', ' ', combined)
        sentences = sent_tokenize(combined)
        
        # Take first few sentences
        if len(sentences) > 5:
            sentences = sentences[:5]
        
        return " ".join(sentences)

    def _clean_script_text(self, text: str) -> str:
        """Remove excessive whitespace and non-content lines."""
        lines = []
        for line in text.split('\n'):
            l = line.strip()
            if not l:
                continue
            # De-emphasize purely formatting lines
            if len(l) <= 2:
                continue
            lines.append(l)
        return "\n".join(lines)

    def _split_into_scenes(self, text: str) -> List[str]:
        """Split by scene headings like INT./EXT. and return scene blocks."""
        blocks: List[str] = []
        current: List[str] = []
        scene_heading = re.compile(r'^(INT\.|EXT\.|INT/EXT\.)', re.IGNORECASE)
        for line in text.split('\n'):
            if scene_heading.match(line):
                if current:
                    blocks.append("\n".join(current))
                    current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        # If too many tiny scenes, merge into larger chunks ~1500-2500 words
        merged: List[str] = []
        buf = []
        word_count = 0
        for b in blocks:
            wc = len(b.split())
            if word_count + wc > 2200 and buf:
                merged.append("\n".join(buf))
                buf = [b]
                word_count = wc
            else:
                buf.append(b)
                word_count += wc
        if buf:
            merged.append("\n".join(buf))
        return merged or blocks

    def _extract_plot_points(self, scenes: List[str]) -> List[str]:
        """Extract key plot points and story beats from scenes."""
        plot_points = []
        
        for i, scene in enumerate(scenes):
            # Look for key story elements
            if self._contains_conflict(scene):
                plot_points.append(f"Scene {i+1}: Conflict/Tension")
            elif self._contains_revelation(scene):
                plot_points.append(f"Scene {i+1}: Revelation/Discovery")
            elif self._contains_climax(scene):
                plot_points.append(f"Scene {i+1}: Climax/Major Event")
            elif self._contains_character_development(scene):
                plot_points.append(f"Scene {i+1}: Character Development")
        
        return plot_points
    
    def _calculate_scene_importance(self, scene_text: str, scene_index: int, total_scenes: int) -> float:
        """Calculate importance score for a scene based on multiple factors."""
        importance = 0.0
        
        # Position-based importance (beginning, middle, end)
        position_ratio = scene_index / max(total_scenes - 1, 1)
        if position_ratio < 0.2:  # Beginning
            importance += 0.3
        elif position_ratio > 0.8:  # End
            importance += 0.4
        else:  # Middle
            importance += 0.2
        
        # Content-based importance
        if self._contains_dialogue(scene_text):
            importance += 0.2
        if self._contains_action(scene_text):
            importance += 0.2
        if self._contains_character_introduction(scene_text):
            importance += 0.3
        if self._contains_conflict(scene_text):
            importance += 0.4
        
        # Length-based importance (longer scenes often more important)
        word_count = len(scene_text.split())
        if word_count > 200:
            importance += 0.1
        elif word_count < 50:
            importance -= 0.1
        
        return min(max(importance, 0.1), 1.0)
    
    def _summarize_scene_enhanced(self, scene_text: str, importance: float) -> str:
        """Enhanced scene summarization with importance weighting."""
        # Extract dialogue and action separately
        dialogue_lines = self._extract_dialogue(scene_text)
        action_lines = self._extract_action(scene_text)
        
        # Prioritize dialogue for character development
        if dialogue_lines:
            dialogue_summary = self._summarize_dialogue(dialogue_lines)
            if action_lines and importance > 0.5:
                action_summary = self._summarize_action(action_lines)
                return f"{action_summary} {dialogue_summary}"
            return dialogue_summary
        
        # Fallback to action summarization
        if action_lines:
            return self._summarize_action(action_lines)
        
        # Last resort: general text summarization
        return self._summarize_textrank(scene_text, max_sentences=2)
    
    def _combine_scene_summaries(self, scene_summaries: List[str], plot_points: List[str], max_sentences: int) -> str:
        """Intelligently combine scene summaries into coherent story."""
        if not scene_summaries:
            return "Unable to generate summary from script content."
        
        # If we have few scenes, combine directly
        if len(scene_summaries) <= 3:
            return " ".join(scene_summaries)
        
        # Create story flow with transitions
        combined = []
        for i, summary in enumerate(scene_summaries):
            if i == 0:
                combined.append(f"Beginning: {summary}")
            elif i == len(scene_summaries) - 1:
                combined.append(f"Finally: {summary}")
            else:
                combined.append(summary)
        
        # Use enhanced summarization to condense
        full_text = " ".join(combined)
        return self._summarize_enhanced(full_text, max_sentences=max_sentences)
    
    def _polish_summary(self, summary: str, max_sentences: int) -> str:
        """Polish and ensure coherence of final summary."""
        if not summary or len(summary.strip()) < 20:
            return "Summary could not be generated from script content."
        
        # Ensure proper sentence structure
        sentences = sent_tokenize(summary)
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Clean up sentences
        polished_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Remove very short fragments
                # Ensure sentence ends properly
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                polished_sentences.append(sentence)
        
        return " ".join(polished_sentences)
    
    def _summarize_enhanced(self, text: str, max_sentences: int = 7) -> str:
        """Enhanced summarization with multiple techniques."""
        # Try LexRank first
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            sentences = summarizer(parser.document, max_sentences)
            summary = " ".join(str(s) for s in sentences if str(s).strip())
            if summary and len(summary) > 50:
                return summary
        except Exception:
            pass
        
        # Enhanced frequency-based scoring with TF-IDF-like weighting
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Calculate word importance scores
        word_scores = self._calculate_word_importance(text)
        
        # Score sentences based on important words and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            
            # Word importance score
            word_score = sum(word_scores.get(w, 0) for w in words)
            
            # Position bonus (beginning and end sentences are often important)
            position_bonus = 1.0
            if i < len(sentences) * 0.1 or i > len(sentences) * 0.9:
                position_bonus = 1.2
            
            # Length penalty (avoid very short or very long sentences)
            length_factor = min(max(len(sentence) / 100, 0.5), 1.5)
            
            final_score = word_score * position_bonus * length_factor
            scored_sentences.append((sentence, final_score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        # Maintain original order
        final_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                final_sentences.append(sentence)
        
        return " ".join(final_sentences)
    
    def _calculate_word_importance(self, text: str) -> Dict[str, float]:
        """Calculate importance scores for words using TF-IDF-like approach."""
        sentences = sent_tokenize(text)
        word_doc_freq = Counter()
        word_total_freq = Counter()
        
        for sentence in sentences:
            words = set(word_tokenize(sentence.lower()))
            words = {w for w in words if w.isalpha() and w not in self.stop_words}
            word_doc_freq.update(words)
            
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [w for w in sentence_words if w.isalpha() and w not in self.stop_words]
            word_total_freq.update(sentence_words)
        
        # Calculate TF-IDF-like scores
        total_sentences = len(sentences)
        word_scores = {}
        
        for word in word_total_freq:
            tf = word_total_freq[word]
            idf = total_sentences / max(word_doc_freq[word], 1)
            word_scores[word] = tf * idf
        
        return word_scores
    
    # Helper methods for scene analysis
    def _contains_dialogue(self, text: str) -> bool:
        """Check if scene contains dialogue."""
        return bool(re.search(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', text, re.MULTILINE))
    
    def _contains_action(self, text: str) -> bool:
        """Check if scene contains action descriptions."""
        action_indicators = ['runs', 'walks', 'enters', 'leaves', 'grabs', 'throws', 'falls', 'stands']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in action_indicators)
    
    def _contains_conflict(self, text: str) -> bool:
        """Check if scene contains conflict or tension."""
        conflict_words = ['argue', 'fight', 'disagree', 'conflict', 'tension', 'angry', 'frustrated', 'attack']
        text_lower = text.lower()
        return any(word in text_lower for word in conflict_words)
    
    def _contains_revelation(self, text: str) -> bool:
        """Check if scene contains revelations or discoveries."""
        revelation_words = ['discovers', 'reveals', 'learns', 'finds out', 'realizes', 'understands', 'sees']
        text_lower = text.lower()
        return any(word in text_lower for word in revelation_words)
    
    def _contains_climax(self, text: str) -> bool:
        """Check if scene contains climax or major events."""
        climax_words = ['suddenly', 'explosion', 'crash', 'finally', 'ultimate', 'final', 'decisive']
        text_lower = text.lower()
        return any(word in text_lower for word in climax_words)
    
    def _contains_character_development(self, text: str) -> bool:
        """Check if scene contains character development."""
        dev_words = ['changes', 'grows', 'learns', 'develops', 'becomes', 'transforms']
        text_lower = text.lower()
        return any(word in text_lower for word in dev_words)
    
    def _contains_character_introduction(self, text: str) -> bool:
        """Check if scene introduces new characters."""
        intro_words = ['meets', 'introduces', 'encounters', 'new', 'first time']
        text_lower = text.lower()
        return any(word in text_lower for word in intro_words)
    
    def _extract_dialogue(self, text: str) -> List[str]:
        """Extract dialogue lines from scene."""
        dialogue_lines = []
        lines = text.split('\n')
        in_dialogue = False
        
        for line in lines:
            line = line.strip()
            if re.match(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', line):
                in_dialogue = True
                continue
            elif in_dialogue and line and not line.startswith(('INT.', 'EXT.', '(')):
                dialogue_lines.append(line)
                in_dialogue = False
        
        return dialogue_lines
    
    def _extract_action(self, text: str) -> List[str]:
        """Extract action description lines from scene."""
        action_lines = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if (line and 
                not re.match(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', line) and
                not line.startswith(('INT.', 'EXT.')) and
                not line.startswith('(')):
                action_lines.append(line)
        
        return action_lines
    
    def _summarize_dialogue(self, dialogue_lines: List[str]) -> str:
        """Summarize dialogue content."""
        if not dialogue_lines:
            return ""
        
        # Join dialogue and summarize
        dialogue_text = " ".join(dialogue_lines)
        return self._summarize_textrank(dialogue_text, max_sentences=1)
    
    def _summarize_action(self, action_lines: List[str]) -> str:
        """Summarize action descriptions."""
        if not action_lines:
            return ""
        
        # Join action and summarize
        action_text = " ".join(action_lines)
        return self._summarize_textrank(action_text, max_sentences=1)

    def _summarize_textrank(self, text: str, max_sentences: int = 7) -> str:
        """Summarize text using LexRank (TextRank variant) from sumy.
        Falls back to frequency-based summarization if needed.
        """
        try:
            # Sumy's tokenizer expects a language code; using 'english'
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            sentences = summarizer(parser.document, max_sentences)
            summary = " ".join(str(s) for s in sentences if str(s).strip())
            if summary:
                return summary
        except Exception:
            pass

        # Fallback: frequency-based scoring
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        word_freq = Counter()
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            word_freq.update(words)
        scored = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            score = sum(word_freq.get(w, 0) for w in words)
            scored.append((sentence, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scored[:max_sentences]]
        return " ".join(top)
    
    def predict_genre(self, script_text: str) -> str:
        """
        Predict genre of the entire script
        
        Args:
            script_text: Raw script text
            
        Returns:
            Predicted genre
        """
        # Extract key text for genre prediction
        # Focus on dialogue and action descriptions
        dialogue_lines = []
        action_lines = []
        
        lines = script_text.split('\n')
        in_dialogue = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if re.match(r'^[A-Z][A-Z\s]+?(?:\s*\([^)]*\))?\s*$', line):
                in_dialogue = True
                continue
            elif line.startswith(('INT.', 'EXT.')):
                in_dialogue = False
                action_lines.append(line)
            elif in_dialogue:
                dialogue_lines.append(line)
            else:
                action_lines.append(line)
        
        # Combine for genre prediction
        text_for_genre = ' '.join(dialogue_lines + action_lines)
        
        # Truncate if too long
        if len(text_for_genre) > 1000:
            text_for_genre = text_for_genre[:1000]
        
        # Use custom model if available, otherwise fallback
        if self.genre_classifier and 'model' in self.genre_classifier:
            # Use custom trained model
            return self._predict_with_custom_model(text_for_genre)
        else:
            # Fallback to simple keyword-based classification
            return self._keyword_genre_classification(text_for_genre)
    
    def _keyword_genre_classification(self, text: str) -> str:
        """Simple keyword-based genre classification"""
        text_lower = text.lower()
        
        genre_keywords = {
            'action': ['fight', 'chase', 'explosion', 'gun', 'battle', 'run', 'escape'],
            'comedy': ['laugh', 'funny', 'joke', 'hilarious', 'comedy', 'humor'],
            'romance': ['love', 'kiss', 'romance', 'heart', 'relationship', 'marriage'],
            'horror': ['scary', 'monster', 'ghost', 'fear', 'terrifying', 'dark'],
            'thriller': ['mystery', 'detective', 'crime', 'suspense', 'investigation'],
            'sci-fi': ['space', 'robot', 'future', 'technology', 'alien', 'scientist'],
            'drama': ['family', 'life', 'death', 'emotion', 'serious', 'tragedy']
        }
        
        scores = {}
        for genre, keywords in genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[genre] = score
        
        if not any(scores.values()):
            return 'drama'  # Default
        
        return max(scores, key=scores.get)
    
    def _predict_with_custom_model(self, text: str) -> str:
        """Predict genre using custom trained model"""
        try:
            classifier = self.genre_classifier
            vocab = classifier['vocab']
            model = classifier['model']
            label_classes = classifier['label_classes']
            device = classifier['device']
            max_len = classifier['max_len']
            
            # Encode text
            input_ids, attention_mask = encode_text(text, vocab, max_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                genre_logits = outputs['genre']
                genre_idx = int(genre_logits.argmax(dim=-1).item())
                return label_classes['genre'][genre_idx]
        except Exception as e:
            print(f"Error in custom model prediction: {e}")
            return self._keyword_genre_classification(text)
    
    def analyze_script(self, file_path: str, abstractive_summary: bool = False, max_summary_sentences: int = 7, **_: object) -> Dict:
        """
        Complete script analysis
        
        Args:
            file_path: Path to script file
            
        Returns:
            Dictionary with all analysis results
        """
        # Load script
        script_text = self.load_script(file_path)
        
        # Extract components
        characters = self.extract_characters(script_text)
        locations = self.extract_locations(script_text)
        summary = self.generate_summary(script_text, max_sentences=max_summary_sentences, abstractive=abstractive_summary)
        genre = self.predict_genre(script_text)
        
        return {
            'summary': summary,
            'characters': {
                'count': len(characters),
                'list': characters
            },
            'locations': {
                'count': len(locations),
                'list': locations
            },
            'genre': genre,
            'total_pages': len(script_text.split('\n')) // 50  # Rough estimate
        }


def main():
    """Example usage"""
    analyzer = ScriptAnalyzer()
    
    # Example with sample data
    sample_script = """
    INT. OFFICE - DAY
    
    JOHN sits at his desk, looking frustrated.
    
    JOHN
    I can't believe this is happening!
    
    MARY enters the room.
    
    MARY
    (concerned)
    What's wrong, John?
    
    JOHN
    Everything is going wrong with this project.
    
    EXT. STREET - NIGHT
    
    A car chase ensues through the city streets.
    """
    
    # Save sample to file for testing
    with open('sample_script.txt', 'w') as f:
        f.write(sample_script)
    
    # Analyze
    results = analyzer.analyze_script('sample_script.txt')
    
    print("Script Analysis Results:")
    print(f"Summary: {results['summary']}")
    print(f"Characters ({results['characters']['count']}):")
    for char, dialogue, total in results['characters']['list']:
        print(f"  - {char}: {dialogue} dialogues, {total} total mentions")
    print(f"Locations ({results['locations']['count']}):")
    for loc, count in results['locations']['list']:
        print(f"  - {loc}: {count} mentions")
    print(f"Predicted Genre: {results['genre']}")


if __name__ == "__main__":
    main()
