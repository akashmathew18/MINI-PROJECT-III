#!/usr/bin/env python3
"""
Debug test to understand why the summarization is not working as expected.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from script_analyzer import ScriptAnalyzer

def debug_summarization():
    """Debug the summarization process."""
    
    # Simple test script
    test_script = """
    INT. OFFICE - DAY
    
    SARAH sits at her desk, looking frustrated.
    
    SARAH
    I can't believe this happened. The client is going to be furious.
    
    MICHAEL enters with coffee.
    
    MICHAEL
    Morning, Sarah. What's wrong?
    
    SARAH
    The quarterly reports are all wrong. We need to fix this immediately.
    
    MICHAEL
    Let me help you. We'll get this done together.
    
    EXT. STREET - NIGHT
    
    SARAH walks to her car in the rain, determined to solve the problem.
    """
    
    print("🔍 Debugging Summarization Process")
    print("=" * 50)
    
    analyzer = ScriptAnalyzer()
    
    # Clean the script text
    cleaned = analyzer._clean_script_text(test_script)
    print(f"Cleaned script length: {len(cleaned)} characters")
    print(f"Cleaned script preview: {cleaned[:200]}...")
    
    # Check scene splitting
    scenes = analyzer._split_into_scenes(cleaned)
    print(f"\nNumber of scenes found: {len(scenes)}")
    for i, scene in enumerate(scenes):
        print(f"Scene {i+1} length: {len(scene)} chars")
        print(f"Scene {i+1} preview: {scene[:100]}...")
    
    # Test the full summarization
    print(f"\n🎬 Testing Full Summarization:")
    summary = analyzer.generate_summary(test_script, max_sentences=10, abstractive=False)
    print(f"Generated summary length: {len(summary)}")
    print(f"Summary content:\n{summary}")
    
    # Test character extraction
    print(f"\n👥 Character Extraction:")
    characters = analyzer.extract_characters(test_script)
    print(f"Characters found: {characters}")
    
    # Test location extraction
    print(f"\n🏢 Location Extraction:")
    locations = analyzer.extract_locations(test_script)
    print(f"Locations found: {locations}")

if __name__ == "__main__":
    debug_summarization()
