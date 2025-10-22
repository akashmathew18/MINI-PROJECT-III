#!/usr/bin/env python3
"""
Test script for the enhanced script summarization functionality.
This demonstrates the improved summarization capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from script_analyzer import ScriptAnalyzer

def test_enhanced_summarization():
    """Test the enhanced summarization functionality."""
    
    # Sample script with a more complex story for testing
    sample_script = """
    INT. CORPORATE OFFICE - DAY
    
    SARAH sits at her desk, reviewing documents. The office buzzes with activity around her. She looks frustrated and stressed.
    
    SARAH
    (frustrated, to herself)
    These numbers don't add up! Someone made a serious mistake here. The client is going to be furious.
    
    MICHAEL enters, carrying two cups of coffee. He notices Sarah's distress.
    
    MICHAEL
    Morning, Sarah. Rough start to the day?
    
    SARAH
    (worried)
    The quarterly reports are all wrong. We're going to have to recalculate everything, and we might miss our deadline.
    
    MICHAEL
    (concerned, sitting down)
    How bad is it? Can I help?
    
    SARAH
    (grateful)
    It's pretty bad. I need to call the client and explain the delay. But first, I need to figure out what went wrong.
    
    MICHAEL
    (determined)
    Let's work on this together. We'll get it done.
    
    EXT. CITY STREET - NIGHT
    
    Rain pours down as SARAH walks quickly toward her car. Her phone rings urgently.
    
    SARAH
    (into phone, stressed)
    Yes, I understand the urgency. I'll have the corrected reports by tomorrow morning. I promise.
    
    She gets into her car and drives away into the stormy night.
    
    INT. SARAH'S APARTMENT - LATE NIGHT
    
    SARAH sits at her laptop, working frantically. Coffee cups and papers surround her. She looks exhausted but determined.
    
    SARAH
    (to herself, tired but focused)
    This has to be perfect. The client is counting on us, and I can't let them down.
    
    Her phone buzzes with a text message. She reads it and smiles for the first time that day.
    
    SARAH
    (reading message, relieved)
    "Found the error in the data source. Sending corrected files now. We've got this!" - Michael
    
    SARAH
    (to herself, grateful)
    Thank you, Michael. I couldn't have done this without your help.
    
    She continues working, more confident now. The storm outside begins to clear.
    
    INT. CORPORATE OFFICE - NEXT MORNING
    
    SARAH arrives at the office, looking much more relaxed. MICHAEL is already there.
    
    MICHAEL
    (smiling)
    Good morning! Did you finish the reports?
    
    SARAH
    (proudly)
    Yes! Thanks to your help, the client was thrilled with the results. They actually increased our contract.
    
    MICHAEL
    (happily)
    That's fantastic! Teamwork really does make the dream work.
    
    SARAH
    (grateful)
    I couldn't have done it without you. Thank you for believing in me when I was ready to give up.
    
    They share a warm smile as the morning sun streams through the office windows.
    """
    
    print("🎬 Testing Enhanced Script Summarization")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ScriptAnalyzer()
    
    # Test comprehensive detailed summarization
    print("\n📝 Comprehensive Detailed Summarization (25+ sentences, 1000+ words):")
    comprehensive_summary = analyzer.generate_summary(sample_script, max_sentences=30, abstractive=False)
    print(f"\n📖 COMPREHENSIVE STORY SUMMARY:")
    print(f"Length: {len(comprehensive_summary)} characters")
    print(f"Word count: {len(comprehensive_summary.split())}")
    print(f"Sentences: {len(comprehensive_summary.split('. '))}")
    print(f"\nContent:\n{comprehensive_summary}")
    
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS:")
    print(f"✅ Target Length: {'ACHIEVED' if len(comprehensive_summary) >= 1000 else 'NOT MET'} ({len(comprehensive_summary)}/1000+ characters)")
    print(f"✅ Sentence Count: {'ACHIEVED' if len(comprehensive_summary.split('. ')) >= 20 else 'NOT MET'} ({len(comprehensive_summary.split('. '))}/20+ sentences)")
    print(f"✅ Word Count: {'ACHIEVED' if len(comprehensive_summary.split()) >= 1000 else 'NOT MET'} ({len(comprehensive_summary.split())}/1000+ words)")
    
    # Check for story elements
    story_elements = ['character', 'story', 'plot', 'narrative', 'conflict', 'resolution']
    elements_found = [elem for elem in story_elements if elem in comprehensive_summary.lower()]
    print(f"✅ Story Elements: {len(elements_found)}/6 found: {elements_found}")
    
    # Test detailed narrative elements extraction
    print("\n🔍 Detailed Narrative Elements Analysis:")
    scenes = analyzer._split_into_scenes(sample_script)
    narrative_elements = analyzer._extract_detailed_narrative_elements(scenes)
    print(f"Main Characters: {narrative_elements.get('main_characters', [])}")
    print(f"Secondary Characters: {narrative_elements.get('secondary_characters', [])}")
    print(f"Key Conflicts: {narrative_elements.get('conflicts', [])}")
    print(f"Story Events: {narrative_elements.get('events', [])}")
    print(f"Locations: {narrative_elements.get('locations', [])}")
    print(f"Dialogue Themes: {narrative_elements.get('dialogue_themes', [])}")
    print(f"Plot Twists: {narrative_elements.get('plot_twists', [])}")
    print(f"Overall Tone: {narrative_elements.get('tone', 'neutral')}")
    
    # Test detailed story arc analysis
    print("\n📊 Detailed Story Arc Analysis:")
    story_arc = analyzer._analyze_detailed_story_arc(scenes)
    print(f"Setup scenes: {story_arc.get('setup', [])}")
    print(f"Inciting incident: {story_arc.get('inciting_incident', [])}")
    print(f"Rising action: {story_arc.get('rising_action', [])}")
    print(f"Climax scenes: {story_arc.get('climax', [])}")
    print(f"Falling action: {story_arc.get('falling_action', [])}")
    print(f"Resolution scenes: {story_arc.get('resolution', [])}")
    print(f"Turning points: {story_arc.get('turning_points', [])}")
    print(f"Climax scene: {story_arc.get('climax_scene', 'None')}")
    print(f"Tension points: {story_arc.get('tension_points', [])}")
    
    # Test enhanced scene analysis
    print("\n🔍 Scene Analysis:")
    scenes = analyzer._split_into_scenes(sample_script)
    print(f"Number of scenes detected: {len(scenes)}")
    
    for i, scene in enumerate(scenes):
        importance = analyzer._calculate_scene_importance(scene, i, len(scenes))
        print(f"Scene {i+1} importance score: {importance:.2f}")
    
    # Test plot point extraction
    print("\n📊 Plot Points Analysis:")
    plot_points = analyzer._extract_plot_points(scenes)
    for point in plot_points:
        print(f"- {point}")
    
    # Test character and location extraction
    print("\n👥 Character Analysis:")
    characters = analyzer.extract_characters(sample_script)
    print(f"Total characters: {len(characters)}")
    for char, dialogue, total in characters:
        print(f"- {char}: {dialogue} dialogues, {total} mentions")
    
    print("\n🏢 Location Analysis:")
    locations = analyzer.extract_locations(sample_script)
    print(f"Total locations: {len(locations)}")
    for loc, count in locations:
        print(f"- {loc}: {count} mentions")
    
    # Test genre prediction
    print("\n🎭 Genre Prediction:")
    genre = analyzer.predict_genre(sample_script)
    print(f"Predicted genre: {genre}")
    
    # Test complete analysis
    print("\n📋 Complete Script Analysis:")
    results = analyzer.analyze_script("test_script.txt")
    
    # Save sample script for testing
    with open("test_script.txt", "w") as f:
        f.write(sample_script)
    
    results = analyzer.analyze_script("test_script.txt")
    
    print(f"Final Summary: {results['summary']}")
    print(f"Characters: {results['characters']['count']}")
    print(f"Locations: {results['locations']['count']}")
    print(f"Genre: {results['genre']}")
    
    # Clean up
    if os.path.exists("test_script.txt"):
        os.remove("test_script.txt")
    
    print("\n✅ Enhanced summarization test completed!")

def test_advanced_features():
    """Test advanced summarization features."""
    
    print("\n🚀 Testing Advanced Features")
    print("=" * 50)
    
    analyzer = ScriptAnalyzer()
    
    # Test dialogue extraction
    test_scene = """
    INT. OFFICE - DAY
    
    JOHN sits at his desk.
    
    JOHN
    I can't believe this is happening!
    
    MARY enters.
    
    MARY
    What's wrong?
    
    JOHN
    Everything is going wrong with this project.
    """
    
    dialogue = analyzer._extract_dialogue(test_scene)
    action = analyzer._extract_action(test_scene)
    
    print(f"Dialogue lines: {len(dialogue)}")
    print(f"Action lines: {len(action)}")
    
    # Test scene importance calculation
    importance = analyzer._calculate_scene_importance(test_scene, 0, 1)
    print(f"Scene importance: {importance:.2f}")
    
    # Test content analysis
    print(f"Contains dialogue: {analyzer._contains_dialogue(test_scene)}")
    print(f"Contains action: {analyzer._contains_action(test_scene)}")
    print(f"Contains conflict: {analyzer._contains_conflict(test_scene)}")

if __name__ == "__main__":
    test_enhanced_summarization()
    test_advanced_features()
