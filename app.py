import streamlit as st
import json
import os
from pathlib import Path
import glob

st.set_page_config(layout="wide")
st.title("Feature Visualizer")

# Get all feature files and sort by score (highest to lowest)
feature_files = sorted(
    glob.glob("autointerp_features/feature_*.json"),
    key=lambda x: json.load(open(x))["score"],
    reverse=True,
)

# Sidebar for feature selection
selected_feature = st.sidebar.selectbox(
    "Select Feature",
    feature_files,
    format_func=lambda x: f"Feature {Path(x).stem.split('_')[1]} (Score: {json.load(open(x))['score']})",
)

if selected_feature:
    feature_id = Path(selected_feature).stem.split("_")[1]

    # Load and display feature information
    with open(selected_feature) as f:
        feature_data = json.load(f)

    st.header(f"Feature {feature_id}: {feature_data['title']}")

    # Feature description and metadata
    with st.expander("Feature Description and Metadata", expanded=True):
        st.markdown("### Description")
        st.write(feature_data["description"])

        st.markdown("### Reasoning")
        st.write(feature_data["reasoning"])

        st.markdown("### Findings")
        st.write(feature_data["findings"])

        st.markdown("### Conclusion")
        st.write(feature_data["conclusion"])

        st.markdown("### Metadata")
        st.write(f"Activation Pattern: {feature_data['activation-pattern']}")
        st.write(f"Score: {feature_data['score']} [ran 5 trials, took the mean score]")

        # Add button to show feature description prompt
        if st.button("Show Feature Description Prompt"):
            st.info(
                """
                This is a visualization of a sparse autoencoder feature trained on DNA language model embeddings. 

                You will be shown several images of high-activation patterns of this feature across genomes, along with genome annotations. In each image, the first row shows the feature activation pattern across a 10kb region of a genome, and the rows below that show annotations of that genome. The annotations are labeled with numbers. In the message following the image, you get a legend matching each number to the corresponding annotation. You will be given 10 such images.

                Your task is to carefully analyze and interpret what genomic pattern or element this feature detects. Start with detailed observation of individual examples, then synthesize the broader pattern:

                1. For each example, examine:
                - Precise location of activation peaks relative to annotations
                - Strength and shape of the activation pattern
                - Specific genes or elements present
                - Genomic context (neighboring elements)

                2. Then analyze across examples:
                - Common elements or motifs that appear repeatedly
                - Consistent positioning patterns (e.g. upstream/downstream)
                - Biological relationships between co-occurring elements
                - Taxonomic or evolutionary patterns across organisms

                This feature description must be consistently evident across MOST examples to be considered valid. Random or inconsistent patterns should be labeled as such.

                Remember, the more often and the stronger a feature fires, the more important the annotation is at the location The feature could be specific to a gene family, a structural motif, a sequence motif, a regulatory element, etc.  These WILL be used to predict unseen DNA sequences by the feature so only highlight relevant factors for this. 

                Please provide the following structured analysis:
                - reasoning: [description of your analytical process]
                - findings: [details of discovered biological patterns, focusing on scientific significance]
                - conclusion: [short assessment of whether this feature captures a biologically meaningful pattern]
                - title: [concise descriptor, max 50 chars]
                - activation-pattern: [start/end/continuous/spike]
                - description: [detailed explanation of:
                  - What specific genomic element(s) the feature detects
                  - How the feature relates to those elements (position, context)
                  - What biological function this may represent
                  - How consistent the pattern is across examples
                  - If significant, what organism this feature tends to be active in]

                Only provide the structured output, no other text.
                """
            )

    # High Activating Samples
    st.header("High Activating Samples")

    # Get all sections from test_output_cached
    test_output_dir = f"test_output_cached/feature_{feature_id}"
    if os.path.exists(test_output_dir):
        sections = sorted(
            list(
                set(
                    [
                        f.split(".")[0].split("_")[1]
                        for f in os.listdir(test_output_dir)
                        if f.endswith(".png") or f.endswith(".txt")
                    ]
                )
            )
        )

        for section in sections:
            with st.expander(f"Section {section}", expanded=False):
                # Display image
                img_path = os.path.join(test_output_dir, f"section_{section}.png")
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

                # Display text
                txt_path = os.path.join(test_output_dir, f"section_{section}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path) as f:
                        st.text_area("Annotations", f.read(), height=300)

    # Scoring Information
    st.header("Scoring Information")

    # Explain scoring function
    st.markdown("### Scoring Explanation")
    st.write(
        "The score for each feature is determined by expert evaluation. Experts are presented with several images of genomic regions, including activations of sparse autoencoder features and annotations. They are tasked with identifying which image (if any) matches the provided feature description."
    )

    # Display prompt in a pop-up
    if st.button("Show Expert Prompt"):
        st.info(
            """
        You are an expert detector of sparse autoencoder features. 
        I will show you several images of genomic regions with annotations and the activations of sparse autoencoder features. One of these images shows the true feature that matches the description, while the others are distractors that have similar activation patterns but represent different biological functions.

        Your task is to determine which image (if any) matches the provided feature description.

        Important considerations:
        - The feature description should match both the activation pattern and the biological context
        - Pay close attention to the relationship between activations and genome annotations
        - Consider both where the feature activates and where it does not activate
        - The match should be precise - similar patterns are not enough if they represent different biological functions

        Here is the feature description:
        __FEATURE_DESCRIPTION__

        You must ONLY output the matching image number (0-4) or -1 if no images match the description:
        """
        )

    scored_features_dir = f"scored_features/feature_{feature_id}"
    if os.path.exists(scored_features_dir):
        trials = sorted(
            [d for d in os.listdir(scored_features_dir) if d.startswith("trial_")]
        )

        for trial in trials:
            with st.expander(f"Trial {trial.split('_')[1]}", expanded=False):
                trial_dir = os.path.join(scored_features_dir, trial)
                sections = sorted(
                    list(
                        set(
                            [
                                f.split(".")[0].split("_")[1]
                                for f in os.listdir(trial_dir)
                                if f.endswith(".png") or f.endswith(".txt")
                            ]
                        )
                    )
                )

                for section in sections:
                    st.subheader(f"Section {section}")

                    # Display image
                    img_path = os.path.join(trial_dir, f"section_{section}.png")
                    if os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)

                    # Display text
                    txt_path = os.path.join(trial_dir, f"section_{section}.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path) as f:
                            st.text_area("Annotations", f.read(), height=200)
