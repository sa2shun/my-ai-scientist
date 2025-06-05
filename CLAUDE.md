# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Scientist v2 is an autonomous scientific research system that generates research hypotheses, conducts experiments through agentic tree search, analyzes results, and writes scientific papers without relying on human-authored templates.

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python dependencies
pip install -r requirements.txt

# For Claude via AWS Bedrock
pip install anthropic[bedrock]
```

### Research Ideation
```bash
# Generate research ideas from a topic description
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "ai_scientist/ideas/topic.md" \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 20 \
  --num-reflections 5
```

### Run Full Experiment Pipeline
```bash
# Run experiments with tree search
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/topic.json" \
  --load_code \
  --add_dataset_ref \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20 \
  --model_review gpt-4o-2024-11-20 \
  --model_agg_plots o3-mini-2025-01-31 \
  --num_cite_rounds 20
```

### Running Individual Components
```bash
# Just run experiments (no writeup)
python launch_scientist_bfts.py --load_ideas "path/to/ideas.json" --no-writeup

# Generate 4-page ICBINB paper instead of 8-page
python launch_scientist_bfts.py --load_ideas "path/to/ideas.json" --icbinb

# Skip review phase
python launch_scientist_bfts.py --load_ideas "path/to/ideas.json" --no-review
```

## Architecture Overview

### Core Modules

1. **Ideation Module** (`perform_ideation_temp_free.py`)
   - Generates research ideas from topic descriptions
   - Performs novelty checking via Semantic Scholar API
   - Outputs structured JSON with research proposals

2. **Tree Search Engine** (`treesearch/`)
   - Implements Best-First Tree Search (BFTS) for experiment exploration
   - Manages parallel agents exploring different experimental paths
   - Key components:
     - `agent_manager.py`: Coordinates multiple research agents
     - `parallel_agent.py`: Individual agent implementation
     - `bfts_utils.py`: Tree search algorithms and utilities
     - `journal.py`: Tracks experiment progress and results

3. **Writing System** (`perform_writeup.py`, `perform_icbinb_writeup.py`)
   - Converts experimental results into LaTeX papers
   - Handles citation gathering and integration
   - Supports both 8-page and 4-page formats

4. **Review System** (`perform_llm_review.py`, `perform_vlm_review.py`)
   - Evaluates generated papers using LLMs
   - Reviews both text content and visual elements

5. **Visualization** (`perform_plotting.py`)
   - Aggregates experimental results into meaningful plots
   - Uses LLMs to create appropriate visualizations

### Key Configuration

The `bfts_config.yaml` file controls tree search behavior:
- `num_workers`: Number of parallel agents
- `num_steps`: Maximum tree search steps
- `debug_mode`: Enable detailed logging
- `model_config`: Different models for various stages

### LLM Integration

The system supports multiple LLM providers through a unified interface:
- `llm.py`: Main LLM interface supporting OpenAI, Anthropic, Google, DeepSeek
- `treesearch/backend/`: Provider-specific implementations
- Token tracking for cost monitoring in `utils/token_tracker.py`

### Safety Considerations

**IMPORTANT**: This system executes LLM-generated code. Always run in a sandboxed environment:
- Use Docker containers for isolation
- Set appropriate resource limits
- Monitor process execution
- The system includes automatic process cleanup, but manual verification is recommended

## Development Tips

1. **Testing Changes**: When modifying the tree search algorithm, use `--debug` flag for detailed logging
2. **Model Selection**: Different models excel at different tasks - o1/o3 for complex reasoning, GPT-4o for general tasks
3. **Cost Management**: Monitor token usage - experiments can cost $15-20+ depending on model choice
4. **Output Location**: Results are saved in `experiments/{idea_name}_{timestamp}/`
5. **Tree Visualization**: Open `unified_tree_viz.html` to explore the experiment tree interactively

## API Keys Required

Set these environment variables:
- `OPENAI_API_KEY`: For OpenAI models
- `GEMINI_API_KEY`: For Gemini models  
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: For Claude via Bedrock
- `S2_API_KEY`: For Semantic Scholar API (optional but recommended)