"""Rich documentation data and articles covering Theta-IDE and the NeSyRL/BlendRL framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DocArticle:
    id: str
    title: str
    category: str
    summary: str
    html_content: str
    keywords: List[str]


ARTICLES: List[DocArticle] = [
    DocArticle(
        id="overview",
        title="Welcome & Architecture Overview",
        category="Getting Started",
        summary="High-level architecture of Theta-IDE, the NeSyRL framework, and BlendRL neurosymbolic RL.",
        keywords=["overview", "architecture", "intro", "welcome", "blendrl", "nesyrl", "reinforcement learning"],
        html_content="""
<h2>Welcome to Theta-IDE</h2>
<p><b>Theta-IDE</b> is a specialized development environment and experimentation platform for <b>Reinforcement Learning (RL)</b>, <b>Neural-Symbolic reasoning (BlendRL)</b>, and <b>modular algorithm benchmarking</b>.</p>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Core Philosophy:</b> Unify the entire research experimentation lifecycle into a single workflow — from visual Hydra configuration composition and real-time training telemetry, to Slurm cluster job orchestration, interactive visualization analysis, and an extensible Community Hub.
</div>

<h3>Framework Architecture</h3>
<ul>
  <li><b>Hydra Configuration Layer (<code>in/config/</code>):</b> Declarative, three-tier hierarchical configuration defining environments, agents, models, cluster resources, and experiment recipes.</li>
  <li><b>FastAPI Backend Daemon (<code>src/app/api/</code>):</b> Decoupled API service managing local training subprocesses, Slurm cluster submissions, and job queue states.</li>
  <li><b>PyTorch Lightning Runtime (<code>src/app/train.py</code>):</b> Standardized training driver providing automated checkpointing, device acceleration, and structured logging.</li>
  <li><b>Neurosymbolic Reasoners (BlendRL):</b> Joint neural and symbolic forward reasoning policies combining Neural Encoders (MLP, ResNet, Transformer) with First-Order Logic Reasoners (NSFR, Neumann).</li>
  <li><b>Extensible Plugin System:</b> Core plugins shipping natively with zero overhead when toggled off, plus community plugins installed from the Community Hub.</li>
</ul>

<h3>Key Workflows</h3>
<ol>
  <li><b>Configure:</b> Select or compose an experiment in the <i>Experiment Config</i> pane.</li>
  <li><b>Execute:</b> Train directly on your local machine or submit Slurm batch jobs to an HPC cluster.</li>
  <li><b>Monitor:</b> Observe real-time reward curves, minibatch losses, and metric tables with moving-average curve smoothing.</li>
  <li><b>Analyze:</b> Generate publication-ready convergence plots, loss breakdowns, and markdown comparative reports.</li>
</ol>
""",
    ),
    DocArticle(
        id="ui_panels",
        title="IDE Panels & Navigation Guide",
        category="Interface",
        summary="Detailed tour of all sidebar panels: Config, Monitor, Queue, Plots, Terminal, and Hub.",
        keywords=["panels", "ui", "navigation", "sidebar", "tabs", "interface", "layout"],
        html_content="""
<h2>IDE Panels & Navigation</h2>
<p>Theta-IDE organizes its interface into vertical sidebar tabs. Each tab represents a specialized workstation for your experiments.</p>

<h3>1. Experiment Config (<code>config</code>)</h3>
<p>The primary workspace for configuring and launching experiments:</p>
<ul>
  <li><b>Config Tree:</b> Hierarchical browser mirroring <code>in/config/experiment/</code>. Browse by group (e.g. <i>cartpole</i>, <i>mimic</i>, <i>quick_tests</i>).</li>
  <li><b>Boxed Config Viewer:</b> Visual, form-driven editor for hyperparameters, paradigms, learning rates, epochs, and Slurm resource allocations.</li>
  <li><b>Hydra YAML Preview:</b> Real-time preview of the fully resolved Hydra configuration with syntax highlighting and validation errors.</li>
  <li><b>Action Bar:</b> Launch Training (<code>F5</code>), Add to Batch Queue (<code>Ctrl+Shift+Q</code>), Save YAML (<code>Ctrl+S</code>), Duplicate Config, and Export Recipe.</li>
</ul>

<h3>2. Training Monitor (<code>monitor</code>)</h3>
<p>Real-time telemetry and telemetry replay for live and finished training runs:</p>
<ul>
  <li><b>Metric Cards:</b> Key scalars including Episode Reward, Minibatch Loss, and Timestep Budget.</li>
  <li><b>Dynamic Curves:</b> Live matplotlib charts tracking reward and loss trajectories.</li>
  <li><b>Smoothing Slider:</b> Exponential Moving Average (EMA) smoothing from 0% (raw points) up to 95% (smoothed trends).</li>
  <li><b>Multi-Run Navigation:</b> Previous/Next buttons and dropdown selector to compare against previous runs in history.</li>
  <li><b>Pin as Baseline:</b> Pin any historic run as a dashed reference curve overlay on top of active runs.</li>
  <li><b>Jump to Live:</b> Instant one-click camera focus back to the actively training experiment.</li>
</ul>

<h3>3. Job Queue (<code>queue</code>)</h3>
<p>Batch management for multi-stage pipelines and cluster tasks:</p>
<ul>
  <li>Tracks <b>Active</b>, <b>Queued</b>, and <b>Finished</b> runs.</li>
  <li>Displays job IDs, backend status, timestamps, and resource consumption.</li>
  <li>Live log inspector: view stdout and stderr logs directly in the IDE.</li>
  <li>Actions: Kill running jobs, Resubmit failed runs, or Clear finished entries.</li>
</ul>

<h3>4. Plots & Visualizations (<code>plots</code>)</h3>
<p>Inspect output plots auto-generated at the end of training pipelines:</p>
<ul>
  <li><b>Convergence Curves:</b> Visualizes mean ± SEM evaluation rewards across seeds and algorithms.</li>
  <li><b>Loss Decomposition:</b> Explores actor loss, critic/Q-loss, policy entropy, and Bellman error over transitions.</li>
  <li><b>Report Generator:</b> Markdown comparison tables showing best metrics and hyperparameter configurations.</li>
</ul>

<h3>5. Embedded Terminal (<code>terminal</code>)</h3>
<p>Built-in interactive terminal powered by xterm.js:</p>
<ul>
  <li>Full shell access with support for zsh, bash, and tmux.</li>
  <li>Terminal precedence mode: allows tmux prefix keys and terminal hotkeys to pass through uninterrupted.</li>
</ul>

<h3>6. Community Hub (<code>components</code>)</h3>
<p>Browse, install, update, and remove modular components:</p>
<ul>
  <li><b>Plugins:</b> Community-contributed tools, visualizers, and extensions.</li>
  <li><b>Methods:</b> Reinforcement learning algorithms (CQL, IQL, PPO, CEW).</li>
  <li><b>Models:</b> Neural and symbolic policy architectures.</li>
  <li><b>Environments:</b> Domain environments and reward wrappers.</li>
</ul>
""",
    ),
    DocArticle(
        id="learning_paradigms",
        title="Learning Paradigms & Constraints",
        category="Machine Learning",
        summary="Explanation of online RL, offline RL, and supervised learning paradigms supported in BlendRL.",
        keywords=["paradigms", "online", "offline", "supervised", "ppo", "cql", "iql", "dataset"],
        html_content="""
<h2>Learning Paradigms</h2>
<p>Theta-IDE strictly validates experiment compatibility using declared <b>paradigms</b>. Every experiment declares <code>paradigm: &lt;name&gt;</code>, which configures the training driver, callbacks, and validation rules.</p>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Paradigm</td>
    <td>Allowed Agents</td>
    <td>Evaluation Mechanism</td>
    <td>Key Constraints</td>
  </tr>
  <tr>
    <td><b><code>online_rl</code></b></td>
    <td>PPO, BlendRL (PPO)</td>
    <td>Fixed-episode live simulator rollouts via <code>EnvironmentEvaluatorCallback</code></td>
    <td><code>offline_only: false</code>; generates transition datasets</td>
  </tr>
  <tr>
    <td><b><code>offline_rl</code></b></td>
    <td>CQL, IQL, CEW, BlendRL (IQL/CQL)</td>
    <td>Replay buffer data module; validation loss and Bellman error via Lightning</td>
    <td><code>intervals_count: 1</code>, <code>eval_episodes: 0</code>; requires static transition dataset</td>
  </tr>
  <tr>
    <td><b><code>supervised</code></b></td>
    <td>Predictive models (DNN, ResNet, Transformer, ECM)</td>
    <td>Validation cross-entropy / MSE / AUROC on held-out splits</td>
    <td>Standalone RL agents forbidden</td>
  </tr>
</table>

<h3>Transition Dataset Schema</h3>
<p>When online agents run or offline datasets are loaded, transitions adhere to the standard schema:</p>
<pre><code>{
  "obs": np.ndarray,            # Primary neural observation vector or image
  "next_obs": np.ndarray,       # Subsequent neural state
  "logic_obs": np.ndarray,      # Symbolic ground facts for logic reasoner
  "next_logic_obs": np.ndarray, # Subsequent symbolic facts
  "action": int | np.ndarray,   # Selected action
  "reward": float,              # Scalar transition reward
  "done": bool                  # Episode termination flag
}</code></pre>
<p>Online generation automatically writes chunked <code>.pkl</code> archives accompanied by <code>dataset_manifest.json</code> containing git commit hash, random seed, transition count, and environment metadata.</p>
""",
    ),
    DocArticle(
        id="blendrl_hybrid",
        title="BlendRL Hybrid Neural-Symbolic Policy",
        category="Machine Learning",
        summary="How neural encoders and first-order logic reasoners are combined into a unified policy.",
        keywords=["blendrl", "symbolic", "nsfr", "neumann", "neural", "prolog", "logic", "hybrid"],
        html_content="""
<h2>BlendRL: Neural-Symbolic Hybrid Architecture</h2>
<p><b>BlendRL</b> bridges deep reinforcement learning with first-order symbolic logic reasoners (NSFR and Neumann) to achieve high sample efficiency, explainability, and verifiable safety constraints.</p>

<div style="background-color: rgba(184, 187, 38, 0.12); border-left: 4px solid #b8bb26; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>The Core Principle:</b> Both a neural network and a symbolic logic reasoner simultaneously process the incoming state. Their respective action probability distributions are combined through an adaptive, confidence-weighted blending module.
</div>

<h3>Constituent Modules</h3>
<ol>
  <li><b>Neural Encoder (<code>neural</code>):</b> Multi-Layer Perceptrons (MLP), Dueling ResNets, or Transformers that process high-dimensional raw observations (e.g. continuous vectors or image pixels).</li>
  <li><b>Symbolic Reasoner (<code>symbolic</code>):</b>
    <ul>
      <li><b>NSFR (Neural Symbolic Forward Reasoner):</b> Differentiable forward-chaining deduction engine operating on grounded facts and clauses.</li>
      <li><b>Neumann Reasoner:</b> Fast matrix-based forward reasoner designed for accelerated rule valuation.</li>
    </ul>
  </li>
  <li><b>The Blender (<code>blender</code>):</b> Combines the neural logits \\(\\pi_{neural}(a|s)\\) and symbolic valuation scores \\(\\pi_{logic}(a|s)\\):
    <pre><code>\\pi_{blended}(a|s) = (1 - \alpha) \\cdot \\pi_{neural}(a|s) + \alpha \\cdot \\pi_{logic}(a|s)</code></pre>
    where \\(\alpha\\) can be fixed, learned, or dynamically gated by symbolic confidence.
  </li>
</ol>

<h3>Rules & Domain Knowledge</h3>
<p>Domain knowledge is defined in human-readable logic rules stored in <code>in/rules/</code>. Logic predicates represent domain concepts (e.g., <code>pole_falling_left</code>, <code>cart_near_boundary</code>, <code>hypotensive_episode</code>).</p>
""",
    ),
    DocArticle(
        id="hotkeys",
        title="Keyboard Shortcuts & Leader Chords",
        category="Workflow & Tools",
        summary="Complete reference for tmux-style leader key navigation and quick action hotkeys.",
        keywords=["hotkeys", "shortcuts", "leader", "tmux", "keyboard", "navigation"],
        html_content="""
<h2>Keyboard Shortcuts & Leader Navigation</h2>
<p>Theta-IDE features a high-efficiency <b>Leader key system</b> inspired by tmux and Vim. You can navigate between any panel instantly without taking your hands off the keyboard.</p>

<h3>The Action Key (Leader)</h3>
<p>Default: <code>Ctrl+B</code> (Customizable in <i>Settings &rarr; Hotkeys</i> to <code>Caps Lock</code>, <code>Alt</code>, <code>Ctrl</code>, or <code>Meta</code>).</p>
<p>Supports two operational modes:</p>
<ul>
  <li><b>Leader (Modal):</b> Tap the Action key, release it, and press a digit within 1.5 seconds.</li>
  <li><b>Chorded:</b> Hold the Action key and tap a digit simultaneously.</li>
</ul>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Target Pane / Action</td>
  </tr>
  <tr>
    <td><code>Action + 0</code></td>
    <td>Settings & Preferences</td>
  </tr>
  <tr>
    <td><code>Action + 1</code></td>
    <td>Community Hub (Components)</td>
  </tr>
  <tr>
    <td><code>Action + 2</code></td>
    <td>Experiment Configuration (Editor)</td>
  </tr>
  <tr>
    <td><code>Action + 3</code></td>
    <td>Training Monitor & Curves</td>
  </tr>
  <tr>
    <td><code>Action + 4</code></td>
    <td>Results Browser</td>
  </tr>
  <tr>
    <td><code>Action + 5</code></td>
    <td>Plot & Visualization Viewer</td>
  </tr>
  <tr>
    <td><code>Action + 6</code></td>
    <td>TensorBoard Dashboard</td>
  </tr>
  <tr>
    <td><code>Action + 7</code></td>
    <td>Job Queue Manager</td>
  </tr>
  <tr>
    <td><code>Action + 8</code></td>
    <td>Interactive Terminal</td>
  </tr>
  <tr>
    <td><code>Action + 9</code></td>
    <td>System Console Log</td>
  </tr>
</table>

<h3>Global Action Hotkeys</h3>
<ul>
  <li><code>F5</code>: Launch training immediately using the currently active experiment configuration.</li>
  <li><code>Ctrl + Shift + Q</code>: Enqueue the loaded experiment to the background job queue.</li>
  <li><code>Ctrl + S</code>: Save active configuration changes to disk.</li>
  <li><code>Ctrl + R</code>: Reload component trees and refresh files from disk.</li>
</ul>
""",
    ),
    DocArticle(
        id="config_system",
        title="3-Tier Hierarchical Configuration",
        category="Getting Started",
        summary="How Hydra configuration files are structured across Tier 1 (Defaults), Tier 2 (Universal), and Tier 3 (Methods).",
        keywords=["config", "hydra", "tier", "yaml", "methods", "params", "hyperparameters"],
        html_content=r"""
<h2>3-Tier Hierarchical Configuration</h2>
<p>Configurations in Theta-IDE use a strict 3-tier hierarchy that eliminates parameter duplication while allowing granular per-method overrides.</p>

<h3>Tier 1: Defaults & Base Profiles</h3>
<p>Stored under <code>in/config/agent/&lt;algo&gt;.yaml</code> and <code>in/config/model/&lt;arch&gt;.yaml</code>. These define the baseline algorithm and model parameters (e.g. default batch size, discount factor \(\gamma\), network layer dimensions).</p>

<h3>Tier 2: Universal Experiment Parameters (<code>methods.params</code>)</h3>
<p>Universal scalars and hyperparameters applied across all methods in a single experiment:</p>
<pre><code>methods:
  params:
    epochs_per_interval: 25
    gamma: 0.99
    lr: 3e-4
    agent:
      cql:
        cql_alpha: 5.0
    model:
      dueling_resnet:
        hidden_dim: 128</code></pre>

<h3>Tier 3: Method-Level Declarations</h3>
<p>Specific methods declared for execution inherit from Tier 1 and Tier 2, specifying only their differences:</p>
<pre><code>methods:
  cql_baseline:
    agent: cql
    model: mlp
  cql_blendrl_hybrid:
    agent: cql
    model:
      blendrl:
        neural: dueling_resnet
        symbolic:
          nsfr:
            ruleset: cartpole_rules</code></pre>

<h3>Environment Keys</h3>
<p>Environment YAMLs (<code>in/config/env/*.yaml</code>) define operational metadata declaratively:</p>
<ul>
  <li><code>offline_only: true | false</code> &mdash; drives paradigm verification</li>
  <li><code>monitor_metric: "eval/reward" | "val/loss"</code> &mdash; target metric for checkpointing and tuning</li>
  <li><code>preprocess_on_load: true | false</code> &mdash; whether dataset requires offline conversion</li>
  <li><code>default_plots: [...]</code> &mdash; default visualizers auto-run after training</li>
</ul>
""",
    ),
    DocArticle(
        id="cluster_slurm",
        title="Cluster Execution & Slurm Runner",
        category="Workflow & Tools",
        summary="How to submit jobs to HPC clusters using Slurm site profiles, resource limits, and email notifications.",
        keywords=["slurm", "cluster", "hpc", "ncshare", "arc", "sbatch", "gpu"],
        html_content="""
<h2>Cluster & Slurm Execution</h2>
<p>Theta-IDE supports seamless transitions between local testing and cluster-scale execution via Slurm.</p>

<h3>Site Profiles (<code>site</code>)</h3>
<ul>
  <li><code>site=local</code> (default): Interactive local execution inside subprocesses.</li>
  <li><code>site=ncshare</code> / <code>site=arc</code>: Automatically generates and dispatches Slurm batch scripts (<code>sbatch</code>).</li>
</ul>

<div style="background-color: rgba(251, 73, 52, 0.12); border-left: 4px solid #fb4934; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Cluster Push Mandate:</b> Always commit and push all code changes to GitHub before submitting cluster jobs, as remote compute nodes sync directly from the repository.
</div>

<h3>Configuring Cluster Resources</h3>
<p>Specify compute requirements directly in your experiment YAML:</p>
<pre><code>resources:
  time: "04:00:00"   # Wall clock limit (hh:mm:ss)
  gpus: 1            # Number of GPU accelerators
  cores: 16          # CPU cores allocated
  memory: "32G"      # RAM allocation</code></pre>
<p>Priority order: CLI flags &gt; experiment YAML &gt; site default config &gt; fallback defaults.</p>

<h3>Automated Slurm Pipeline Features</h3>
<ul>
  <li><b>Dependency Chaining:</b> Online data generation jobs automatically establish Slurm dependency holds (<code>--dependency=afterok:&lt;job_id&gt;</code>) on downstream offline comparison jobs.</li>
  <li><b>Status Notifications:</b> Configured with <code>--mail-type=END,FAIL</code> for immediate progress updates.</li>
</ul>
""",
    ),
    DocArticle(
        id="plugins_guide",
        title="Core Plugins & Community Extensions",
        category="Extensibility",
        summary="Guide to creating and managing plugins: Core vs Community plugins, lifecycle hooks, and the Plugin Context API.",
        keywords=["plugins", "extensions", "core", "community", "lifecycle", "api", "hub"],
        html_content="""
<h2>Plugin Architecture & Extensibility</h2>
<p>Theta-IDE features a modular plugin architecture modeled on Obsidian's core/community plugin design.</p>

<h3>Core Plugins vs. Community Plugins</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Aspect</td>
    <td>Core Plugins (e.g. Documentation)</td>
    <td>Community Plugins</td>
  </tr>
  <tr>
    <td><b>Origin</b></td>
    <td>Shipped natively with Theta-IDE source repository (<code>frontend/plugins/core/</code>)</td>
    <td>Installed from Community Hub into user storage (<code>.thetaide/plugins/</code>)</td>
  </tr>
  <tr>
    <td><b>Uninstallable</b></td>
    <td><b>No</b> &mdash; core features cannot be deleted from the filesystem</td>
    <td><b>Yes</b> &mdash; can be uninstalled and removed via trash button</td>
  </tr>
  <tr>
    <td><b>System Impact</b></td>
    <td><b>Zero impact when toggled off</b> &mdash; all panes, listeners, and widgets are cleanly unmounted</td>
    <td>Zero impact when toggled off</td>
  </tr>
  <tr>
    <td><b>Settings Location</b></td>
    <td><i>Settings &rarr; Core Plugins</i></td>
    <td><i>Settings &rarr; Community Plugins</i></td>
  </tr>
</table>

<h3>Plugin Structure</h3>
<p>Every plugin requires a directory containing:</p>
<ol>
  <li><code>plugin.json</code>: Metadata manifest (ID, name, description, version, entry point, <code>core: true/false</code>).</li>
  <li><code>__init__.py</code>: Exports a class subclassing <code>Plugin</code>.</li>
</ol>

<h3>Plugin Lifecycle Hooks</h3>
<pre><code>from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

class MyPlugin(Plugin):
    def activate(self, context: PluginContext) -> None:
        # Called when the plugin is enabled
        self.context = context
        self.widget = MyCustomWidget()
        context.add_sidebar_tab(
            tab_id="my_plugin",
            widget=self.widget,
            title="My Extension",
            icon_name="my_icon",
            short_label="Extension"
        )

    def deactivate(self) -> None:
        # Called when toggled off or during IDE shutdown
        # Must clean up all widgets and event listeners
        if self.context:
            self.context.remove_sidebar_tab("my_plugin")
            self.widget.deleteLater()
            self.widget = None
            self.context = None</code></pre>
""",
    ),
    DocArticle(
        id="cli_workflows",
        title="Command-Line (CLI) Workflows",
        category="Workflow & Tools",
        summary="Complete CLI reference for run_pipeline.py, standalone training, and auto-plotting.",
        keywords=["cli", "terminal", "commands", "run_pipeline", "train.py", "optuna", "sweeps"],
        html_content="""
<h2>CLI & Command Reference</h2>
<p>Theta-IDE is backed by a fully scriptable CLI pipeline. All operations can be invoked directly from the terminal or embedded terminal emulator.</p>

<h3>Running Pipelines</h3>
<pre><code># Run full end-to-end experiment pipeline (Online -> Offline -> Plotting)
python run_pipeline.py cartpole/cp_final

# Run MIMIC offline comparison benchmark
python run_pipeline.py mimic/mimic_comparison

# Override experiment parameters via Hydra CLI
python run_pipeline.py cartpole/cp_final total_timesteps=50000 eval_episodes=10

# Run in quick-check mode with local execution
python run_pipeline.py quick_tests/smoke_test local=true</code></pre>

<h3>Direct Training Execution</h3>
<pre><code># Train PPO directly in online mode
python src/app/train.py +experiment=cartpole/cp_final mode=online agent=ppo/cp_tuned

# Train Offline IQL on CartPole replay buffer
python src/app/train.py +experiment=cartpole/cp_final mode=offline agent=iql/cp_tuned mode.dataset_path=in/datasets/cartpole/cp_final/ppo_cp_tuned</code></pre>

<h3>Standalone Visualizations</h3>
<pre><code># Dispatch all plotters configured for an experiment
python plot/manager.py cartpole/cp_final

# Generate convergence curves with custom smoothing
python plot/convergence.py cartpole/cp_final --window 20 --dpi 300

# Plot specific loss functions
python plot/losses.py cartpole/cp_final --metrics losses/q_loss losses/actor_loss --window 15</code></pre>

<h3>Optuna Hyperparameter Sweeps</h3>
<pre><code># Launch multi-trial parameter sweep across all architectures
python run_pipeline.py tune_mimic_all -m</code></pre>
""",
    ),
    DocArticle(
        id="troubleshooting",
        title="Troubleshooting & FAQ",
        category="Getting Started",
        summary="Answers to common setup, backend, OpenGL, and cluster connection questions.",
        keywords=["troubleshooting", "faq", "errors", "backend", "connection", "opengl", "debug"],
        html_content="""
<h2>Troubleshooting & FAQ</h2>

<h3>1. Backend status says "connecting…" or "offline"</h3>
<p>Theta-IDE communicates with a background FastAPI daemon (default port: <code>8000</code>). If the backend is not responding:</p>
<ul>
  <li>Check <i>Settings &rarr; Backend API</i> to verify the daemon URL (typically <code>http://127.0.0.1:8000</code>).</li>
  <li>Ensure no other application is holding port 8000.</li>
  <li>Simulated demo runs (<i>Run &rarr; Start simulated demo</i>) do not require a live backend.</li>
</ul>

<h3>2. OpenGL / Qt Display Errors on Headless or Cluster Nodes</h3>
<p>When running tests or GUI components on a headless server, export the offscreen platform plugin:</p>
<pre><code>export QT_QPA_PLATFORM=offscreen</code></pre>

<h3>3. Where are my training results and logs stored?</h3>
<p>Results adhere to a standardized hierarchical structure:</p>
<ul>
  <li><b>Metrics & Logs:</b> <code>results/logs/[GROUP]/[EXP_ID]/[AGENT]/version_X/metrics.csv</code></li>
  <li><b>Checkpoints:</b> <code>results/checkpoints/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Replay Buffers:</b> <code>results/datasets/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Plots & Reports:</b> <code>results/plots/[GROUP]/[EXP_ID]/</code></li>
</ul>

<h3>4. How do I reset the IDE's layout or cached settings?</h3>
<p>Navigate to <i>Settings &rarr; Workspace & Storage</i> and click <b>Reset Sidebar Layout</b> to restore default panel visibility and tab ordering.</p>
""",
    ),
]


def get_all_articles() -> List[DocArticle]:
    """Return all available documentation articles."""
    return list(ARTICLES)


def get_article_by_id(article_id: str) -> DocArticle | None:
    """Retrieve an article by its unique identifier."""
    for art in ARTICLES:
        if art.id == article_id:
            return art
    return None


def search_articles(query: str) -> List[DocArticle]:
    """Filter articles by search query against title, summary, keywords, and content."""
    query = query.strip().lower()
    if not query:
        return get_all_articles()

    results: List[DocArticle] = []
    for art in ARTICLES:
        if (
            query in art.title.lower()
            or query in art.summary.lower()
            or query in art.category.lower()
            or any(query in kw.lower() for kw in art.keywords)
            or query in art.html_content.lower()
        ):
            results.append(art)
    return results
