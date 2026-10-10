Announcements
#############

Release notes, technical updates, examples, and deployment stories from the Model Optimizer team.

.. raw:: html

   <section class="announcement-section" aria-label="Announcement index">
     <div class="announcement-toolbar">
       <label class="announcement-search-label" for="announcement-search">Search announcements</label>
       <input id="announcement-search" class="announcement-search" type="search" placeholder="Search announcements" autocomplete="off" />
       <div class="announcement-tags" aria-label="Announcement tags">
         <button class="announcement-tag is-active" type="button" data-tag="all" aria-pressed="true">All</button>
         <button class="announcement-tag" type="button" data-tag="release" aria-pressed="false">Release</button>
         <button class="announcement-tag" type="button" data-tag="autoquantize" aria-pressed="false">AutoQuantize</button>
         <button class="announcement-tag" type="button" data-tag="speculative-decoding" aria-pressed="false">Speculative decoding</button>
         <button class="announcement-tag" type="button" data-tag="dflash" aria-pressed="false">DFlash</button>
         <button class="announcement-tag" type="button" data-tag="dspark" aria-pressed="false">DSpark</button>
         <button class="announcement-tag" type="button" data-tag="domino" aria-pressed="false">Domino</button>
         <button class="announcement-tag" type="button" data-tag="architecture" aria-pressed="false">Architecture</button>
         <button class="announcement-tag" type="button" data-tag="docs" aria-pressed="false">Docs</button>
         <button class="announcement-tag" type="button" data-tag="github-pages" aria-pressed="false">GitHub Pages</button>
       </div>
     </div>

   <div class="announcement-grid" id="announcement-grid">
     <article class="announcement-card" data-date="2026-08-24" data-title="AutoQuantize: A Fast Automatic Mixed-Precision Assignment" data-summary="AutoQuantize finds low-sensitivity mixed-precision assignments with gradient-based scoring under a modeled effective-bits budget." data-tags="autoquantize quantization mixed-precision modelopt">
       <div class="announcement-card-meta">August 24, 2026 &middot; Model Optimizer Team</div>
       <h2><a href="announcements/autoquantize.html">AutoQuantize: A Fast Automatic Mixed-Precision Assignment</a></h2>
       <p>AutoQuantize finds low-sensitivity mixed-precision assignments with gradient-based scoring under a modeled effective-bits budget.</p>
       <div class="announcement-card-tags"><span>autoquantize</span><span>quantization</span><span>mixed-precision</span><span>modelopt</span></div>
     </article>
     <article class="announcement-card" data-date="2026-08-13" data-title="Model Optimizer announcements are moving to GitHub Pages" data-summary="The public Model Optimizer site is gaining a PR-updated announcements landing page within the existing Sphinx documentation." data-tags="release docs github-pages">
       <div class="announcement-card-meta">August 13, 2026 &middot; Model Optimizer Team</div>
       <h2><a href="announcements/github-pages-announcements.html">Model Optimizer announcements are moving to GitHub Pages</a></h2>
       <p>The GitHub Pages site now starts with announcements while the existing API documentation remains available in the docs navigation.</p>
       <div class="announcement-card-tags"><span>release</span><span>docs</span><span>github-pages</span></div>
     </article>
     <article class="announcement-card" data-date="2026-07-13" data-title="DSpark vs Domino: Same DFlash Backbone, Different Correction Heads" data-summary="DSpark and Domino both build on block-parallel DFlash draft generation but diverge in their token-level correction heads." data-tags="speculative-decoding dflash dspark domino architecture">
       <div class="announcement-card-meta">July 13, 2026 &middot; Model Optimizer Team</div>
       <h2><a href="announcements/dspark-vs-domino.html">DSpark vs Domino: Same DFlash Backbone, Different Correction Heads</a></h2>
       <p>DSpark and Domino share a DFlash backbone but make different correction-head tradeoffs: ModelOpt's <code>vanilla</code> Markov head versus a GRU.</p>
       <div class="announcement-card-tags"><span>speculative-decoding</span><span>dflash</span><span>dspark</span><span>domino</span><span>architecture</span></div>
     </article>
   </div>

   <p class="announcement-empty" id="announcement-empty" hidden>No announcements match this search.</p>
   <nav class="announcement-pager" id="announcement-pager" aria-label="Announcement pages" hidden>
     <button class="announcement-page-button" id="announcement-prev" type="button">Previous</button>
     <span class="announcement-page-status" id="announcement-page-status"></span>
     <button class="announcement-page-button" id="announcement-next" type="button">Next</button>
   </nav>
   </section>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Announcements

   self

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/[0-9]*
   Quick Start: PTQ - PyTorch <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq>
   Quick Start: PTQ - ONNX <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq>
   Quick Start: PTQ - PyTorch to ONNX <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/torch_onnx>
   Quick Start: PTQ - Windows <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows>
   Quick Start: QAT <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat>
   Quick Start: Pruning <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning>
   Quick Start: Distillation <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_distill>
   Quick Start: Speculative Decoding <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/speculative_decoding>
   Quick Start: Sparsity <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_sparsity>

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Guides

   guides/[0-9]*

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Deployment

   deployment/[0-9]*

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Examples

   examples/[0-9]*

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Reference

   reference/[0-9]*

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1
   :caption: Support

   support/[0-9]*
