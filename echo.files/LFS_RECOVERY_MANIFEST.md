# LFS Recovery Manifest

## Background

During KSM Evolution Cycle 13 (2026-03-21), all 25 Git LFS objects in this repository
were found to be missing from the GitHub LFS server (HTTP 404). This blocked all CI/CD
pipelines and Copilot SWE Agent runs.

The LFS tracking was removed and replaced with direct file storage. Files that could be
recovered from upstream sources were restored. Echo-specific files that had no backup
are listed below as requiring regeneration.

## Files Requiring Regeneration

The following files were echo-specific additions whose LFS objects were never properly
stored on the server. They need to be regenerated or re-uploaded:

### Architecture Diagrams (can be regenerated from descriptions)
| File | Original Size | Status |
|------|--------------|--------|
| `echo.files/EchoPipeLLM.png` | 1.3 MB | Needs regeneration - Echo LLM pipeline diagram |
| `echo.files/echo_propagation_engine.png` | 2.3 MB | Needs regeneration - Echo propagation engine |
| `echo.rkwv/docs/architecture/deep_tree_echo_architecture_overview.png` | 2.2 MB | Needs regeneration |
| `echo.rkwv/docs/architecture/extension_layer_architecture.png` | 2.5 MB | Needs regeneration |
| `echo.rkwv/docs/architecture/hypergraph_memory_space.png` | 2.8 MB | Needs regeneration |
| `echo.rkwv/docs/architecture/neural_symbolic_integration.png` | 2.2 MB | Needs regeneration |

### EVA Architecture Files (from external sources)
| File | Original Size | Status |
|------|--------------|--------|
| `echo.self/eva/architecture/Basic_Forward_Model.png` | 4 KB | Needs re-download |
| `echo.self/eva/architecture/PUMA-Overview.png` | 132 KB | Needs re-download |
| `echo.self/eva/architecture/ci2cv-demo-cropped.jpg` | 58 KB | Needs re-download |
| `echo.self/eva/architecture/concept-graph.svg` | 104 KB | Needs re-download |
| `echo.self/eva/architecture/data-graph-nodes-edges.jpg` | 48 KB | Needs re-download |
| `echo.self/eva/architecture/embodiment.pdf` | 582 KB | Needs re-download |
| `echo.self/eva/architecture/think9.jpg` | 156 KB | Needs re-download |

### Research Papers
| File | Original Size | Status |
|------|--------------|--------|
| `echo.kern/HyperGESN-2310.10177v1.pdf` | 842 KB | Re-download from arXiv:2310.10177 |
| `echo.kern/melandri_luca_tesi.pdf` | 1.6 MB | Needs re-download from source |

### Application Assets
| File | Original Size | Status |
|------|--------------|--------|
| `echo.self/public/favicon.ico` | 17 KB | Needs regeneration |
| `echo.self/public/favicon.svg` | 304 B | Needs regeneration |
| `echo.self/public/logo-dark.png` | 80 KB | Needs regeneration |
| `echo.self/public/logo-light.png` | 6 KB | Needs regeneration |
| `echo.sys/Pyper - *.png` | 4.3 MB | Needs regeneration |

### Pilot Exports
| File | Original Size | Status |
|------|--------------|--------|
| `echo.pilot/*.tar.gz` | 369 KB + 45 KB | Needs re-export from source |

## Recovery Instructions

1. **Architecture diagrams**: Regenerate using AI image generation based on the descriptions
   in the corresponding README files and code documentation.
2. **Research papers**: Download from arXiv or original academic sources.
3. **Application assets**: Regenerate favicons and logos from the project branding guide.
4. **Pilot exports**: Re-export from the original Copilot conversation threads.

## Prevention

The `.gitattributes` has been updated to only track truly large ML model files (>50MB)
via LFS. All small media files are now committed directly to the repository.
