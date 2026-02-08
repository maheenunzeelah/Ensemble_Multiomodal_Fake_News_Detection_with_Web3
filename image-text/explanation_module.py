import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import Rectangle
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Your data
data = {
    "index": 2732,
    "person_names": ["gillian_anderson"],
    "caption": "gillian anderson a young woman with a banana in her mouth",
    "headline": "gillian anderson with a banana in her nose on the set of the xfiles",
    "rouge1": {"precision": 0.636, "recall": 0.5, "f1": 0.56},
    "rouge2": {"precision": 0.5, "recall": 0.385, "f1": 0.435},
    "rougeL": {"precision": 0.636, "recall": 0.5, "f1": 0.56},
    "alignment": [
        {"token": "gillian", "matched": True},
        {"token": "anderson", "matched": True},
        {"token": "a", "matched": True},
        {"token": "young", "matched": False},
        {"token": "woman", "matched": False},
        {"token": "with", "matched": True},
        {"token": "a", "matched": True},
        {"token": "banana", "matched": True},
        {"token": "in", "matched": True},
        {"token": "her", "matched": True},
        {"token": "mouth", "matched": False}
    ],
    "coverage": {
        "covered": ["gillian", "anderson", "with", "a", "banana", "in", "her"],
        "missing": ["nose", "on", "the", "set", "of", "the", "xfiles"],
        "extra": ["young", "woman", "mouth"]
    },
    "semantic": {"semantic_similarity_score": 0.766},
    "visual_grounding": {"vgs": 0.0},
    "kg_similarity": {
        "kg_similarity_score": 0.8,
        "shared_entities": [
            "http://dbpedia.org/resource/Gillian_Anderson",
            "http://dbpedia.org/resource/Banana"
        ],
        "shared_types": [
            "Wikidata:Q5", "DBpedia:Person", "Wikidata:Q756", "DBpedia:Plant"
        ]
    },
    "headline_kg": {
        "entities": [
            {"surface": "gillian anderson", "uri": "http://dbpedia.org/resource/Gillian_Anderson"},
            {"surface": "banana", "uri": "http://dbpedia.org/resource/Banana"}
        ]
    },
    "caption_kg": {
        "entities": [
            {"surface": "gillian anderson", "uri": "http://dbpedia.org/resource/Gillian_Anderson"},
            {"surface": "banana", "uri": "http://dbpedia.org/resource/Banana"}
        ]
    }
}

# Create figure
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)

# Color scheme
colors = {
    'matched': '#27ae60',
    'unmatched': '#e74c3c',
    'extra': '#f39c12',
    'missing': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'tertiary': '#e67e22'
}

# 1. TOKEN OVERLAP VISUALIZATION (Venn-style)
ax1 = fig.add_subplot(gs[0:2, 0])
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw two overlapping circles
caption_circle = Circle((-0.35, 0), 0.8, color=colors['primary'], alpha=0.3, ec='black', linewidth=2)
headline_circle = Circle((0.35, 0), 0.8, color=colors['secondary'], alpha=0.3, ec='black', linewidth=2)
ax1.add_patch(caption_circle)
ax1.add_patch(headline_circle)

# Labels for circles
ax1.text(-0.75, 1.0, 'Generated\nCaption', ha='center', va='center', 
         fontsize=11, fontweight='bold', color=colors['primary'])
ax1.text(0.75, 1.0, 'Reference\nHeadline', ha='center', va='center', 
         fontsize=11, fontweight='bold', color=colors['secondary'])

# Overlap section (covered tokens)
covered_text = '\n'.join([f"• {t}" for t in data['coverage']['covered'][:4]])
if len(data['coverage']['covered']) > 4:
    covered_text += f"\n• +{len(data['coverage']['covered'])-4} more"
ax1.text(0, 0, covered_text, ha='center', va='center', 
         fontsize=9, fontweight='bold', color='#2c3e50')
ax1.text(0, -1.15, f"{len(data['coverage']['covered'])} shared tokens", 
         ha='center', fontsize=9, style='italic', color='#2c3e50')

# Extra tokens (only in caption)
extra_text = '\n'.join([f"• {t}" for t in data['coverage']['extra']])
ax1.text(-0.85, 0, extra_text, ha='center', va='center', 
         fontsize=8, color=colors['primary'], style='italic')
ax1.text(-0.85, -1.15, f"{len(data['coverage']['extra'])} extra", 
         ha='center', fontsize=8, color=colors['primary'])

# Missing tokens (only in headline)
missing_text = '\n'.join([f"• {t}" for t in data['coverage']['missing'][:4]])
if len(data['coverage']['missing']) > 4:
    missing_text += f"\n• +{len(data['coverage']['missing'])-4} more"
ax1.text(0.85, 0, missing_text, ha='center', va='center', 
         fontsize=8, color=colors['secondary'], style='italic')
ax1.text(0.85, -1.15, f"{len(data['coverage']['missing'])} missing", 
         ha='center', fontsize=8, color=colors['secondary'])

ax1.set_title('Token Overlap Analysis', fontsize=13, fontweight='bold', pad=10)

# 2. KNOWLEDGE GRAPH VISUALIZATION
ax2 = fig.add_subplot(gs[0:2, 1:])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'Knowledge Graph Analysis', ha='center', fontsize=13, 
         fontweight='bold', color='#2c3e50')

# Caption KG
caption_kg_y = 7.5
ax2.add_patch(FancyBboxPatch((0.5, caption_kg_y-0.5), 4, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#e8f4f8', edgecolor=colors['primary'], linewidth=2))
ax2.text(2.5, caption_kg_y+1.5, 'Caption Entities', ha='center', 
         fontsize=10, fontweight='bold', color=colors['primary'])

y_pos = caption_kg_y + 0.8
for ent in data['caption_kg']['entities']:
    name = ent['surface'].title()
    ax2.text(2.5, y_pos, f"• {name}", ha='center', fontsize=9, color='#2c3e50')
    y_pos -= 0.4

# Headline KG
headline_kg_y = 7.5
ax2.add_patch(FancyBboxPatch((5.5, headline_kg_y-0.5), 4, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#f4e8f8', edgecolor=colors['secondary'], linewidth=2))
ax2.text(7.5, headline_kg_y+1.5, 'Headline Entities', ha='center', 
         fontsize=10, fontweight='bold', color=colors['secondary'])

y_pos = headline_kg_y + 0.8
for ent in data['headline_kg']['entities']:
    name = ent['surface'].title()
    ax2.text(7.5, y_pos, f"• {name}", ha='center', fontsize=9, color='#2c3e50')
    y_pos -= 0.4

# Shared entities (middle)
shared_y = 4.5
ax2.add_patch(FancyBboxPatch((2, shared_y-0.5), 6, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#d5f4e6', edgecolor=colors['matched'], linewidth=2))
ax2.text(5, shared_y+1, 'Shared Entities ✓', ha='center', 
         fontsize=10, fontweight='bold', color=colors['matched'])

y_pos = shared_y + 0.3
for uri in data['kg_similarity']['shared_entities']:
    name = uri.split('/')[-1].replace('_', ' ')
    ax2.text(5, y_pos, f"• {name}", ha='center', fontsize=9, color='#2c3e50')
    y_pos -= 0.4

# KG Similarity Score
score_box_y = 1.5
ax2.add_patch(FancyBboxPatch((3, score_box_y-0.3), 4, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2))
score = data['kg_similarity']['kg_similarity_score']
ax2.text(5, score_box_y+0.5, f'KG Similarity Score', ha='center', 
         fontsize=10, fontweight='bold', color='#856404')
ax2.text(5, score_box_y-0.1, f'{score:.2%}', ha='center', 
         fontsize=20, fontweight='bold', color='#f39c12')

# Interpretation
ax2.text(5, 0.3, f'Both caption and headline identify the same key entities\n(person and object), yielding high knowledge overlap', 
         ha='center', fontsize=8, style='italic', color='#666', wrap=True)

# 3. ROUGE METRICS HEATMAP
ax3 = fig.add_subplot(gs[2, :])
rouge_data = np.array([
    [data['rouge1']['precision'], data['rouge1']['recall'], data['rouge1']['f1']],
    [data['rouge2']['precision'], data['rouge2']['recall'], data['rouge2']['f1']],
    [data['rougeL']['precision'], data['rougeL']['recall'], data['rougeL']['f1']]
])

im = ax3.imshow(rouge_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=10, fontweight='bold')
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], fontsize=10, fontweight='bold')

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax3.text(j, i, f'{rouge_data[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax3.set_title('ROUGE Metrics Performance', fontsize=13, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax3, orientation='vertical', pad=0.02)

# 4. TOKEN ALIGNMENT WITH COLORS
ax4 = fig.add_subplot(gs[3, :])
ax4.axis('off')

# Caption with color coding
caption_tokens = [item['token'] for item in data['alignment']]
caption_matched = [item['matched'] for item in data['alignment']]

x_pos = 0.05
y_pos = 0.7
ax4.text(x_pos, y_pos+0.15, 'Generated Caption (Token-by-Token):', 
         fontsize=11, fontweight='bold', transform=ax4.transAxes)

for token, matched in zip(caption_tokens, caption_matched):
    color = colors['matched'] if matched else colors['unmatched']
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax4.text(x_pos, y_pos, token, fontsize=10, transform=ax4.transAxes, 
             bbox=bbox, fontweight='bold' if matched else 'normal')
    x_pos += len(token) * 0.012 + 0.025

# Headline tokens
headline_tokens = data['headline'].split()
x_pos = 0.05
y_pos = 0.35
ax4.text(x_pos, y_pos+0.15, 'Reference Headline:', 
         fontsize=11, fontweight='bold', transform=ax4.transAxes)

for token in headline_tokens:
    if token in data['coverage']['covered']:
        color = colors['matched']
        alpha = 0.3
        weight = 'bold'
    elif token in data['coverage']['missing']:
        color = colors['missing']
        alpha = 0.2
        weight = 'normal'
    else:
        color = 'gray'
        alpha = 0.1
        weight = 'normal'
    
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=alpha, 
                edgecolor=color, linewidth=2)
    ax4.text(x_pos, y_pos, token, fontsize=10, transform=ax4.transAxes, 
             bbox=bbox, fontweight=weight)
    x_pos += len(token) * 0.012 + 0.025

# Legend
legend_y = 0.05
ax4.text(0.05, legend_y, '■ Matched', fontsize=9, color=colors['matched'], 
         fontweight='bold', transform=ax4.transAxes)
ax4.text(0.15, legend_y, '■ Missing/Unmatched', fontsize=9, color=colors['unmatched'], 
         transform=ax4.transAxes)
ax4.text(0.35, legend_y, '■ Extra (not in reference)', fontsize=9, 
         color=colors['extra'], transform=ax4.transAxes)

# 5. OVERALL METRICS DASHBOARD
ax5 = fig.add_subplot(gs[4, :])

metrics_data = [
    ('Semantic\nSimilarity', data['semantic']['semantic_similarity_score'], 
     'Embedding-based\nmeaning overlap', colors['primary']),
    ('KG\nSimilarity', data['kg_similarity']['kg_similarity_score'], 
     'Shared entities\nin knowledge graph', colors['secondary']),
    ('ROUGE-1\nF1', data['rouge1']['f1'], 
     'Unigram\noverlap score', colors['tertiary']),
    ('ROUGE-L\nF1', data['rougeL']['f1'], 
     'Longest common\nsubsequence', colors['matched']),
    ('Visual\nGrounding', data['visual_grounding']['vgs'], 
     'Object detection\nalignment', colors['unmatched'])
]

bar_width = 0.15
x_positions = np.arange(len(metrics_data))

for i, (label, score, desc, color) in enumerate(metrics_data):
    bar = ax5.bar(i, score, bar_width*3, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.text(i, score + 0.05, f'{score:.3f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')
    ax5.text(i, -0.15, desc, ha='center', va='top', fontsize=7, 
             style='italic', color='#666', multialignment='center')

ax5.set_xticks(x_positions)
ax5.set_xticklabels([m[0] for m in metrics_data], fontsize=10, fontweight='bold')
ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
ax5.set_ylim(-0.25, 1.1)
ax5.set_title('Multi-Dimensional Similarity Metrics', fontsize=13, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Main title
fig.suptitle(f'Caption Quality Assessment Dashboard - Image #{data["index"]}', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

# Print detailed interpretation
print("=" * 80)
print("KNOWLEDGE GRAPH INTERPRETATION")
print("=" * 80)
print(f"\n📊 KG Similarity Score: {data['kg_similarity']['kg_similarity_score']:.1%}")
print(f"\n✅ Shared Entities ({len(data['kg_similarity']['shared_entities'])}):")
for uri in data['kg_similarity']['shared_entities']:
    entity_name = uri.split('/')[-1].replace('_', ' ')
    print(f"   • {entity_name}")

print(f"\n🔍 Entity Types Overlap ({len(data['kg_similarity']['shared_types'])} types):")
type_samples = [t.split(':')[-1] for t in data['kg_similarity']['shared_types'][:5]]
print(f"   • {', '.join(type_samples)}...")

print("\n💡 INTERPRETATION:")
print(f"   Both the caption and headline correctly identify the same real-world entities:")
print(f"   - Person: Gillian Anderson (DBpedia entity)")
print(f"   - Object: Banana (DBpedia entity)")
print(f"   This {data['kg_similarity']['kg_similarity_score']:.0%} similarity indicates strong entity-level agreement,")
print(f"   even though the spatial relationship differs (mouth vs. nose).")
print("=" * 80)