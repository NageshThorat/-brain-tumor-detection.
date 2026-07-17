import json
with open('c:/Users/thora/OneDrive/Desktop/brain tumor detection/papers.json', encoding='utf-16') as f:
    data = json.load(f)

res = data.get('results', [])

with open('c:/Users/thora/OneDrive/Desktop/brain tumor detection/literature_review.md', 'w', encoding='utf-8') as f:
    f.write('# Literature Survey: Brain Tumor Detection using MRI Images\n\n')
    for i, p in enumerate(res[:20]):
        title = p.get('title')
        year = p.get('publication_year')
        doi = p.get('doi') or p.get('id')
        citations = p.get('cited_by_count')
        
        abstract_data = p.get('abstract_inverted_index')
        abstract = "No abstract available."
        if abstract_data:
            # Reconstruct abstract from inverted index
            words = []
            for word, positions in abstract_data.items():
                for pos in positions:
                    words.append((pos, word))
            words.sort(key=lambda x: x[0])
            abstract = ' '.join([w[1] for w in words])
            
        f.write(f'## {i+1}. {title}\n')
        f.write(f'**Year**: {year}\n')
        f.write(f'**DOI/URL**: {doi}\n')
        f.write(f'**Citations**: {citations}\n\n')
        f.write(f'**Abstract**: {abstract}\n\n')
