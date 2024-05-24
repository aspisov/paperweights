"""
Contains various utility functions for word2vec: save embeddings, load embeddings, and plot embeddings.
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Save the embeddings from the model to a txt file
def save_embeddings(model, vocab, file_path):
    embeddings = model.input_embeddings.weight.data.cpu().numpy()
    words = vocab

    with open(file_path, "w") as f:
        for word, vector in zip(words, embeddings):
            vector_str = " ".join(map(str, vector))
            f.write(f"{word} {vector_str}\n")


# Load the embeddings from a txt file
def load_embeddings(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    words = []
    embeddings = []

    for line in lines:
        parts = line.strip().split()
        words.append(parts[0])
        embeddings.append([float(val) for val in parts[1:]])

    return words, np.array(embeddings)


def plot_embeddings(embeddings, words):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(20, 20))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], s=10)
        plt.annotate(
            word,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            fontsize=8,
            alpha=0.7,
        )
    plt.title("Word Embeddings Visualization", fontsize=14)  # Title size
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(False)  # Remove grid
    plt.axis("off")
    plt.show()
    
def get_list_of_interesting_words():
    interesting_words = [
        'king', 'queen', 'man', 'woman', 'paris', 'france', 'apple', 'orange', 'computer', 'science',
        'math', 'physics', 'chemistry', 'biology', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'venus',
        'mercury', 'pluto', 'galaxy', 'universe', 'star', 'planet', 'asteroid', 'comet', 'meteor', 'telescope',
        'microscope', 'atom', 'molecule', 'cell', 'gene', 'dna', 'rna', 'protein', 'enzyme', 'bacteria',
        'virus', 'fungi', 'plant', 'animal', 'human', 'brain', 'heart', 'liver', 'kidney', 'lung',
        'bone', 'muscle', 'nerve', 'skin', 'blood', 'oxygen', 'carbon', 'hydrogen', 'nitrogen', 'helium',
        'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon', 'sodium', 'magnesium',
        'aluminum', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon', 'potassium', 'calcium', 'scandium', 'titanium',
        'vanadium', 'chromium', 'manganese', 'iron', 'cobalt', 'nickel', 'copper', 'zinc', 'gallium', 'germanium',
        'arsenic', 'selenium', 'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium', 'niobium', 'molybdenum',
        'sun', 'moon', 'mars', 'comet', 'nebula', 'supernova', 'blackhole', 'quantum', 'relativity', 'string',
        'gravity', 'dimension', 'wormhole', 'singularity', 'cosmos', 'astro', 'biosphere', 'cryosphere', 'hydrosphere', 'lithosphere',
        'atmosphere', 'ecosystem', 'climate', 'weather', 'tornado', 'hurricane', 'earthquake', 'volcano', 'eruption', 'tsunami',
        'seismic', 'plate', 'tectonic', 'crust', 'mantle', 'core', 'magnetosphere', 'ionosphere', 'thermosphere', 'exosphere',
        'troposphere', 'stratosphere', 'mesosphere', 'ozone', 'greenhouse', 'carbon', 'dioxide', 'oxygen', 'methane', 'nitrous',
        'chlorofluorocarbon', 'ozone', 'depletion', 'pollution', 'smog', 'acid', 'rain', 'renewable', 'nonrenewable', 'fossil',
        'fuel', 'solar', 'wind', 'hydro', 'geothermal', 'biomass', 'nuclear', 'fusion', 'fission', 'reactor',
        'radiation', 'isotope', 'half-life', 'alpha', 'beta', 'gamma', 'particle', 'wave', 'duality', 'photon',
        'neutron', 'proton', 'electron', 'quark', 'lepton', 'boson', 'gluon', 'hadron', 'meson', 'baryon',
        'nucleus', 'orbital', 'spin', 'magnetic', 'moment', 'vector', 'scalar', 'tensor', 'field', 'force',
        'mass', 'energy', 'momentum', 'inertia', 'velocity', 'acceleration', 'gravity', 'electromagnetic', 'weak', 'strong',
        'interaction', 'fundamental', 'particle', 'antimatter', 'dark', 'matter', 'energy', 'big', 'bang', 'cosmic',
        'microwave', 'background', 'radiation', 'inflation', 'singularity', 'multiverse', 'parallel', 'universe', 'dimension', 'time',
        'space', 'continuum', 'fabric', 'curvature', 'warp', 'drive', 'wormhole', 'starship', 'enterprise', 'shuttle',
        'spacecraft', 'satellite', 'probe', 'rover', 'lander', 'orbiter', 'telescope', 'observatory', 'spectrometer', 'detector',
        'sensor', 'camera', 'lens', 'mirror', 'prism', 'filter', 'aperture', 'focal', 'length', 'resolution',
        'magnification', 'zoom', 'focus', 'exposure', 'contrast', 'brightness', 'saturation', 'hue', 'color', 'balance',
        'temperature', 'white', 'black', 'gray', 'scale', 'tone', 'shade', 'tint', 'palette', 'spectrum',
        'wavelength', 'frequency', 'amplitude', 'phase', 'interference', 'diffraction', 'refraction', 'reflection', 'absorption', 'emission',
        'transmission', 'scattering', 'polarization', 'coherence', 'decoherence', 'entanglement', 'superposition', 'quantum', 'bit', 'qubit',
        'algorithm', 'encryption', 'decryption', 'cryptography', 'cipher', 'key', 'hash', 'function', 'blockchain', 'ledger',
        'bitcoin', 'cryptocurrency', 'wallet', 'transaction', 'mining', 'hashrate', 'proof', 'stake', 'work', 'distributed',
        'network', 'node', 'peer', 'decentralized', 'protocol', 'smart', 'contract', 'token', 'ico', 'exchange',
        'trading', 'market', 'price', 'value', 'supply', 'demand', 'volatility', 'liquidity', 'capital', 'investment',
        'portfolio', 'diversification', 'risk', 'return', 'asset', 'equity', 'bond', 'stock', 'commodity', 'future',
        'option', 'derivative', 'hedge', 'fund', 'mutual', 'exchange', 'traded', 'etf', 'index', 'benchmark',
        'performance', 'yield', 'dividend', 'interest', 'rate', 'inflation', 'deflation', 'recession', 'depression', 'growth',
        'gdp', 'gni', 'per', 'capita', 'income', 'consumption', 'expenditure', 'saving', 'investment', 'capital',
        'formation', 'accumulation', 'distribution', 'inequality', 'poverty', 'wealth', 'income', 'class', 'mobility', 'opportunity',
        'development', 'sustainability', 'environment', 'climate', 'change', 'global', 'warming', 'greenhouse', 'gas', 'emission',
        'carbon', 'footprint', 'renewable', 'energy', 'solar', 'wind', 'hydro', 'geothermal', 'biomass', 'nuclear',
        'power', 'plant', 'grid', 'distribution', 'transmission', 'smart', 'meter', 'efficiency', 'conservation', 'recycling',
        'waste', 'management', 'pollution', 'control', 'air', 'water', 'soil', 'noise', 'light', 'land',
        'use', 'planning', 'urban', 'rural', 'development', 'infrastructure', 'transportation', 'road', 'rail', 'air',
        'water', 'network', 'communication', 'telecommunication', 'internet', 'broadband', 'fiber', 'optic', 'satellite', 'wireless',
        'wifi', 'bluetooth', 'nfc', 'iot', 'smart', 'city', 'automation', 'robotics', 'ai', 'artificial',
        'intelligence', 'machine', 'learning', 'deep', 'neural', 'network', 'algorithm', 'data', 'big', 'cloud',
        'computing', 'virtual', 'reality', 'augmented', 'reality', 'mixed', 'reality', 'blockchain', 'cryptocurrency', 'bitcoin',
        'ethereum', 'smart', 'contract', 'decentralized', 'finance', 'defi', 'ico', 'token', 'exchange', 'wallet',
        'transaction', 'mining', 'proof', 'work', 'stake', 'consensus', 'protocol', 'node', 'peer', 'network',
        'distributed', 'ledger', 'hash', 'encryption', 'security', 'privacy', 'anonymity', 'pseudonymity', 'trust', 'transparency',
        'governance', 'regulation', 'compliance', 'audit', 'risk', 'management', 'fraud', 'prevention', 'detection', 'response',
        'cybersecurity', 'attack', 'threat', 'vulnerability', 'exploit', 'malware', 'virus', 'worm', 'trojan', 'botnet',
        'ddos', 'phishing', 'ransomware', 'spyware', 'adware', 'rootkit', 'backdoor', 'zero-day', 'patch', 'update',
        'firewall', 'antivirus', 'intrusion', 'detection', 'system', 'ids', 'ips', 'siem', 'log', 'monitoring',
        'incident', 'response', 'disaster', 'recovery', 'business', 'continuity', 'plan', 'bcp', 'backup', 'restore',
        'redundancy', 'failover', 'resilience', 'capacity', 'planning', 'scalability', 'performance', 'optimization', 'tuning', 'load',
        'balancing', 'high', 'availability', 'service', 'level', 'agreement', 'sla', 'quality', 'assurance', 'qa',
        'testing', 'unit', 'integration', 'system', 'user', 'acceptance', 'uat', 'automation', 'manual', 'regression',
        'performance', 'load', 'stress', 'security', 'testing', 'penetration', 'test', 'ethical', 'hacking', 'bug',
        'bounty', 'program', 'vulnerability', 'management', 'patch', 'management', 'configuration', 'management', 'change', 'control',
        'version', 'control', 'svn', 'git', 'repository', 'branch', 'merge', 'commit', 'push', 'pull',
        'clone', 'fork', 'tag', 'release', 'deploy', 'rollback', 'continuous', 'integration', 'ci', 'continuous',
        'delivery', 'cd', 'devops', 'agile', 'scrum', 'kanban', 'lean', 'methodology', 'project', 'management',
        'pmp', 'prince2', 'itil', 'cobit', 'iso', 'certification', 'training', 'course', 'workshop', 'seminar',
        'conference', 'webinar', 'summit', 'symposium', 'forum', 'panel', 'discussion', 'roundtable', 'networking', 'event',
        'meetup', 'user', 'group', 'community', 'conference', 'expo', 'trade', 'show', 'fair', 'exhibition',
        'booth', 'sponsorship', 'partnership', 'collaboration', 'cooperation', 'joint', 'venture', 'alliance', 'consortium', 'association'
    ]
    return interesting_words