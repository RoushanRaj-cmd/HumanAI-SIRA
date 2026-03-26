import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SIRA1 Epidemic PINN'
copyright = '2026, HumanAI GSoC'
author = 'HumanAI GSoC'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
