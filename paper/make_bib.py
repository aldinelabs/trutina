"""Generate refs.bib: arXiv entries from the arXiv API (title, authors, year taken from the record),
plus hand-written entries for models, datasets and software that have no arXiv record.

  uv run python make_bib.py
"""
import re, urllib.request, xml.etree.ElementTree as ET

IDS = {  # bibtex key -> arXiv id
 'lora': '2106.09685', 'gdn': '2412.06464', 'deltapar': '2406.06484', 'tunedlens': '2303.08112',
 'hinton': '1503.02531', 'calm': '2207.07061', 'layerskip': '2404.16710', 'guo': '1706.04599',
 'arora': '2407.05483', 'leviathan': '2512.14982', 'pride': '2309.03882', 'pine': '2407.01100',
 'liusie': '2403.13590', 'llmjev': '2610.02076', 'coconut': '2412.06769', 'recdepth': '2502.05171',
 'pause': '2310.02226', 'qrhead': '2506.09944', 'judgebench': '2410.12784', 'massive': '2204.08582',
 'clinc': '1909.02027', 'banking': '2003.04807', 'arc': '1803.05457', 'mmlu': '2009.03300',
 'apibank': '2304.08244', 'hellaswag': '1905.07830', 'cuad': '2103.06268', 'maud': '2301.00876',
 'sharc': '1809.01494', 'legalbench': '2308.11462', 'prm800k': '2305.20050', 'helpsteer': '2311.09528',
 'mtbench': '2306.05685', 'summeval': '2007.12626', 'halueval': '2305.11747', 'musique': '2108.00573',
 'wiki2': '2011.01060', 'hotpot': '1809.09600', 'drop': '1903.00161', 'tatqa': '2105.07624',
 'boolq': '1905.10044', 'strategyqa': '2101.02235', 'ropes': '1908.05852', 'mhs': '2009.10277',
 'sbic': '1911.03891', 'chaosnli': '2010.03532', 'snli': '1508.05326', 'mnli': '1704.05426',
 'anli': '1908.05739', 'mind2web': '2306.06070', 'som': '2310.11441', 'aokvqa': '2206.01718',
 'scienceqa': '2209.09513', 'tot': '2406.09170', 'tempreason': '2306.08952', 'finqa': '2109.00122',
 'hitab': '2108.06712', 'wikisql': '1709.00103', 'proofwriter': '2012.13048', 'boardgameqa': '2306.07934',
 'winogrande': '1907.10641', 'truthfulqa': '2109.07958', 'fpb': '1307.5336', 'ragtruth': '2401.00396',
 'circa': '2010.03450',
}
NS = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}

def fetch(ids):
    out = {}
    lst = list(ids)
    for i in range(0, len(lst), 40):
        url = 'https://export.arxiv.org/api/query?max_results=40&id_list=' + ','.join(lst[i:i+40])
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for e in root.findall('a:entry', NS):
            aid = re.search(r'abs/([\d.]+)', e.find('a:id', NS).text).group(1)
            title = ' '.join(e.find('a:title', NS).text.split())
            authors = [a.find('a:name', NS).text for a in e.findall('a:author', NS)]
            year = e.find('a:published', NS).text[:4]
            cat = e.find('x:primary_category', NS).attrib['term']
            out[aid] = (title, authors, year, cat)
    return out

import unicodedata
_ACC = {'\u0300': '`', '\u0301': "'", '\u0302': '^', '\u0303': '~', '\u0308': '"', '\u0304': '=', '\u0307': '.'}
_ACC_BRACE = {'\u030c': 'v', '\u0327': 'c', '\u030a': 'r', '\u0306': 'u', '\u0328': 'k', '\u030b': 'H'}
_SPECIAL = {'ł': r'{\l}', 'Ł': r'{\L}', 'ø': r'{\o}', 'Ø': r'{\O}', 'ß': r'{\ss}', 'æ': r'{\ae}', 'đ': r'{\dj}', 'ı': r'{\i}'}

def tex_accents(s):
    out = []
    for ch in unicodedata.normalize('NFD', s):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch]); continue
        if unicodedata.combining(ch):
            base = out.pop() if out else ''
            if ch in _ACC:
                out.append('{\\' + _ACC[ch] + base + '}')
            elif ch in _ACC_BRACE:
                out.append('{\\' + _ACC_BRACE[ch] + '{' + base + '}}')
            else:
                out.append(base)
            continue
        out.append(_SPECIAL.get(ch, ch))
    return ''.join(out)

def tex_escape(s):
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')):
        s = s.replace(a, b)
    return s

def fmt_authors(names):
    def one(n):
        p = n.split()
        return f"{p[-1]}, {' '.join(p[:-1])}" if len(p) > 1 else n
    if len(names) > 10:
        names = names[:3]
        return ' and '.join(one(n) for n in names) + ' and others'
    return ' and '.join(one(n) for n in names)

MANUAL = r'''
@misc{qwen35,
  author = {{Qwen Team}},
  title = {Qwen3.5: Towards Native Multimodal Agents},
  year = {2026},
  howpublished = {\url{https://qwen.ai/blog?id=qwen3.5}},
  note = {Model card: \url{https://huggingface.co/Qwen/Qwen3.5-4B}}
}
@misc{qwen38,
  author = {{Qwen Team}},
  title = {Qwen3.8-27B},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/Qwen/Qwen3.8-27B}}
}
@misc{logitlens,
  author = {{nostalgebraist}},
  title = {interpreting {GPT}: the logit lens},
  year = {2020},
  howpublished = {LessWrong, \url{https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens}}
}
@misc{fla,
  author = {{fla-org}},
  title = {flash-linear-attention: efficient implementations of linear attention},
  year = {2024},
  howpublished = {Software, \url{https://github.com/fla-org/flash-linear-attention}}
}
@misc{mlx,
  author = {{ml-explore}},
  title = {{MLX}: an array framework for {Apple} silicon},
  year = {2023},
  howpublished = {Software, \url{https://github.com/ml-explore/mlx}}
}
@misc{cudagraphs,
  author = {Gray, Alan},
  title = {Getting Started with {CUDA} Graphs},
  year = {2019},
  howpublished = {NVIDIA Technical Blog, \url{https://developer.nvidia.com/blog/cuda-graphs/}}
}
@misc{rknn,
  author = {{airockchip}},
  title = {rknn-llm: the {RKLLM} runtime},
  year = {2026},
  howpublished = {Software, \url{https://github.com/airockchip/rknn-llm}}
}
@misc{jevbench,
  author = {{Benchmark Heaven}},
  title = {{JevBench}: benchmark and leaderboard for {AI} decision models},
  year = {2026},
  howpublished = {Software and data, \url{https://github.com/fstandhartinger/jevbench}}
}
@misc{localllama,
  author = {{LocalLLaMA}},
  title = {typed-decisions},
  year = {2026},
  howpublished = {Hugging Face dataset card, \url{https://huggingface.co/datasets/LocalLLaMA/typed-decisions}}
}
@misc{decisionindex,
  author = {{apolinario}},
  title = {Decision Index: reproduce the decision-model benchmark suite},
  year = {2026},
  howpublished = {Software, \url{https://github.com/apolinario/decision-index}}
}
@misc{decisionbench,
  author = {{akhilaaa3}},
  title = {decision-bench},
  year = {2026},
  howpublished = {Hugging Face dataset, \url{https://huggingface.co/datasets/akhilaaa3/decision-bench}}
}
@misc{thisthat,
  author = {{limberc}},
  title = {this-that-complex-decisions},
  year = {2026},
  howpublished = {Hugging Face dataset, \url{https://huggingface.co/datasets/limberc/this-that-complex-decisions}}
}
@misc{fastdecisions,
  author = {{fastino}},
  title = {fast-decisions},
  year = {2026},
  howpublished = {Hugging Face dataset, \url{https://huggingface.co/datasets/fastino/fast-decisions}}
}
@misc{jev,
  author = {{TypeSafe}},
  title = {{Jev} 1.13 and the {System One} {API}},
  year = {2026},
  howpublished = {Documentation, \url{https://docs.typesafe.ai/concepts/system-one}}
}
@misc{pplx,
  author = {{Perplexity}},
  title = {pplx-decider-v1-27b},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/perplexity-ai/pplx-decider-v1-27b}}
}
@misc{d1,
  author = {{Liquid AI}},
  title = {Open d1},
  year = {2026},
  howpublished = {Blog post, \url{https://www.liquid.ai/blog/d1-open}}
}
@misc{semif,
  author = {{TheoLeeCJ}},
  title = {{SemIf} (formerly {OpenJev}): semantic ifs from open models},
  year = {2026},
  howpublished = {Software, \url{https://github.com/TheoLeeCJ/SemIf}}
}
@misc{plumb,
  author = {{crh225}},
  title = {Plumb-4B},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/crh225/plumb-4b}}
}
@misc{jevk5,
  author = {{ngrok-adhoc}},
  title = {{JevK5}},
  year = {2026},
  howpublished = {Software, \url{https://github.com/ngrok-adhoc/jevk5}}
}
@misc{decider4b,
  author = {{Mapika}},
  title = {decider-4b},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/Mapika/decider-4b}}
}
@misc{sieve,
  author = {{sthanika-ai}},
  title = {Sieve-4B},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/sthanika-ai/Sieve-4B}}
}
@misc{spark,
  author = {{abhishek085}},
  title = {spark-s1-4b-v6},
  year = {2026},
  howpublished = {Hugging Face model card, \url{https://huggingface.co/abhishek085/spark-s1-4b-v6}}
}
@misc{bfcl,
  author = {{Gorilla project}},
  title = {Berkeley Function-Calling Leaderboard},
  year = {2026},
  howpublished = {\url{https://gorilla.cs.berkeley.edu/leaderboard.html}}
}
'''

if __name__ == '__main__':
    meta = fetch(set(IDS.values()))
    missing = [i for i in IDS.values() if i not in meta]
    assert not missing, missing
    parts = []
    for key, aid in IDS.items():
        title, authors, year, cat = meta[aid]
        parts.append(f"@misc{{{key},\n  author = {{{tex_accents(fmt_authors(authors))}}},\n  title = {{{{{tex_accents(tex_escape(title))}}}}},\n"
                     f"  year = {{{year}}},\n  howpublished = {{arXiv:{aid}}}\n}}\n")
    manual = re.sub(r'title = \{(.*)\},\n  year', r'title = {{\1}},\n  year', MANUAL)
    open('refs.bib', 'w').write('% Generated by make_bib.py: arXiv entries come from the arXiv API.\n\n' + '\n'.join(parts) + manual)
    print(len(IDS), 'arXiv entries,', MANUAL.count('@misc'), 'manual entries')
