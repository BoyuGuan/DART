# DART: Disambiguation-Aware Reasoning for Video-guided Machine Translation

This repository contains the official code implementation of our ACL 2026 main conference paper:

**DART: Disambiguation-Aware Reasoning for Video-guided Machine Translation**

📄 **Paper:** https://aclanthology.org/2026.acl-long.352/


~~The code is currently being organized and will be open-sourced soon.~~

We apologize for the delay in releasing the code. The first author has recently been occupied with fall recruitment and job-search preparations, which has slowed down the code organization process. Thank you for your patience and understanding.

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{guan-etal-2026-dart,
    title = "{DART}: Disambiguation-Aware Reasoning for Video-guided Machine Translation",
    author = "Guan, Boyu  and
      Han, Chuang  and
      Zhao, Yang  and
      Zong, Chengqing",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.352/",
    doi = "10.18653/v1/2026.acl-long.352",
    pages = "7752--7772",
    ISBN = "979-8-89176-390-6",
    abstract = "Video-guided Machine Translation (VMT) seeks to enhance translation quality by incorporating contextual information derived from paired short video clips. However, many VMT samples are text-sufficient; even when visual information is needed, only minimal cues are required. Aiming to tackle these issues, we propose a novel framework \textbf{DART} (\textbf{D}isambiguation-\textbf{A}ware \textbf{R}easoning for Video-guided Machine \textbf{T}ranslation). Reinforcement learning is used to incorporate multimodal large language models' multimodal reasoning into VMT. The model dynamically switches between text-only processing and multimodal integration, contingent on the necessity of visual disambiguation. Furthermore, we present \textbf{TVRF} (\textbf{T}ranslation-oriented \textbf{V}ideo \textbf{R}elevance \textbf{F}iltering), a systematic pipeline for constructing training data based on multimodal relevance to translation. This pipeline filters samples where video information is translation-relevant, mitigating training collapse caused by video-irrelevant data in conventional VMT. Experimental results show that our approach improves multimodal information utilization in VMT, yielding gains in both translation quality and computational efficiency."
}
```

## 📬 Contact

If you have any questions, please feel free to contact:

**Boyu Guan** — [guanboyu2022@ia.ac.cn](mailto:guanboyu2022@ia.ac.cn)
