# CSC398, Winter 2026 - DND Policy Chatbot

## Overview

This repository contains the implementation of a **policy-grounded AI chatbot** designed to assist users in navigating and interpreting **Treasury Board of Canada Secretariat (TBS)** and **Department of National Defence (DND)** policies, directives, and guidelines.

The system leverages a **Retrieval-Augmented Generation (RAG)** architecture powered by the **Cohere LLM API** to provide fast, accurate, and explainable policy guidance. Rather than acting as a general-purpose language model, the chatbot strictly grounds all responses in unclassified policy documents and provides direct citations to source material.

The project explores how modern LLM-based systems can improve efficiency, consistency, and transparency in policy consultation, while explicitly handling **policy hierarchies** and **contradictory directives**.

---

## Key Features

- **Natural-Language Policy Querying**  
  Users can ask questions in plain English and receive policy-aligned responses.

- **Strict Retrieval-Augmented Generation (RAG)**  
  All answers are generated solely from retrieved policy excerpts—no hallucinated content.

- **Policy Hierarchy Awareness**  
  The system identifies and prioritizes higher-order policies (e.g., TBS over DND directives) when applicable.

- **Contradiction Detection**  
  When multiple policies provide overlapping or conflicting guidance, the chatbot explicitly surfaces and labels these conflicts.

- **Automated Citations & References**  
  Each response includes exact references to the policy documents and sections used.

- **Interactive Web Interface & Dashboard**  
  A browser-based chatbot UI and a dashboard for visualizing retrieval performance, coverage, and contradiction analysis.

---

## System Architecture

The system is composed of three main components:

### 1. Policy Ingestion & Indexing
- Collection of unclassified TBS and DND policy documents (PDF / HTML)
- Text cleaning, parsing, and semantic chunking
- Embedding generation using Cohere’s embedding models
- Storage in a vector database with metadata (document source, hierarchy level, section ID)

### 2. Retrieval & Reasoning Layer
- Semantic retrieval based on user queries
- Reranking and authority-aware filtering
- Detection of overlapping, hierarchical, or contradictory policy statements

### 3. LLM-Based Response Generation
- Grounded response generation using the Cohere API
- Structured prompts enforcing:
  - Evidence-only responses
  - Explicit citations
  - Hierarchy and contradiction awareness

