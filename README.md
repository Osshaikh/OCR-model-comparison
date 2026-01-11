# 🔍 OCR Model Comparison Tool

A comprehensive web application for comparing multiple OCR (Optical Character Recognition) and vision-based AI models side-by-side. This tool helps AI engineers and developers evaluate which model best fits their image-to-text extraction needs.

![OCR Comparison Tool](https://img.shields.io/badge/OCR-Model%20Comparison-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![Azure](https://img.shields.io/badge/Azure-App%20Service-0078D4)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo

**Production URL:** [https://ocr-comparison-app.azurewebsites.net](https://ocr-comparison-app.azurewebsites.net)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Models Compared](#-models-compared)
- [Model Comparison Matrix](#-model-comparison-matrix)
- [When to Use Which Model](#-when-to-use-which-model)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Scoring Methodology](#-scoring-methodology)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🎯 Overview

The **OCR Model Comparison Tool** is designed to help AI engineers make informed decisions when selecting OCR and vision models for their projects. Instead of manually testing each model separately, this tool allows you to:

- Upload a single image and compare results from **5 different AI models** simultaneously
- View extracted text side-by-side with quality scores
- Compare response times and cost estimates
- Make data-driven decisions based on real performance metrics

### Who is this for?

- **AI/ML Engineers** evaluating OCR solutions for production
- **Developers** building document processing applications
- **Data Scientists** analyzing handwritten or printed text extraction
- **Product Managers** comparing AI service costs and capabilities
- **Researchers** benchmarking vision models

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Multi-Model Comparison** | Compare 5 AI models with a single image upload |
| 📊 **Quality Scoring** | Automated scoring based on text quality, structure, and coherence |
| ⚡ **Speed Metrics** | Real-time response time tracking for each model |
| 💰 **Cost Estimation** | Approximate cost per request based on token usage |
| 🏆 **Winner Highlighting** | Automatic identification of best accuracy, fastest, and most cost-effective |
| 🔍 **Debug Mode** | Detailed logs for troubleshooting and understanding model behavior |
| 🚦 **Rate Limiting** | Built-in protection (4 requests/minute per IP) |
| 📱 **Responsive UI** | Modern, dark-themed interface that works on all devices |

---

## 🤖 Models Compared

### 1. Mistral Large 3 (Azure OpenAI)

| Attribute | Details |
|-----------|---------|
| **Provider** | Mistral AI (via Azure OpenAI) |
| **Type** | Large Language Model with Vision |
| **Parameters** | 123B |
| **Strengths** | Excellent multilingual support, strong reasoning, good at structured extraction |
| **Weaknesses** | Higher latency for complex documents |
| **Best For** | Multi-language documents, forms with mixed content |
| **Pricing** | ~$0.002/1K input, ~$0.006/1K output tokens |

**Pros:**
- ✅ Excellent at understanding document context
- ✅ Strong multilingual capabilities (100+ languages)
- ✅ Good at preserving document structure
- ✅ Handles handwritten text well

**Cons:**
- ❌ Slower than specialized OCR models
- ❌ Can be verbose in output
- ❌ Higher cost for high-volume processing

---

### 2. DeepSeek OCR (Azure GPU - A100)

| Attribute | Details |
|-----------|---------|
| **Provider** | DeepSeek (Self-hosted on Azure Container Apps) |
| **Type** | Vision-Language Model (3.3B parameters) |
| **Infrastructure** | NVIDIA A100 80GB GPU |
| **Strengths** | Fast inference, cost-effective for self-hosted |
| **Weaknesses** | Limited non-Latin script support |
| **Best For** | English documents, high-throughput scenarios |
| **Pricing** | ~$0.002/1K tokens (self-hosted cost) |

**Pros:**
- ✅ Very fast inference on GPU
- ✅ Good for English text extraction
- ✅ Cost-effective when self-hosted
- ✅ No API rate limits (self-managed)

**Cons:**
- ❌ Poor performance on Hindi/Indic scripts
- ❌ Sometimes outputs gibberish on complex layouts
- ❌ Requires GPU infrastructure management
- ❌ May add strange prefixes to output

---

### 3. GPT-5.2 (Azure OpenAI)

| Attribute | Details |
|-----------|---------|
| **Provider** | OpenAI (via Azure OpenAI Service) |
| **Type** | Multimodal Large Language Model |
| **Strengths** | State-of-the-art accuracy, excellent reasoning |
| **Weaknesses** | Highest cost, longer latency |
| **Best For** | Complex documents requiring understanding, not just extraction |
| **Pricing** | ~$0.01/1K input, ~$0.03/1K output tokens |

**Pros:**
- ✅ Highest accuracy across all document types
- ✅ Excellent at understanding context and intent
- ✅ Superior handling of complex layouts
- ✅ Great at extracting structured data (tables, forms)

**Cons:**
- ❌ Most expensive option
- ❌ Slower response times
- ❌ May over-interpret or add explanations
- ❌ Rate limits on Azure OpenAI

---

### 4. Gemini 3 Flash (Google AI)

| Attribute | Details |
|-----------|---------|
| **Provider** | Google DeepMind |
| **Type** | Multimodal Model (Flash variant) |
| **Strengths** | Extremely fast, very cost-effective |
| **Weaknesses** | Limited TIFF support, quota limits on free tier |
| **Best For** | High-volume processing, real-time applications |
| **Pricing** | See detailed pricing table below |

#### 💰 Gemini Flash Pricing for Image-to-Text (OCR)

| Model | Tier | Input (text/image/video) | Output (text) |
|-------|------|--------------------------|---------------|
| **Gemini 3 Flash Preview** | Free | Free of charge | Free of charge |
| | Paid | $0.50 / 1M tokens | $3.00 / 1M tokens |
| **Gemini 2.5 Flash** | Free | Free of charge | Free of charge |
| | Paid | $0.30 / 1M tokens | $2.50 / 1M tokens |
| **Gemini 2.5 Flash-Lite** | Free | Free of charge | Free of charge |
| | Paid | $0.10 / 1M tokens | $0.40 / 1M tokens |
| **Gemini 2.0 Flash** | Free | Free of charge | Free of charge |
| | Paid | $0.10 / 1M tokens | $0.40 / 1M tokens |

#### Cost per Image Estimate (~560 tokens/image)

| Model | Approx. Cost/Image | Best For |
|-------|-------------------|----------|
| **Gemini 2.5 Flash-Lite** | ~$0.00006 | High-volume, cost-sensitive OCR |
| **Gemini 2.0 Flash** | ~$0.00006 | Balanced performance/cost |
| **Gemini 2.5 Flash** | ~$0.00017 | Better accuracy, hybrid reasoning |
| **Gemini 3 Flash** | ~$0.00028 | Best quality, latest features |

> **Note:** Free tier includes generous rate limits (500-1500 RPD) but data may be used to improve Google products.

**Pros:**
- ✅ Blazingly fast response times
- ✅ Most cost-effective option
- ✅ Good accuracy for standard documents
- ✅ Generous free tier with all Flash models
- ✅ Multiple model variants for different needs

**Cons:**
- ❌ No TIFF image support
- ❌ Quota limits can be restrictive on free tier
- ❌ Less accurate on handwritten text
- ❌ May miss fine details

---

### 5. Azure Document Intelligence

| Attribute | Details |
|-----------|---------|
| **Provider** | Microsoft Azure |
| **Type** | Specialized OCR Service |
| **Strengths** | Purpose-built for OCR, highly accurate |
| **Weaknesses** | Limited to text extraction (no reasoning) |
| **Best For** | Production document processing, forms, invoices |
| **Pricing** | ~$0.001/page |

**Pros:**
- ✅ Purpose-built for document processing
- ✅ Excellent accuracy on printed text
- ✅ Fast and reliable
- ✅ Supports 300+ languages
- ✅ Pre-built models for invoices, receipts, IDs

**Cons:**
- ❌ No semantic understanding
- ❌ Struggles with complex handwriting
- ❌ Cannot answer questions about content
- ❌ Layout preservation can be inconsistent

---

## 📊 Model Comparison Matrix

| Model | Accuracy | Speed | Cost | Handwriting | Multi-language | Complex Layouts |
|-------|----------|-------|------|-------------|----------------|-----------------|
| **Mistral Large 3** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **DeepSeek OCR** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **GPT-5.2** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Gemini 3 Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Azure Doc Intel** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 When to Use Which Model

### Use Case Recommendations

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| **High-volume invoice processing** | Azure Doc Intelligence | Purpose-built, fast, cost-effective |
| **Handwritten notes/letters** | GPT-5.2 or Mistral Large 3 | Best at interpreting handwriting |
| **Real-time mobile app** | Gemini 3 Flash | Fastest response, lowest cost |
| **Multi-language documents** | Mistral Large 3 | 100+ language support |
| **Complex forms with tables** | GPT-5.2 | Best at structured extraction |
| **Budget-conscious projects** | Gemini 3 Flash | Extremely low cost |
| **Self-hosted requirement** | DeepSeek OCR | No external API dependencies |
| **Historical documents** | GPT-5.2 | Best at handling degraded text |
| **Medical/Legal documents** | Azure Doc Intelligence | Compliance-friendly, reliable |

### Decision Flowchart

```
Start
  │
  ├─► Need semantic understanding? ──► YES ──► GPT-5.2 or Mistral Large 3
  │                                    │
  │                                    NO
  │                                    │
  ├─► High volume (>10K pages/day)? ──► YES ──► Azure Doc Intelligence
  │                                    │
  │                                    NO
  │                                    │
  ├─► Real-time requirement? ──► YES ──► Gemini 3 Flash
  │                              │
  │                              NO
  │                              │
  ├─► Handwritten text? ──► YES ──► GPT-5.2
  │                        │
  │                        NO
  │                        │
  └─► Multi-language? ──► YES ──► Mistral Large 3
                         │
                         NO
                         │
                         └──► Azure Doc Intelligence (default choice)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                              │
│                    (Upload Image + Select Models)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure App Service                             │
│                 (Flask + Gunicorn + Rate Limiting)               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   /api/ocr  │  │  /static/*  │  │      /      │              │
│  │  (Limited)  │  │  (Assets)   │  │   (HTML)    │              │
│  └──────┬──────┘  └─────────────┘  └─────────────┘              │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model APIs (Parallel Calls)                 │
│                                                                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │
│  │ Azure OpenAI  │ │   Google AI   │ │Azure Doc Intel│          │
│  │(Mistral/GPT)  │ │ (Gemini 3)    │ │               │          │
│  └───────────────┘ └───────────────┘ └───────────────┘          │
│                                                                  │
│  ┌───────────────────────────────────┐                          │
│  │  Azure Container Apps (A100 GPU)  │                          │
│  │       DeepSeek OCR Model          │                          │
│  └───────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Azure subscription (for Azure-based models)
- Google AI API key (for Gemini)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Osshaikh/OCR-model-comparison.git
   cd OCR-model-comparison
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python server.py
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

### Environment Variables

Create a `.env` file with the following:

```env
# Mistral/GPT (Azure OpenAI)
MISTRAL_ENDPOINT=https://your-resource.openai.azure.com
MISTRAL_API_KEY=your-api-key
GPT52_API_KEY=your-api-key

# DeepSeek OCR (Azure Container Apps)
DEEPSEEK_OCR_GPU_ENDPOINT=https://your-container-app.azurecontainerapps.io

# Gemini (Google AI)
GEMINI_API_KEY=your-google-api-key

# Azure Document Intelligence
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=your-doc-intel-key
```

---

## ☁️ Deployment

### Deploy to Azure App Service

```bash
# Create resource group
az group create --name ocr-app-rg --location centralus

# Create App Service plan
az appservice plan create --name ocr-app-plan --resource-group ocr-app-rg --sku B1 --is-linux

# Create web app
az webapp create --name your-app-name --resource-group ocr-app-rg --plan ocr-app-plan --runtime "PYTHON:3.11"

# Configure startup command
az webapp config set --name your-app-name --resource-group ocr-app-rg --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 server:app"

# Set environment variables
az webapp config appsettings set --name your-app-name --resource-group ocr-app-rg --settings \
  MISTRAL_API_KEY="..." \
  GPT52_API_KEY="..." \
  GEMINI_API_KEY="..." \
  AZURE_DOC_INTELLIGENCE_ENDPOINT="..." \
  AZURE_DOC_INTELLIGENCE_KEY="..."

# Deploy
Compress-Archive -Path server.py, requirements.txt, static, startup.sh, .deployment -DestinationPath deploy.zip -Force
az webapp deploy --name your-app-name --resource-group ocr-app-rg --src-path deploy.zip --type zip
```

---

## 📡 API Reference

### POST /api/ocr

Process an image with selected OCR models.

**Request:**
```json
{
  "image": "base64-encoded-image-data",
  "image_type": "image/png",
  "models": ["mistral-large-3", "gpt-5.2", "azure-doc-intelligence"]
}
```

**Response:**
```json
{
  "results": [
    {
      "model": "mistral-large-3",
      "model_name": "Mistral Large 3",
      "text": "Extracted text content...",
      "response_time": 3.45,
      "quality_score": 92,
      "cost_estimate": 0.0012,
      "debug_logs": ["[10:30:45] Starting OCR request..."]
    }
  ],
  "best_accuracy": "gpt-5.2",
  "fastest": "gemini-3-flash",
  "most_cost_effective": "gemini-3-flash"
}
```

**Rate Limiting:** 4 requests per minute per IP address.

---

## 📈 Scoring Methodology

### Quality Score (50% of total)

- **Text Length**: Longer, meaningful extractions score higher
- **Word Count**: More words indicate better extraction
- **Structure**: Presence of paragraphs, lists, formatting
- **Coherence**: Detects gibberish or repeated characters

### Speed Score (30% of total)

- Relative to the fastest model in the comparison
- Faster response = higher score

### Cost Score (20% of total)

- Based on estimated token usage and model pricing
- Lower cost = higher score

---

## 🗺️ Roadmap

### 🔜 Coming Soon

| Model | Provider | Expected |
|-------|----------|----------|
| **Claude 3.5 Sonnet** | Anthropic | Q1 2026 |
| **Llama 3.2 Vision** | Meta (Azure) | Q1 2026 |
| **Qwen2-VL** | Alibaba | Q2 2026 |
| **Florence-2** | Microsoft | Q2 2026 |
| **GOT-OCR2** | StepFun | Q2 2026 |

### Planned Features

- [ ] PDF support (multi-page documents)
- [ ] Batch processing mode
- [ ] Export comparison results (CSV/JSON)
- [ ] Custom model endpoint support
- [ ] A/B testing mode
- [ ] Historical comparison tracking
- [ ] Webhook notifications
- [ ] API authentication

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Osama Shaikh**
- GitHub: [@Osshaikh](https://github.com/Osshaikh)

---

## 🙏 Acknowledgments

- Microsoft Azure for hosting infrastructure
- OpenAI, Google, Mistral AI, and DeepSeek for their amazing models
- The open-source community for Flask and related libraries

---

<p align="center">
  <b>⭐ Star this repository if you find it helpful!</b>
</p>
