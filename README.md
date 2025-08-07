# 🌸 Muse.me — Your Life, Romanticized by AI

<div align="center">

![Muse.me Logo](https://img.shields.io/badge/Mus### 🧾 1. Prompt Engineering

<details>
<summary><strong>Click to expand</strong></summary>

A sophisticated system prompt guides the model to respond with poetic yet structured output:

```
"You are a poetic, emotionally intelligent AI with a rich aesthetic vocabulary…"
```

**The prompt instructs the model to return:**
- ✨ Aesthetic identity name  
- 📅 Fictional daily routine (3–5 steps)  
- � 3–5 personality traits  
- �🌸 Vibe description  
- 🎨 Moodboard prompts  
- 🎵 Spotify playlist name

</details>yle=for-the-badge)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-13+-black.svg?style=for-the-badge&logo=next.js)](https://nextjs.org)

**Transform real-life monotony into aesthetic alternate lives through creative, intelligent automation.**

[✨ Live Demo](https://muse-me.vercel.app) • [📖 Documentation](docs/README.md) • [🐛 Report Bug](https://github.com/kalviumcommunity/Muse.me/issues) • [💡 Request Feature](https://github.com/kalviumcommunity/Muse.me/issues)

</div>

---

## 🎯 What is Muse.me?

**Muse.me** is a poetic AI-powered web experience that transforms your mundane journal entries, bios, or daily routines into **aesthetic alter egos** — complete with identity titles, fictional schedules, personality traits, moodboard prompts, Spotify playlists, and downloadable visual identity cards.

> 💡 **Why?** Because life is more beautiful when viewed through an aesthetic lens. Turn your "I woke up, had coffee, worked" into "Velvet Morning Dreamer: Dawn ritual curator who finds poetry in porcelain and productivity."

### 🎭 Perfect For
- **Content Creators** seeking aesthetic persona inspiration
- **Writers** exploring character development
- **Dreamers** who want to romanticize their daily existence
- **Developers** learning advanced AI integration patterns

This project serves as both a delightful user tool **and** a robust demonstration of key programming concepts including Prompt Engineering, RAG, Structured Output, and Function Calling.

---

## ✨ Features

<table>
<tr>
<td align="center" width="33%">

### 🪞 Aesthetic Alter Ego Generator
Transform mundane descriptions into poetic personas with structured AI responses

</td>
<td align="center" width="33%">

### 🧠 RAG-powered Archetype Blending
Intelligently mix traits from curated aesthetic datasets for unique identities

</td>
<td align="center" width="33%">

### 🎨 Moodboard Visualizer
Generate dreamlike visuals via Stable Diffusion that match your persona

</td>
</tr>
<tr>
<td align="center" width="33%">

### 🔊 Spotify Soundtrack Mapper
AI-curated playlists that perfectly capture your fictional persona's vibe

</td>
<td align="center" width="33%">

### 🖼️ Downloadable Identity Cards
Beautiful, shareable SVG/PNG cards with complete aesthetic summaries

</td>
<td align="center" width="33%">

### 📱 Responsive Design
Seamless experience across desktop, tablet, and mobile devices

</td>
</tr>
</table>

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Supabase account
- OpenRouter API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kalviumcommunity/Muse.me.git
   cd Muse.me
   ```

2. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   cp .env.example .env
   # Add your API keys to .env
   uvicorn main:app --reload
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Visit `http://localhost:3000` and start creating your aesthetic alter egos! ✨**

---

## 🧪 Core AI Concepts

### 🧾 1. Prompt Engineering

A structured system prompt guides the model to respond with poetic yet structured output based on user input.  
> “You are a poetic, emotionally intelligent AI with a rich aesthetic vocabulary…”

It instructs the model to return:

- Aesthetic identity name  
- Fictional daily routine (3–5 steps)  
- 3–5 personality traits  
- Vibe description  
- Moodboard prompts  
- Spotify playlist name

---

### 🔍 2. Retrieval-Augmented Generation (RAG)

<details>
<summary><strong>Click to expand</strong></summary>

Muse.me uses a curated internal dataset of aesthetic archetypes:

```json
{
  "aesthetic": "Cyberpunk Poet",
  "traits": ["Melancholic", "Tech-addicted", "Nocturnal"],
  "moodboard_prompts": ["neon skyline", "vintage CRT", "punk jacket with blossoms"]
}
```

**RAG Process:**
1. **Keyword Extraction**: User inputs are parsed for relevant keywords
2. **Retrieval Layer**: Matching archetypes are retrieved from Supabase
3. **Creative Blending**: Elements are recombined to construct unique fictional identities

</details>

### 📦 3. Structured Output

<details>
<summary><strong>Click to expand</strong></summary>

The model returns data in a clearly structured JSON format:

```json
{
  "aesthetic_identity": "Cloudcore Catnapper",
  "routine": ["Morning meditation with chamomile tea", "Afternoon cloud-watching sessions"],
  "traits": ["Dreamy", "Introspective", "Gentle"],
  "vibe_description": "Soft mornings and cotton candy skies",
  "moodboard_prompts": ["fluffy clouds", "pastel sunrise", "cozy blankets"],
  "spotify_playlist": "Lofi for Living in My Head"
}
```

This enables seamless frontend rendering and downstream function calling.

</details>

### 🔧 4. Function Calling

<details>
<summary><strong>Click to expand</strong></summary>

Muse.me dynamically calls external tools for real-world integration:

| Action | Tool | Purpose |
|--------|------|---------|
| 🎨 **Moodboard Generation** | Replicate API (Stable Diffusion) | Create visual representations |
| 🔊 **Playlist Fetching** | Spotify API | Match music to personas |
| 🖼️ **Identity Card Creation** | SVG.js / html2canvas | Generate shareable cards |
| 🧠 **Archetype Enhancement** | Supabase Logging | Improve future outputs |

</details>

---

## 🛠️ Tech Stack

<div align="center">

### Backend
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)

**FastAPI** → RESTful API with clean routing and async responses  
**OpenRouter** → Mixtral/LLaMA3 for creative generation  
**Supabase** → Archetype storage (RAG) + user metadata  
**Replicate API** → AI image generation via Stable Diffusion  
**Spotify API** → Intelligent playlist matching  

### Frontend
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![Framer Motion](https://img.shields.io/badge/Framer_Motion-0055FF?style=for-the-badge&logo=framer&logoColor=white)](https://framer.com/motion)

**Next.js** → React framework with SSR and optimized performance  
**TailwindCSS** → Utility-first styling for aesthetic design  
**Framer Motion** → Smooth animations and page transitions  
**SVG.js & html2canvas** → Dynamic card generation and export  

</div>

## ✅ Project Goals & Evaluation

<table>
<tr>
<td width="33%">

### ✔️ **Correctness**
- Structured, valid output for every input
- Accurate API integrations (images, playlists, cards)
- Context-aware content generation
- Reliable persona-to-content matching

</td>
<td width="33%">

### ⚡ **Efficiency**
- Optimized prompts reduce token usage
- Async API endpoints for fast responses
- Smart caching for repeated requests
- Minimal redundant API calls

</td>
<td width="33%">

### 📈 **Scalability**
- Modular, extensible architecture
- Rate limiting & queuing strategies
- Efficient database indexing
- Multi-language support ready

</td>
</tr>
</table>

## � Future Monetization & Growth

<div align="center">

| 💸 **Revenue Streams** | 🚀 **Growth Features** |
|------------------------|------------------------|
| 🌟 **Deluxe Dream Self** → Premium in-depth analysis | 🏆 **Aesthetic Leaderboards** → Viral content discovery |
| 🎙️ **AI Voiceovers** → Spoken identity cards (ElevenLabs) | 🖼️ **Gallery Builder** → Public persona collections |
| 📱 **Mobile App** → Subscription-based premium features | 🤖 **Bot Integration** → Telegram/Discord expansion |
| 🎨 **Custom Aesthetics** → User-generated archetype packs | 📊 **Analytics Dashboard** → Personal growth tracking |

</div>

## 🌱 Roadmap

- [ ] 🚀 **Live Demo Deployment** (Vercel + Railway)
- [ ] 🤖 **Telegram/Discord Bot Integration**
- [ ] 🎨 **Public Gallery** for browsing community personas
- [ ] 👥 **Archetype Contribution Panel** (community-driven dataset)
- [ ] 🌙 **Dream Decoder Spin-off** (sleep journal analysis)
- [ ] 📊 **"You vs Dream You" Mood Tracker**
- [ ] 🎵 **Advanced Spotify Integration** (playlist generation)
- [ ] 🌍 **Multi-language Support** (aesthetic diversity)

## 📁 Project Structure

```
Muse.me/
├── 🔧 backend/
│   ├── main.py              # FastAPI routes & middleware
│   ├── llm_engine.py        # Prompt engineering & structured output
│   ├── rag_layer.py         # Supabase integration & matching logic
│   ├── api/
│   │   ├── spotify.py       # Playlist generation
│   │   ├── replicate.py     # Image generation
│   │   └── cards.py         # Identity card creation
│   └── requirements.txt
├── 🎨 frontend/
│   ├── pages/
│   │   ├── index.js         # Landing page
│   │   ├── generate.js      # Main generation interface
│   │   └── gallery.js       # Public persona gallery
│   ├── components/
│   │   ├── PersonaCard.js   # Identity card component
│   │   ├── MoodboardGrid.js # Visual layout
│   │   └── PlaylistEmbed.js # Spotify integration
│   ├── api/                 # Next.js API routes
│   ├── utils/               # Helper functions
│   └── package.json
├── 📚 docs/
│   ├── API.md              # API documentation
│   ├── CONTRIBUTING.md     # Contribution guidelines
│   └── DEPLOYMENT.md       # Deployment instructions
├── .env.example            # Environment variables template
├── .gitignore
└── README.md
```
## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **� Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push to the branch** (`git push origin feature/AmazingFeature`)
5. **🔄 Open a Pull Request**

### Areas where we need help:
- 🎨 New aesthetic archetypes and datasets
- 🌍 Multi-language support and localization
- 🔧 Performance optimizations
- 📱 Mobile app development
- 🧪 Testing and quality assurance

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Muse.me** is an open-source project for creative exploration and learning.

## ✨ Credits & Acknowledgments

<div align="center">

**Built with 💖 by [Nisha Tayub](https://github.com/nishatayub)**

*For the love of aesthetic internet, creative code, and emotional resonance.*

### Special Thanks
- 🤖 **OpenAI** for inspiring creative AI applications
- 🎨 **Stable Diffusion Community** for democratizing AI art
- 🎵 **Spotify** for their amazing API and music ecosystem
- 🌸 **Aesthetic Internet** for endless inspiration

---

<sub>⭐ Star this repo if it sparks joy in your aesthetic soul! ⭐</sub>

</div>







