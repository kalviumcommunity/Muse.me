# 🌸 Muse.me — Your Life, Romanticized by AI

<div align="center">

**Transform real-life monotony into aesthetic alternate lives through creative, intelligent automation.**

[✨ Live Demo](https://muse-me.vercel.app) • [ Report Bug](https://github.com/nishatayub/Muse.me/issues) • [💡 Request Feature](https://github.com/nishatayub/Muse.me/issues)

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

> **Unified Setup!** Everything runs with a single command - no separate backend needed!

### Prerequisites
- Node.js 16+ 
- OpenRouter API key ([Get it here](https://openrouter.ai/))
- Spotify API credentials ([Get them here](https://developer.spotify.com/dashboard))

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/nishatayub/Muse.me.git
   cd Muse.me
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local and add your API keys:
   # - OPENROUTER_API_KEY
   # - SPOTIFY_CLIENT_ID  
   # - SPOTIFY_CLIENT_SECRET
   ```

4. **Run the application**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   ```
   http://localhost:3000
   ```

That's it! 🎉

---

## 📁 Project Structure

```
Muse.me/
├── pages/
│   ├── api/              # Backend API Routes
│   │   ├── generate.js   # Main persona generation
│   │   └── health.js     # Health check
│   ├── index.js          # Landing page
│   ├── generate.js       # Generator UI
│   └── _app.js           # App wrapper
├── lib/                  # Backend Logic
│   ├── archetypes.js     # Archetype dataset & RAG
│   ├── prompts.js        # LLM prompts
│   ├── spotify.js        # Spotify API client
│   └── cardGenerator.js  # SVG card generator
├── components/           # React components
├── styles/              # Global styles
├── .env.local           # Environment variables (not in git)
└── package.json         # Dependencies
```

---

## 🌐 Deployment to Vercel

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Unified Next.js application"
   git push
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Select your repository
   - Vercel auto-detects Next.js!
   - Add environment variables:
     - `OPENROUTER_API_KEY`
     - `SPOTIFY_CLIENT_ID`
     - `SPOTIFY_CLIENT_SECRET`
   - Click "Deploy"

---

## 🧪 Core AI Concepts Demonstrated

### 🧾 1. Prompt Engineering

<details>
<summary><strong>Click to expand</strong></summary>

A sophisticated system prompt guides the model to respond with poetic yet structured output:

```
"You are a poetic, emotionally intelligent AI with a rich aesthetic vocabulary…"
```

**The prompt instructs the model to return:**
- ✨ Aesthetic identity name  
- 📅 Fictional daily routine (3–5 steps)  
- 🎭 3–5 personality traits  
- 🌸 Vibe description  
- 🎨 Moodboard prompts  
- 🎵 Spotify playlist name

</details>

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
2. **Retrieval Layer**: Matching archetypes are retrieved from the dataset
3. **Creative Blending**: Elements are recombined to construct unique fictional identities

</details>

---

---

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

---

### 🔧 4. Function Calling

<details>
<summary><strong>Click to expand</strong></summary>

Muse.me dynamically calls external tools for real-world integration:

| Action | Tool | Purpose |
|--------|------|---------|
| 🎨 **Moodboard Generation** | Replicate API (Stable Diffusion) | Create visual representations |
| 🔊 **Playlist Fetching** | Spotify API | Match music to personas |
| 🖼️ **Identity Card Creation** | SVG/Canvas | Generate shareable cards |

</details>

---

## 🛠️ Tech Stack

<div align="center">

**Core Technologies**

[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![Framer Motion](https://img.shields.io/badge/Framer_Motion-0055FF?style=for-the-badge&logo=framer&logoColor=white)](https://framer.com/motion)

</div>

**Frontend**
- **Next.js** → React framework with API routes and SSR
- **TailwindCSS** → Utility-first styling for aesthetic design
- **Framer Motion** → Smooth animations and page transitions
- **SVG.js & html2canvas** → Dynamic card generation and export

**Backend & AI**
- **Next.js API Routes** → Serverless API endpoints
- **OpenRouter** → Mixtral/LLaMA3 for creative generation
- **Replicate API** → AI image generation via Stable Diffusion
- **Spotify API** → Intelligent playlist matching

---

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

## 💰 Future Monetization & Growth

<div align="center">

| 💸 **Revenue Streams** | 🚀 **Growth Features** |
|------------------------|------------------------|
| 🌟 **Deluxe Dream Self** → Premium in-depth analysis | 🏆 **Aesthetic Leaderboards** → Viral content discovery |
| 🎙️ **AI Voiceovers** → Spoken identity cards (ElevenLabs) | 🖼️ **Gallery Builder** → Public persona collections |
| 📱 **Mobile App** → Subscription-based premium features | 🤖 **Bot Integration** → Telegram/Discord expansion |
| 🎨 **Custom Aesthetics** → User-generated archetype packs | 📊 **Analytics Dashboard** → Personal growth tracking |

</div>

## 🌱 Roadmap

- [ ] 🚀 **Live Demo Deployment** (Vercel)
- [ ] 🤖 **Telegram/Discord Bot Integration**
- [ ] 🎨 **Public Gallery** for browsing community personas
- [ ] 👥 **Archetype Contribution Panel** (community-driven dataset)
- [ ] 🌙 **Dream Decoder Spin-off** (sleep journal analysis)
- [ ] 📊 **"You vs Dream You" Mood Tracker**
- [ ] 🎵 **Advanced Spotify Integration** (playlist generation)
- [ ] 🌍 **Multi-language Support** (aesthetic diversity)

---


## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push to the branch** (`git push origin feature/AmazingFeature`)
5. **🔄 Open a Pull Request**

### Areas where we need help:
- 🎨 New aesthetic archetypes and datasets
- 🌍 Multi-language support and localization
- 🔧 Performance optimizations
- 📱 Mobile app development
- 🧪 Testing and quality assurance
- 📝 Documentation improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Credits & Acknowledgments

<div align="center">

**Built with 💖 by [Nishat Ayub](https://github.com/nishatayub)**

*For the love of aesthetic internet, creative code, and emotional resonance.*

### Special Thanks
- 🤖 **OpenRouter** for accessible AI model APIs
- 🎨 **Stable Diffusion Community** for democratizing AI art
- 🎵 **Spotify** for their amazing API and music ecosystem
- 🌸 **Aesthetic Internet** for endless inspiration

---

<sub>⭐ Star this repo if it sparks joy in your aesthetic soul! ⭐</sub>

</div>
