# 🌸 Project: **Muse.me**
_Your Life, Romanticized by AI_

A whimsical and aesthetic AI that transforms your mundane routines, bios, or journal rants into dreamy alternate lives — complete with personality archetypes, fictional routines, moodboard prompts, and even a Spotify soundtrack. Then it renders a downloadable **Aesthetic ID Card** to celebrate your new persona.

---

## 💖 Why It’s Irresistible

- **Escapism with Style**: Everyone dreams of being someone else. _Muse.me_ makes that dream art.
- **Highly Personal**: Based on your real life, but transformed into a romantic, aesthetic alter ego.
- **Incredibly Shareable**: Outputs moodboards + identity cards perfect for social media or digital journals.
- **Emotionally Sticky**: It feels good to be seen, even if by a fictional self.

---

## 🤖 How It Covers Each AI Topic

### **1. Prompting**

**System Prompt:**

> “You are a poetic, emotionally intelligent AI with a rich aesthetic vocabulary. Given a user’s real-life input (bio, daily schedule, journal entry), respond with:
> - An aesthetic identity title
> - A fictional daily routine (3–5 steps)
> - 3–5 personality traits
> - A short vibe description (2 lines)
> - Moodboard image prompts
> - A Spotify playlist name”

**User Input Examples:**
- “I'm a CS student who sleeps at 3AM and eats cold Maggi.”
- “My job is remote but my dreams aren’t.”
- “I just scrolled for 4 hours straight and now I feel like a potato.”

---

### **2. RAG (Retrieval-Augmented Generation)**

**Dataset:** Curated bank of aesthetics and vibes (e.g., _Cottagecore Hacker_, _Digital Mystic_, _Soft Doom Girl_), with associated:

```json
{
  "aesthetic": "Cyberpunk Poet",
  "traits": ["Melancholic", "Tech-addicted", "Nocturnal"],
  "moodboard_prompts": ["rainy neon skyline", "terminal code on vintage CRT", "punk jacket with cherry blossoms"]
}
```

**Augment Response:** Based on keywords in the user’s input, pull similar archetypes and recombine traits, routines, and imagery for creative personalization.

---

### **3. Structured Output**

```json
{
  "aesthetic_identity": "Cloudcore Catnapper",
  "routine": [
    "Wake up when the sky turns lavender.",
    "Make tea and think about past lives.",
    "Ignore responsibilities to tend digital houseplants.",
    "Nap to the sound of ambient rain playlists."
  ],
  "traits": ["Soft-spoken", "Detached", "Dreamy", "Highly online"],
  "vibe_description": "Lives on airplane mode. Thinks in Tumblr quotes.",
  "moodboard_prompts": ["window light on unmade bed", "old poetry books", "sunset on a computer screen"],
  "spotify_playlist": "Lofi for Living in My Head"
}
```

---

### **4. Function Calling**

**Dynamic Actions:**
- 🎨 **Call Stable Diffusion (via Replicate API)** to generate images from `moodboard_prompts`.
- 🔊 **Call Spotify API** to fetch a playlist by vibe.
- 🖼️ **Generate an Aesthetic ID Card** as a downloadable SVG or PNG with:
  - Aesthetic name
  - Traits
  - Vibe description
  - Moodboard collage or prompt list
- 🧠 **Log user data** (anonymized) to improve archetype suggestions over time.

---

## 🛠️ Tech Stack

### **Backend**
- `Python + FastAPI` (API routes + function calling)
- `OpenRouter` (LLMs like Mixtral/LLaMA3 for creativity)
- `Supabase` (RAG database + user storage)
- `Replicate API` (Stable Diffusion for image gen)
- `Spotify API` (vibe-matched playlists)

### **Frontend**
- `Next.js + TailwindCSS` (dreamy, responsive UI)
- `Framer Motion` (smooth transitions & reveal effects)
- `html2canvas` / `SVG.js` (for shareable identity cards)
- **Optional**: Telegram bot or Discord bot for aesthetic identities on the go

---

## 💸 Monetization & Growth Hooks

- **“Deluxe Dream Self”**: Pay to get long-form character diary entries or dream analyses.
- **Voiceovers**: Use ElevenLabs to narrate your new life in an aesthetic voice (e.g., anime girl, soft ASMR tone).
- **Aesthetic Leaderboards**: Most viewed, downloaded, or saved alternate lives.
- **Profile Builder**: Let users create a public gallery of their favorite alter egos.

---

## ✨ Spin-Off Ideas

- **Fiction.Me**: Full-on character generator for fiction writers based on your traits or prompts.
- **Dream Decoder AI**: Enter dreams → AI creates aesthetic meaning + story.
- **IRL You vs. Dream You Tracker**: A dual mood tracker: your real schedule vs. your ideal one.

---

## ✅ Why This Works

- **Zero Friction**: User just types their real-life mess — gets poetic identity in return.
- **Emotionally Viral**: People love sharing versions of themselves that feel beautiful.
- **Educational & Practical**: Teaches Prompting, RAG, Structured Output, Function Calling in a real-world app.

---

## 🔗 Coming Soon

- [ ] Live demo site
- [ ] Public moodboard generator
- [ ] User login + save your muses
- [ ] Community aesthetic gallery
