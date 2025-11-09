import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import AnimatedHero from '../components/AnimatedHero';
import FeatureCard from '../components/FeatureCard';

export default function Home() {
  const features = [
    {
      icon: '🪞',
      title: 'Aesthetic Alter Ego Generator',
      description: 'Transform mundane descriptions into poetic personas with structured AI responses'
    },
    {
      icon: '🧠',
      title: 'RAG-powered Archetype Blending',
      description: 'Intelligently mix traits from curated aesthetic datasets for unique identities'
    },
    {
      icon: '🎨',
      title: 'Moodboard Visualizer',
      description: 'Generate dreamlike visuals that match your persona'
    },
    {
      icon: '🔊',
      title: 'Spotify Soundtrack Mapper',
      description: 'AI-curated playlists that perfectly capture your fictional persona\'s vibe'
    },
    {
      icon: '🖼️',
      title: 'Downloadable Identity Cards',
      description: 'Beautiful, shareable SVG/PNG cards with complete aesthetic summaries'
    },
    {
      icon: '📱',
      title: 'Responsive Design',
      description: 'Seamless experience across desktop, tablet, and mobile devices'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  };

  return (
    <>
      <Head>
        <title>Muse.me — Your Life, Romanticized by AI</title>
        <meta name="description" content="Transform your mundane life into aesthetic alter egos powered by AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen bg-gradient-to-b from-muse-cream to-white">
        {/* Hero Section */}
        <AnimatedHero />

        {/* CTA Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex justify-center py-16"
        >
          <Link href="/generate">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              className="px-16 py-5 bg-gradient-to-r from-muse-rose via-pink-400 to-muse-rose text-white rounded-full font-serif text-xl font-semibold shadow-2xl hover:shadow-3xl transition-all duration-300 flex items-center gap-3"
            >
              <span>✨</span>
              <span>Begin Your Transformation</span>
              <span>→</span>
            </motion.button>
          </Link>
        </motion.div>

        {/* What is Muse.me */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="py-24 px-4 bg-white"
        >
          <div className="max-w-4xl mx-auto">
            <h2 className="text-5xl font-serif font-bold text-center text-muse-dark mb-6">
              What is Muse.me?
            </h2>
            <div className="section-divider"></div>
            <p className="text-xl text-gray-700 text-center font-serif mb-8 leading-relaxed px-4">
              <strong className="text-muse-rose">Muse.me</strong> is a poetic AI-powered web experience that transforms your mundane journal entries, bios, or daily routines into <strong className="text-muse-dark">aesthetic alter egos</strong> — complete with identity titles, fictional schedules, personality traits, moodboard prompts, Spotify playlists, and downloadable visual identity cards.
            </p>
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="text-center text-muse-rose font-serif italic text-lg bg-muse-cream bg-opacity-50 py-6 px-8 rounded-2xl max-w-2xl mx-auto"
            >
              <span className="text-2xl">💡</span>
              <p className="mt-2">Because life is more beautiful when viewed through an aesthetic lens.</p>
            </motion.div>
          </div>
        </motion.section>

        {/* Features Grid */}
        <motion.section
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="py-24 px-4 bg-gradient-to-b from-muse-cream/30 to-white"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-5xl font-serif font-bold text-center text-muse-dark mb-4">
              ✨ Features
            </h2>
            <div className="section-divider mb-16"></div>
            <div className="feature-grid">
              {features.map((feature, index) => (
                <motion.div key={index} variants={itemVariants}>
                  <FeatureCard feature={feature} />
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Core AI Concepts */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="py-20 px-4 bg-white"
        >
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-serif font-bold text-center text-muse-dark mb-12">
              🧪 Core AI Concepts
            </h2>

            <div className="space-y-12">
              {/* Prompt Engineering */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6 }}
              >
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-4">
                  🧾 Prompt Engineering
                </h3>
                <p className="text-gray-700 font-serif">
                  A sophisticated system prompt guides the model to respond with poetic yet structured output. The model is instructed to return aesthetic identity names, fictional daily routines, personality traits, vibe descriptions, moodboard prompts, and Spotify playlist recommendations.
                </p>
              </motion.div>

              {/* RAG */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-4">
                  🔍 Retrieval-Augmented Generation (RAG)
                </h3>
                <p className="text-gray-700 font-serif mb-4">
                  Muse.me uses a curated dataset of aesthetic archetypes. The RAG process:
                </p>
                <ul className="list-disc list-inside text-gray-700 font-serif space-y-2">
                  <li>Extracts keywords from user input</li>
                  <li>Retrieves matching archetypes from the database</li>
                  <li>Blends elements to construct unique fictional identities</li>
                </ul>
              </motion.div>

              {/* Structured Output */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-4">
                  📦 Structured Output
                </h3>
                <p className="text-gray-700 font-serif">
                  The model returns data in a carefully structured JSON format, enabling seamless frontend rendering and downstream function calling for image generation, playlist creation, and card design.
                </p>
              </motion.div>

              {/* Function Calling */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-4">
                  🔧 Function Calling
                </h3>
                <p className="text-gray-700 font-serif mb-4">
                  Muse.me dynamically calls external tools for real-world integration:
                </p>
                <ul className="list-disc list-inside text-gray-700 font-serif space-y-2">
                  <li>🎨 Moodboard Generation via Stable Diffusion</li>
                  <li>🔊 Playlist Fetching from Spotify API</li>
                  <li>🖼️ Identity Card Creation with SVG/Canvas</li>
                </ul>
              </motion.div>
            </div>
          </div>
        </motion.section>

        {/* Perfect For Section */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="py-20 px-4 bg-muse-cream bg-opacity-50"
        >
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-serif font-bold text-center text-muse-dark mb-12">
              🎭 Perfect For
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {[
                { emoji: '📝', title: 'Content Creators', desc: 'Seeking aesthetic persona inspiration' },
                { emoji: '✍️', title: 'Writers', desc: 'Exploring character development' },
                { emoji: '💭', title: 'Dreamers', desc: 'Who want to romanticize their daily existence' },
                { emoji: '👨‍💻', title: 'Developers', desc: 'Learning advanced AI integration patterns' }
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: idx * 0.1 }}
                  className="bg-white p-6 rounded-lg shadow-md"
                >
                  <div className="text-4xl mb-4">{item.emoji}</div>
                  <h3 className="text-xl font-serif font-bold text-muse-dark mb-2">{item.title}</h3>
                  <p className="text-gray-700 font-serif">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Tech Stack */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="py-20 px-4 bg-white"
        >
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-serif font-bold text-center text-muse-dark mb-12">
              🛠️ Tech Stack
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
              <div>
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-6">Backend</h3>
                <ul className="space-y-3 font-serif text-gray-700">
                  <li><strong>FastAPI</strong> — RESTful API with clean routing</li>
                  <li><strong>OpenRouter</strong> — Mixtral/LLaMA3 for creative generation</li>
                  <li><strong>Pydantic</strong> — Type validation & structured output</li>
                  <li><strong>Spotify API</strong> — Intelligent playlist matching</li>
                  <li><strong>Replicate API</strong> — Stable Diffusion image generation</li>
                </ul>
              </div>
              <div>
                <h3 className="text-2xl font-serif font-bold text-muse-dark mb-6">Frontend</h3>
                <ul className="space-y-3 font-serif text-gray-700">
                  <li><strong>Next.js</strong> — React framework with SSR</li>
                  <li><strong>TailwindCSS</strong> — Utility-first styling</li>
                  <li><strong>Framer Motion</strong> — Smooth animations</li>
                  <li><strong>SVG.js & html2canvas</strong> — Card generation & export</li>
                  <li><strong>Axios</strong> — HTTP client for API calls</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.section>

        {/* CTA Footer */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="py-20 px-4 bg-muse-rose text-white text-center"
        >
          <h2 className="text-4xl font-serif font-bold mb-6">
            Ready to Create Your Aesthetic Alter Ego?
          </h2>
          <p className="text-lg font-serif mb-8 opacity-90">
            Transform your mundane into the magnificent ✨
          </p>
          <Link href="/generate">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-12 py-4 bg-white text-muse-rose rounded-full font-serif text-lg font-bold hover:shadow-lg transition-shadow"
            >
              Begin Your Journey →
            </motion.button>
          </Link>
        </motion.section>
      </main>
    </>
  );
}
